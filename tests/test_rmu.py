import re
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from pydantic import ValidationError
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from config.training_config import build_orchestrator_config
from orchestrator import build_unlearning_run
from schemas import FinalTrainingConfigRequest
from tests.test_forget_only import Tokenizer


def tiny_model():
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=16, pad_token_id=0,
    ))


def request(overrides=None, **kwargs):
    return FinalTrainingConfigRequest(
        model_name='test', method='full', unlearning_method='rmu',
        gpu_ids=[0], forget_set_path='forget.parquet',
        hyperparams=dict(
            general=dict(max_steps=1, learning_rate=0.001, context_length=8),
            optimization=dict(batch_size=1, grad_accum=1, weight_decay=0),
            schedule=dict(save_steps=1), memory={}, method=overrides or {},
        ), **kwargs,
    )


class RMUConfigTests(unittest.TestCase):
    def test_saved_run_preserves_original_adapter_and_layer_configuration(self):
        from orchestrator import run_orchestrator
        from schemas import LoadMethod, LoraSettings
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            original_base = tiny_model()
            old_adapter = get_peft_model(copy.deepcopy(original_base), LoraConfig(
                r=2, lora_alpha=4, target_modules=['q_proj'], task_type='CAUSAL_LM',
            ))
            with torch.no_grad():
                for name, param in old_adapter.named_parameters():
                    if 'lora_' in name:
                        param.fill_(0.2)
            old_adapter.save_pretrained(root / 'old_adapter')
            expected_base = old_adapter.merge_and_unload()
            pd.DataFrame({'question': ['Q'], 'answer': ['ab']}).to_parquet(root / 'forget.parquet')
            req = request({'layers_to_transform': [1]})
            req.method = LoadMethod.lora
            req.adaptor_path = str(root / 'old_adapter')
            req.forget_set_path = str(root / 'forget.parquet')
            req.hyperparams.lora_settings = LoraSettings(target_modules=['q_proj'])
            config = build_orchestrator_config(req)
            config['output_dir'] = str(root / 'run')

            class SavingTokenizer(Tokenizer):
                def save_pretrained(self, path):
                    pass

            with patch('model.model_loader.load_base_model', side_effect=lambda *a, **k: copy.deepcopy(original_base)), patch(
                'model.model_loader.load_tokenizer', return_value=SavingTokenizer()
            ), patch('orchestrator.build_training_args', return_value=TrainingArguments(
                output_dir=str(root / 'run'), use_cpu=True, report_to=[],
                max_steps=1, save_strategy='no', disable_tqdm=True,
                remove_unused_columns=False,
            )):
                result = run_orchestrator(config)
            # Final output must include the old merged adapter as well as RMU changes.
            self.assertTrue((Path(result['output_dir']) / 'config.json').is_file())
            reloaded = AutoModelForCausalLM.from_pretrained(result['output_dir'])
            torch.testing.assert_close(
                reloaded.model.layers[0].self_attn.q_proj.weight,
                expected_base.model.layers[0].self_attn.q_proj.weight,
            )
            self.assertEqual(result['trainable_parameters'], 128)

    def test_layer_changes_regenerate_regex_and_default_module(self):
        params = build_orchestrator_config(request({
            'layers_to_transform': [2, 0], 'steering_coeff': 7,
            'trainable_params_regex': 'stale',
        }))['hyperparams']['method']
        self.assertEqual(params['layers_to_transform'], [0, 2])
        self.assertEqual(params['steering_coeff'], 7)
        self.assertTrue(re.search(params['module_regex'], 'model.layers.2'))
        for name in ('model.layers.0.self_attn.q_proj.weight',
                     'base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight'):
            self.assertIsNotNone(re.search(params['trainable_params_regex'], name))
        self.assertIsNone(re.search(params['trainable_params_regex'], 'model.layers.20.weight'))

    def test_reject_invalid_or_wrong_method_parameters(self):
        for values in ({'layers_to_transform': []}, {'layers_to_transform': [-1]},
                       {'layers_to_transform': [1, 1]}, {'layers_to_transform': [True]},
                       {'layers_to_transform': [1.5]}, {'module_regex': '['},
                       {'module_regex': ''}, {'steering_coeff': -1},
                       {'steering_coeff': float('nan')}, {'beta': 0.1}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                request(values)

    def test_lora_loader_attaches_only_requested_layers_and_targets(self):
        from model.model_loader import load_lora_model
        with patch('model.model_loader.load_base_model', return_value=tiny_model()), patch(
            'model.model_loader.load_tokenizer', return_value=Tokenizer()
        ):
            model, _, _ = load_lora_model('test', 0,
                target_modules=['q_proj'], layers_to_transform=[0, 2])
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all('q_proj.lora_' in name for name in names))
        self.assertTrue(all('.layers.0.' in name or '.layers.2.' in name for name in names))
        self.assertEqual(model.peft_config['default'].layers_to_transform, [0, 2])


class RMUTrainingTests(unittest.TestCase):
    def test_truncated_prompts_keep_batch_shapes(self):
        from unlearning.data_helpers.data_module import ForgetOnlyDataset
        data = pd.DataFrame({'question': ['Q' * 20], 'answer': ['ab']})
        ids, labels, mask = ForgetOnlyDataset(data, Tokenizer(), 8)[0]
        self.assertEqual(ids.shape, (8,))
        self.assertEqual(labels.tolist(), [-100] * 8)
        self.assertEqual(mask.shape, (8,))

    def test_bad_module_or_layer_selection_fails_before_training(self):
        from unlearning.rmu.trainer import RMUForgetOnlyTrainer
        for params, message in (
            ({'module_regex': r'.*layers\.99$', 'layers_to_transform': [1]}, 'exactly one'),
            ({'module_regex': r'.*layers\.[01]$', 'layers_to_transform': [1]}, 'exactly one'),
            ({'module_regex': r'.*layers\.2$', 'layers_to_transform': [5]}, 'No trainable parameters'),
        ):
            with self.subTest(params=params), tempfile.TemporaryDirectory() as output:
                with self.assertRaisesRegex(ValueError, message):
                    RMUForgetOnlyTrainer(model=tiny_model(), args=TrainingArguments(
                        output_dir=output, use_cpu=True, report_to=[],
                    ), **params)

    def test_hook_cleanup_on_model_failure(self):
        from unlearning.rmu.rmu_utils import forward_with_cache
        model = tiny_model()
        module = model.model.layers[1]
        with patch.object(model, 'forward', side_effect=RuntimeError('failed forward')):
            with self.assertRaisesRegex(RuntimeError, 'failed forward'):
                forward_with_cache(model, {'input_ids': None, 'attention_mask': None}, module)
        self.assertFalse(module._forward_hooks)

    def test_strengths_scale_activation_losses_and_evaluation_uses_lm_loss(self):
        from unlearning.rmu.trainer import RMUTrainer
        data = pd.DataFrame({'question': ['Q'], 'answer': ['ab']})
        dataset, collator, _, kwargs, _ = build_unlearning_run(
            'rmu', data, data, Tokenizer(), 8,
            method_hyperparams={'layers_to_transform': [1]},
        )
        with tempfile.TemporaryDirectory() as output:
            model = tiny_model()
            trainer = RMUTrainer(model=model, args=TrainingArguments(
                output_dir=output, use_cpu=True, report_to=[],
            ), **kwargs)
            batch = collator([dataset[0]])
            # Change the policy after the snapshot so the retain term is nonzero.
            with torch.no_grad():
                model.model.layers[1].self_attn.v_proj.weight.add_(0.1)
            trainer.gamma, trainer.alpha = 1, 0
            forget = trainer.compute_loss(model, batch)
            trainer.gamma, trainer.alpha = 0, 1
            retain = trainer.compute_loss(model, batch)
            self.assertGreater(retain.item(), 0)
            trainer.gamma, trainer.alpha = 2, 3
            loss = trainer.compute_loss(model, batch)
            torch.testing.assert_close(loss, 2 * forget + 3 * retain)
            from unlearning.data_helpers.collators import ForgetCollator
            eval_batch = ForgetCollator([dataset[0][0]])
            expected = model(**eval_batch).loss.detach()
            eval_loss, _, _ = trainer.prediction_step(model, eval_batch, True)
            torch.testing.assert_close(eval_loss, expected)

    def test_training_both_data_modes_and_parameter_isolation(self):
        data = pd.DataFrame({'question': ['Q', 'R'], 'answer': ['ab', 'cd']})
        for with_retain in (False, True):
            for lora in (False, True):
                with self.subTest(retain=with_retain, lora=lora), tempfile.TemporaryDirectory() as output:
                    params = build_orchestrator_config(request({
                        'layers_to_transform': [1], 'module_regex': r'.*layers\.2$',
                        'steering_coeff': 7,
                    }))['hyperparams']['method']
                    dataset, collator, cls, kwargs, run_type = build_unlearning_run(
                        'rmu', data, data if with_retain else None, Tokenizer(), 8,
                        gamma=2, alpha=0.5, method_hyperparams=params,
                    )
                    self.assertEqual(run_type, 'rmu_forget_retain' if with_retain else 'rmu_forget_only')
                    model = tiny_model()
                    if lora:
                        model = get_peft_model(model, LoraConfig(
                            r=2, lora_alpha=4, target_modules=['q_proj'],
                            layers_to_transform=[1], task_type='CAUSAL_LM',
                        ))
                    before = {name: p.detach().clone() for name, p in model.named_parameters()}
                    trainer = cls(model=model, args=TrainingArguments(
                        output_dir=output, use_cpu=True, report_to=[], max_steps=2,
                        per_device_train_batch_size=1, gradient_accumulation_steps=2,
                        learning_rate=0.01, save_strategy='no', disable_tqdm=True,
                        remove_unused_columns=False,
                    ), train_dataset=dataset, data_collator=collator, **kwargs)
                    self.assertFalse(trainer.model_accepts_loss_kwargs)
                    if with_retain:
                        self.assertIsNotNone(trainer.ref_model)
                        ref_before = [p.detach().clone() for p in trainer.ref_model.parameters()]
                    else:
                        self.assertIsNone(trainer.ref_model)
                    result = trainer.train()
                    self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
                    changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
                    self.assertTrue(changed)
                    self.assertTrue(all('.layers.1.' in name for name in changed))
                    if lora:
                        self.assertTrue(all('.lora_' in name for name in changed))
                    self.assertAlmostEqual(trainer._control_vec.norm().item(), 7, places=5)
                    if with_retain:
                        self.assertFalse(trainer.ref_model.training)
                        for old, p in zip(ref_before, trainer.ref_model.parameters()):
                            self.assertTrue(torch.equal(old, p))
                            self.assertFalse(p.requires_grad)
                            self.assertIsNone(p.grad)

    def test_masked_loss_ignores_prompt_and_padding(self):
        from unlearning.rmu.rmu_utils import masked_mse_over_tokens
        activation = torch.tensor([[[100.0, 100.0], [2.0, 4.0], [100.0, 100.0]]])
        mask = torch.tensor([[False, True, False]])
        self.assertEqual(masked_mse_over_tokens(activation, torch.zeros_like(activation), mask).item(), 10)
        self.assertEqual(masked_mse_over_tokens(activation, activation, mask).item(), 0)
        self.assertEqual(masked_mse_over_tokens(activation, torch.zeros_like(activation), ~torch.ones_like(mask)).item(), 0)


if __name__ == '__main__':
    unittest.main()
