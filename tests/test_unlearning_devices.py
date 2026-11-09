import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from orchestrator import configure_training_devices
from unlearning.dpo.trainer import DPOTrainer
from unlearning.npo.trainer import NPOTrainer


class UnlearningDeviceTests(unittest.TestCase):
    def test_reference_methods_use_first_selected_gpu(self):
        for method in ('dpo', 'npo', 'rmu'):
            for gpu in ({'gpu_ids': [6, 2]}, {'gpu_id': 6}):
                with self.subTest(method=method, gpu=gpu), patch.dict(os.environ, {}, clear=True):
                    configure_training_devices({'gpu': gpu, 'unlearning': {'method': method}})
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '6')
                    self.assertEqual(os.environ['UL_MODEL_DEVICE_MAP'], 'cuda:0')
                    self.assertEqual(gpu.get('gpu_ids', [6]), [6, 2] if 'gpu_ids' in gpu else [6])

    def test_other_methods_keep_multiple_gpus(self):
        for method in ('grad_ascent', 'grad_diff', 'simnpo'):
            with self.subTest(method=method), patch.dict(os.environ, {}, clear=True):
                configure_training_devices({'gpu': {'gpu_ids': [6, 2]}, 'unlearning': {'method': method}})
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '6,2')
                self.assertEqual(os.environ['UL_MODEL_DEVICE_MAP'], 'balanced')

    def test_reference_methods_reject_distributed_launch(self):
        for method in ('dpo', 'npo', 'rmu'):
            with self.subTest(method=method), patch.dict(os.environ, {'WORLD_SIZE': '2'}):
                with self.assertRaisesRegex(ValueError, 'single training process'):
                    configure_training_devices({'gpu': {'gpu_ids': [6, 2]}, 'unlearning': {'method': method}})

    def make_trainer(self, trainer_cls):
        trainer = object.__new__(trainer_cls)
        trainer.accelerator = SimpleNamespace(
            device=torch.device('cpu'), num_processes=1,
            prepare_model=lambda model, **kwargs: model,
        )
        trainer.args = SimpleNamespace(n_gpu=0)
        trainer.is_deepspeed_enabled = False
        return trainer

    def test_reference_is_independent_frozen_and_colocated(self):
        for trainer_cls in (DPOTrainer, NPOTrainer):
            with self.subTest(trainer=trainer_cls.__name__):
                policy = torch.nn.Linear(2, 2)
                reference = self.make_trainer(trainer_cls)._prepare_ref_model(policy)
                self.assertFalse(reference.training)
                for original, copied in zip(policy.parameters(), reference.parameters()):
                    self.assertEqual(original.device, copied.device)
                    self.assertTrue(original.requires_grad)
                    self.assertFalse(copied.requires_grad)
                    self.assertNotEqual(original.data_ptr(), copied.data_ptr())
                    torch.testing.assert_close(original, copied)

    def test_reference_trainers_consolidate_policy_with_multiple_gpus_reported(self):
        for trainer_cls in (DPOTrainer, NPOTrainer):
            trainer = self.make_trainer(trainer_cls)
            trainer.args._n_gpu = 2
            model = torch.nn.Linear(2, 2)
            model.hf_device_map = {"weight": "cuda:0", "bias": "cuda:1"}
            reference = trainer._prepare_ref_model(model)
            self.assertEqual(trainer.args._n_gpu, 0)  # CPU test runtime
            self.assertFalse(trainer.is_model_parallel)
            self.assertEqual(model.hf_device_map, {"": "cpu"})
            self.assertEqual(reference.hf_device_map, {"": "cpu"})
            self.assertTrue(all(p.device.type == "cpu" for p in reference.parameters()))
