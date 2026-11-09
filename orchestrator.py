import os
import inspect
from typing import Any, Dict

from unlearning.methods import (  # noqa: F401
    DEFAULT_UNLEARNING_METHOD,
    FORGET_ONLY_METHODS,
    RETAIN_REQUIRED_METHODS,
    UNLEARNING_METHOD_ARGS,
)


def build_training_args(api_config: Dict[str, Any]):
    from transformers import TrainingArguments

    hp = api_config["hyperparams"]
    output_dir = api_config.get("output_dir", "outputs/run")

    kwargs = {
        "output_dir": output_dir,
        "learning_rate": float(hp["general"]["learning_rate"]),
        "per_device_train_batch_size": hp["optimization"]["batch_size"],
        "save_strategy": "steps",
        "save_steps": hp["schedule"]["save_steps"],
        "weight_decay": hp["optimization"]["weight_decay"],
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 10,
        "label_names": ["labels"],
        "bf16": True,
        "gradient_accumulation_steps": hp["optimization"]["grad_accum"],
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
    }

    if hp["general"].get("max_steps") is not None:
        kwargs["max_steps"] = hp["general"]["max_steps"]

    if hp["general"].get("epochs") is not None:
        kwargs["num_train_epochs"] = hp["general"]["epochs"]

    # transformers changed eval_strategy/evaluation_strategy across versions
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"

    return TrainingArguments(**kwargs)


def build_trainer_kwargs(
    model,
    tokenizer,
    training_args,
    train_dataset,
    data_collator,
):
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    # transformers newer versions use processing_class instead of tokenizer
    trainer_sig = inspect.signature(__import__("transformers").Trainer.__init__)

    if "processing_class" in trainer_sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return kwargs


def build_unlearning_run(method, forget_df, retain_df, tokenizer, context_length, gamma=1.0, alpha=1.0, method_hyperparams=None):
    """Resolve a method name to its dataset, collator, trainer class and trainer args."""

    from unlearning.data_helpers.collators import (
        DpoForgetCollator,
        DpoRetainCollator,
        ForgetCollator,
        RetainCollator,
    )
    from unlearning.data_helpers.data_module import (
        ForgetOnlyDataset,
        ForgetRetainDataset,
        IdkForgetOnlyDataset,
        IdkForgetRetainDataset,
    )
    from unlearning.dpo.trainer import DPOForgetOnlyTrainer, DPOTrainer
    from unlearning.ga.trainer import GradAscentTrainer
    from unlearning.gd.trainer import GradDiffTrainer
    from unlearning.npo.trainer import NPOForgetOnlyTrainer, NPOTrainer
    from unlearning.rmu.trainer import RMUForgetOnlyTrainer, RMUTrainer
    from unlearning.rmu.config import resolve_rmu_params
    from unlearning.snpo.trainer import (
        SimNPOForgetOnlyTrainer,
        SimNPOForgetRetainTrainer,
    )

    if method not in UNLEARNING_METHOD_ARGS:
        raise ValueError(
            f"Unknown unlearning method '{method}'. "
            f"Expected one of: {sorted(UNLEARNING_METHOD_ARGS)}."
        )

    if method in RETAIN_REQUIRED_METHODS and retain_df is None:
        raise ValueError(f"Unlearning method '{method}' requires a retain set.")

    trainer_args = dict(UNLEARNING_METHOD_ARGS[method], gamma=gamma, alpha=alpha)
    for key, value in (method_hyperparams or {}).items():
        if key not in trainer_args or key in {"gamma", "alpha", "retain_loss_type"}:
            raise ValueError(f"Unsupported hyperparameter '{key}' for {method}.")
        trainer_args[key] = value
    if method == "rmu":
        trainer_args.update(resolve_rmu_params(method_hyperparams))
    dataset_kwargs = {
        "tokenizer": tokenizer,
        "max_length": context_length,
        "question_key": "question",
        "answer_key": "answer",
    }

    forget_only = method in FORGET_ONLY_METHODS or retain_df is None

    if forget_only:
        if method == "dpo":
            train_dataset = IdkForgetOnlyDataset(forget_data=forget_df, **dataset_kwargs)
            data_collator = DpoForgetCollator
        else:
            train_dataset = ForgetOnlyDataset(forget_data=forget_df, **dataset_kwargs)
            data_collator = ForgetCollator
        trainer_cls = {
            "grad_ascent": GradAscentTrainer,
            "simnpo": SimNPOForgetOnlyTrainer,
            "npo": NPOForgetOnlyTrainer,
            "dpo": DPOForgetOnlyTrainer,
            "rmu": RMUForgetOnlyTrainer,
        }[method]
        trainer_args.pop("retain_loss_type", None)
    elif method == "dpo":
        train_dataset = IdkForgetRetainDataset(
            forget_data=forget_df, retain_data=retain_df, **dataset_kwargs
        )
        data_collator = DpoRetainCollator
        trainer_cls = DPOTrainer
    else:
        train_dataset = ForgetRetainDataset(
            forget_data=forget_df, retain_data=retain_df, **dataset_kwargs
        )
        data_collator = RetainCollator
        trainer_cls = {
            "grad_diff": GradDiffTrainer,
            "npo": NPOTrainer,
            "simnpo": SimNPOForgetRetainTrainer,
            "rmu": RMUTrainer,
        }[method]

    run_type = f"{method}_{'forget_only' if forget_only else 'forget_retain'}"
    return train_dataset, data_collator, trainer_cls, trainer_args, run_type


def get_trainable_parameter_counts(model):
    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return trainable_params, total_params


def configure_training_devices(api_config: Dict[str, Any]) -> None:
    """Choose visibility before Accelerate or model loading initializes CUDA."""
    from gpu.gpu_utils import selected_gpu_ids, set_cuda_visible_devices

    method = api_config.get("unlearning", {}).get("method", DEFAULT_UNLEARNING_METHOD)
    gpu_ids = selected_gpu_ids(api_config["gpu"])
    single_gpu = method in {"dpo", "npo", "rmu"}
    if single_gpu and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("DPO, NPO and RMU require a single training process on one GPU.")
    set_cuda_visible_devices(gpu_ids[:1] if single_gpu else gpu_ids)
    # The first selected physical GPU is remapped to logical cuda:0.
    os.environ["UL_MODEL_DEVICE_MAP"] = "cuda:0" if single_gpu else "balanced"


def run_orchestrator(api_config: Dict[str, Any]) -> Dict[str, Any]:
    configure_training_devices(api_config)

    from accelerate import Accelerator

    from dataset.dataset_loader import read_file
    from model.model_loader import load_model_by_method

    accelerator = Accelerator()

    model_cfg = api_config["model"]
    data_cfg = api_config["dataset"]
    hp = api_config["hyperparams"]
    method = api_config.get("unlearning", {}).get(
        "method", DEFAULT_UNLEARNING_METHOD
    )

    model, tokenizer, merged = load_model_by_method(
        method=model_cfg["method"],
        base_model_path=model_cfg["model_name"],
        adaptor_path=model_cfg.get("adaptor_path"),
        gpu_id=0,  # because CUDA_VISIBLE_DEVICES remaps selected GPU to cuda:0
        hf_key=api_config["model"].get("hf_key"),
        **({
            "target_modules": (hp.get("lora_settings") or {}).get("target_modules"),
            "layers_to_transform": (
                hp.get("method", {}).get("layers_to_transform", UNLEARNING_METHOD_ARGS["rmu"]["layers_to_transform"])
                if method == "rmu" else None
            ),
        } if model_cfg["method"] == "lora" else {}),
    )

    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    trainable_params, total_params = get_trainable_parameter_counts(model)
    if trainable_params == 0:
        raise ValueError(
            "Loaded model has no trainable parameters. "
            "For method='adaptor', keep the PEFT adapter attached instead of "
            "calling merge_and_unload() before training."
        )

    forget_df = read_file(data_cfg["forget_set_path"])

    retain_path = data_cfg.get("retain_set_path")
    retain_df = read_file(retain_path) if retain_path else None

    context_length = hp["general"]["context_length"]

    training_args = build_training_args(api_config)

    train_dataset, data_collator, trainer_cls, trainer_args, run_type = (
        build_unlearning_run(
            method=method,
            forget_df=forget_df,
            retain_df=retain_df,
            tokenizer=tokenizer,
            context_length=context_length,
            gamma=hp["general"].get("gamma", 1.0),
            alpha=hp["general"].get("alpha", 1.0),
            method_hyperparams=hp.get("method"),
        )
    )

    trainer_kwargs = build_trainer_kwargs(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer = trainer_cls(**trainer_args, **trainer_kwargs)
    trainable_params, total_params = get_trainable_parameter_counts(model)

    result = trainer.train()

    accelerator.wait_for_everyone()

    output_dir = os.path.join(
        api_config.get("output_dir", "outputs/run"),
        run_type,
    )

    if merged and model_cfg["method"] == "lora":
        # The new adapter was trained on a base containing a previous adapter.
        # An adapter-only artifact would lose that base when evaluation reloads it.
        model.merge_and_unload().save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    import torch
    import gc
    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "status": "success",
        "unlearning_method": method,
        "run_type": run_type,
        "merged_adapter": merged,
        "output_dir": output_dir,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "metrics": getattr(result, "metrics", {}),
    }
