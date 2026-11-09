import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model


loaded_models = {}


def load_tokenizer(base_model_path: str, hf_key: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        token=hf_key,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    base_model_path: str,
    gpu_id: int,
    hf_key: str | None = None,
):
    # gpu_id is a logical index after CUDA_VISIBLE_DEVICES remapping.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        token=hf_key,
        device_map=os.environ.get("UL_MODEL_DEVICE_MAP", "balanced"),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    return model


def load_full_model(
    base_model_path: str,
    gpu_id: int,
    hf_key: str | None = None,
    adaptor_path: str | None = None,
):
    model = load_base_model(base_model_path, gpu_id, hf_key)
    tokenizer = load_tokenizer(base_model_path, hf_key)

    merged = False

    if adaptor_path:
        peft_model = PeftModel.from_pretrained(
            model,
            adaptor_path,
            is_trainable=True,
            device_map=os.environ.get("UL_MODEL_DEVICE_MAP", "balanced"),
        )

        model = peft_model.merge_and_unload()
        merged = True

    model.eval()

    return model, tokenizer, merged


def load_lora_model(
    base_model_path: str,
    gpu_id: int,
    hf_key: str | None = None,
    adaptor_path: str | None = None,
    target_modules: list[str] | None = None,
    layers_to_transform: list[int] | None = None,
):
    model = load_base_model(base_model_path, gpu_id, hf_key)
    tokenizer = load_tokenizer(base_model_path, hf_key)

    merged = False

    if adaptor_path:
        peft_model = PeftModel.from_pretrained(
            model,
            adaptor_path,
            is_trainable=True,
            device_map=os.environ.get("UL_MODEL_DEVICE_MAP", "balanced"),
        )

        model = peft_model.merge_and_unload()
        merged = True

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        layers_to_transform=layers_to_transform,
        bias='none', task_type='CAUSAL_LM',
    ))
    model.eval()

    return model, tokenizer, merged


def load_adaptor_model(
    base_model_path: str,
    adaptor_path: str,
    gpu_id: int,
    hf_key: str | None = None,
):
    if adaptor_path is None:
        raise ValueError("adaptor_path is mandatory when method='adaptor'.")

    model = load_base_model(base_model_path, gpu_id, hf_key)
    tokenizer = load_tokenizer(base_model_path, hf_key)

    model = PeftModel.from_pretrained(
        model,
        adaptor_path,
        is_trainable=True,
        device_map=os.environ.get("UL_MODEL_DEVICE_MAP", "balanced"),
    )

    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, tokenizer, False


def load_model_by_method(
    method: str,
    base_model_path: str,
    gpu_id: int,
    hf_key: str | None = None,
    adaptor_path: str | None = None,
    target_modules: list[str] | None = None,
    layers_to_transform: list[int] | None = None,
):
    if method == "full":
        return load_full_model(
            base_model_path=base_model_path,
            hf_key=hf_key,
            adaptor_path=adaptor_path,
            gpu_id=gpu_id,
        )

    if method == "lora":
        return load_lora_model(
            base_model_path=base_model_path,
            hf_key=hf_key,
            adaptor_path=adaptor_path,
            gpu_id=gpu_id,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
        )

    if method == "adaptor":
        return load_adaptor_model(
            base_model_path=base_model_path,
            adaptor_path=adaptor_path,
            hf_key=hf_key,
            gpu_id=gpu_id,
        )

    raise ValueError(f"Unknown method: {method}")


def store_loaded_model(
    model_key: str,
    model,
    tokenizer,  
):
    loaded_models[model_key] = {
        "model": model,
        "tokenizer": tokenizer,
    }


def get_loaded_model(model_key: str):
    return loaded_models.get(model_key)
