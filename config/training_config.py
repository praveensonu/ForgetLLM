from schemas import FinalTrainingConfigRequest
from unlearning.methods import UNLEARNING_METHOD_ARGS
from unlearning.rmu.config import resolve_rmu_params


def build_orchestrator_config(request: FinalTrainingConfigRequest) -> dict:
    method_params = request.hyperparams.method.model_dump(exclude_none=True)
    if request.unlearning_method.value == "rmu":
        method_params = resolve_rmu_params(method_params)
    return {
        "project_id": request.project_id,
        "project_name": request.project_name,
        "model": {
            "model_name": request.model_name,
            "adaptor_path": request.adaptor_path,
            "method": request.method.value,
            "hf_key": request.hf_key,
        },
        "dataset": {
            "forget_set_path": request.forget_set_path,
            "retain_set_path": request.retain_set_path,
        },
        "gpu": {
            "gpu_id": request.gpu_ids[0],
            "gpu_ids": request.gpu_ids,
        },
        "unlearning": {
            "method": request.unlearning_method.value,
        },
        "hyperparams": {
            "method": {
                **{key: value for key, value in UNLEARNING_METHOD_ARGS[request.unlearning_method.value].items()
                   if key in {"beta", "delta"}},
                **method_params,
            },
            "general": {
                "gamma": request.hyperparams.general.gamma,
                "alpha": request.hyperparams.general.alpha,
                "max_steps": request.hyperparams.general.max_steps,
                "epochs": request.hyperparams.general.epochs,
                "learning_rate": request.hyperparams.general.learning_rate,
                "context_length": request.hyperparams.general.context_length,
            },
            "optimization": {
                "batch_size": request.hyperparams.optimization.batch_size,
                "grad_accum": request.hyperparams.optimization.grad_accum,
                "weight_decay": request.hyperparams.optimization.weight_decay,
            },
            "schedule": {
                "save_steps": request.hyperparams.schedule.save_steps,
            },
            "memory": {
                "assistant_completions_only": request.hyperparams.memory.assistant_completions_only,
            },
            "lora_settings": (
                {
                    "target_modules": [
                        m.value
                        for m in request.hyperparams.lora_settings.target_modules
                    ]
                }
                if request.hyperparams.lora_settings is not None
                else None
            ),
        },
    }
