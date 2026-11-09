from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from dataset.dataset_loader import read_file, validate_qa_columns


ProgressCallback = Callable[[str, str], None]


def _report(
    callback: Optional[ProgressCallback], stage: str, message: str
) -> None:
    if callback is not None:
        callback(stage, message)


def _load_unlearned_model(
    model_path: str,
    base_model_path: str,
    gpu_id: int,
    hf_key: str | None,
):
    """Load either a full saved model or a saved PEFT adapter."""

    from model.model_loader import (
        load_base_model,
        load_full_model,
        load_tokenizer,
    )

    if not (Path(model_path) / "adapter_config.json").is_file():
        model, tokenizer, _ = load_full_model(
            base_model_path=model_path,
            gpu_id=gpu_id,
            hf_key=hf_key,
        )
        return model, tokenizer

    from peft import PeftModel

    base_model = load_base_model(
        base_model_path=base_model_path,
        gpu_id=gpu_id,
        hf_key=hf_key,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        is_trainable=False,
        device_map=os.environ.get("UL_MODEL_DEVICE_MAP", "cuda:0"),
    )
    try:
        tokenizer = load_tokenizer(model_path, hf_key)
    except (OSError, ValueError):
        tokenizer = load_tokenizer(base_model_path, hf_key)
    model.eval()
    return model, tokenizer


def _load_pre_unlearning_model(model_config: Dict[str, Any]):
    from model.model_loader import load_model_by_method

    model, tokenizer, _ = load_model_by_method(
        # LoRA evaluation starts from the original base (plus an optional merged
        # input adapter), without allocating a fresh untrained LoRA adapter.
        method="full" if model_config["method"] == "lora" else model_config["method"],
        base_model_path=model_config["model_name"],
        adaptor_path=model_config.get("adaptor_path"),
        gpu_id=0,
        hf_key=model_config.get("hf_key"),
    )
    model.eval()
    return model, tokenizer


def _prepare_evaluation_dataframe(
    df: pd.DataFrame,
    tokenizer,
    dataset_name: str,
    max_new_tokens: int,
) -> pd.DataFrame:
    validate_qa_columns(df, dataset_name)
    if df.empty:
        raise ValueError(f"{dataset_name} must contain at least one row.")

    prepared = df.copy()
    prepared["question"] = prepared["question"].astype(str)
    prepared["answer"] = prepared["answer"].astype(str)
    encoded_answers = tokenizer(
        prepared["answer"].tolist(), add_special_tokens=False
    )["input_ids"]
    inferred_lengths = pd.Series(
        [max(1, len(token_ids)) for token_ids in encoded_answers],
        index=prepared.index,
    )

    if "num_tokens" in prepared.columns:
        supplied_lengths = pd.to_numeric(
            prepared["num_tokens"], errors="coerce"
        )
        generation_lengths = supplied_lengths.where(
            supplied_lengths > 0, inferred_lengths
        )
    else:
        generation_lengths = inferred_lengths

    prepared["num_tokens"] = (
        generation_lengths.clip(lower=1, upper=max_new_tokens).astype(int)
    )
    return prepared


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def _release_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    gc.collect()


def _json_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Evaluation produced a non-finite metric.")
    return result


def _collect_model_outputs_phase(
    *,
    label: str,
    loader: Callable[[], Any],
    forget_source: pd.DataFrame,
    retain_source: pd.DataFrame,
    max_new_tokens: int,
    forget_output_path: str,
    retain_output_path: str,
    torch_module,
    progress_callback: Optional[ProgressCallback],
    phase: str = "both",
    batch_size: int = 4,
    hf_key: str | None = None,
    include_benchmarks: bool = False,
    gpu_ids: list[int] | None = None,
) -> dict | None:
    from eval.eval_utils import compute_model_outputs

    phase_description = {
        "metrics": "perplexity and conditional probability",
        "generation": "generations",
        "both": "perplexity, conditional probability, and generations",
    }[phase]
    stage_prefix = "pre" if label == "Pre-unlearning" else "post"
    _report(
        progress_callback,
        f"loading_{stage_prefix}_model",
        f"Loading {label.lower()} model.",
    )
    model = None
    tokenizer = None
    try:
        model, tokenizer = loader()
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        device = _model_input_device(model)
        _report(
            progress_callback,
            f"{stage_prefix}_model_loaded",
            f"{label} model loaded.",
        )

        forget_df = _prepare_evaluation_dataframe(
            forget_source, tokenizer, "forget set", max_new_tokens
        )
        retain_df = _prepare_evaluation_dataframe(
            retain_source, tokenizer, "retain set", max_new_tokens
        )

        _report(
            progress_callback,
            f"calculating_{stage_prefix}_forget_model_metrics",
            f"Calculating {label.lower()} forget-set {phase_description}.",
        )
        forget_results = compute_model_outputs(
            forget_df, model, tokenizer, device, f"{label} forget set", phase=phase,
            batch_size=batch_size, progress_callback=progress_callback,
        )
        if phase == "generation":
            scored = pd.read_parquet(forget_output_path)
            scored["gen_answer"] = forget_results["gen_answer"]
            forget_results = scored
        forget_results.to_parquet(forget_output_path, index=False)

        _report(
            progress_callback,
            f"calculating_{stage_prefix}_retain_model_metrics",
            f"Calculating {label.lower()} retain-set {phase_description}.",
        )
        retain_results = compute_model_outputs(
            retain_df, model, tokenizer, device, f"{label} retain set", phase=phase,
            batch_size=batch_size, progress_callback=progress_callback,
        )
        if phase == "generation":
            scored = pd.read_parquet(retain_output_path)
            scored["gen_answer"] = retain_results["gen_answer"]
            retain_results = scored
        retain_results.to_parquet(retain_output_path, index=False)
        _report(
            progress_callback,
            f"stored_{stage_prefix}_model_outputs",
            f"Stored {label.lower()} {phase_description}.",
        )
        if include_benchmarks and phase != "generation":
            from eval.benchmarks import evaluate_benchmarks

            def report_benchmark(stage, message):
                _report(progress_callback, f"{stage_prefix}_{stage}", f"{label}: {message}")

            return evaluate_benchmarks(
                model, tokenizer, batch_size=batch_size, hf_key=hf_key,
                progress_callback=report_benchmark, gpu_ids=gpu_ids,
            )
        return None
    finally:
        del tokenizer
        del model
        _release_cuda(torch_module)
        _report(
            progress_callback,
            f"removed_{stage_prefix}_model",
            f"Removed {label.lower()} model from memory.",
        )


def _collect_model_outputs(*, gpu_ids: list[int], **kwargs) -> dict | None:
    # Do not change CUDA visibility after initialization. Explicit model placement
    # confines scoring to logical cuda:0; generation balances across the selection.
    previous = os.environ.get("UL_MODEL_DEVICE_MAP")
    try:
        os.environ["UL_MODEL_DEVICE_MAP"] = "cuda:0"
        benchmarks = _collect_model_outputs_phase(
            **kwargs, gpu_ids=gpu_ids, phase="metrics" if len(gpu_ids) > 1 else "both"
        )
        if len(gpu_ids) > 1:
            os.environ["UL_MODEL_DEVICE_MAP"] = "balanced"
            _collect_model_outputs_phase(**kwargs, phase="generation")
        return benchmarks
    finally:
        if previous is None:
            os.environ.pop("UL_MODEL_DEVICE_MAP", None)
        else:
            os.environ["UL_MODEL_DEVICE_MAP"] = previous


def _summarize_model(
    forget_df: pd.DataFrame, retain_df: pd.DataFrame
) -> Dict[str, Any]:
    from eval.eval_utils import compute_fq_scores, compute_mu_scores

    (
        forget_components,
        forget_quality,
        forget_perplexity,
        forget_probability,
        forget_rouge_l,
    ) = compute_fq_scores(forget_df)
    (
        utility_components,
        model_utility,
        retain_perplexity,
        retain_probability,
        retain_rouge_l,
        retain_cosine_similarity,
    ) = compute_mu_scores(retain_df)

    return {
        "forget_quality": {
            "evaluated_rows": len(forget_df),
            "score": _json_float(forget_quality),
            "average_perplexity": _json_float(forget_perplexity),
            "mean_conditional_probability": _json_float(forget_probability),
            "mean_rouge_l": _json_float(forget_rouge_l),
            "component_scores": [
                _json_float(score) for score in forget_components
            ],
        },
        "model_utility": {
            "evaluated_rows": len(retain_df),
            "score": _json_float(model_utility),
            "average_perplexity": _json_float(retain_perplexity),
            "mean_conditional_probability": _json_float(retain_probability),
            "mean_rouge_l": _json_float(retain_rouge_l),
            "mean_cosine_similarity": _json_float(
                retain_cosine_similarity
            ),
            "component_scores": [
                _json_float(score) for score in utility_components
            ],
        },
    }


def run_eval_orchestrator(
    api_config: Dict[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Compare pre- and post-unlearning evaluation results in memory phases."""

    orchestrator_config = api_config["orchestrator_config"]
    training_result = api_config["training_result"]
    model_config = orchestrator_config["model"]
    dataset_config = orchestrator_config["dataset"]
    from gpu.gpu_utils import selected_gpu_ids, set_cuda_visible_devices
    gpu_ids = selected_gpu_ids(orchestrator_config["gpu"])
    set_cuda_visible_devices(gpu_ids)

    import torch

    model_path = training_result["output_dir"]
    forget_set_path = dataset_config["forget_set_path"]
    retain_set_path = dataset_config.get("retain_set_path")
    if not retain_set_path:
        raise ValueError("A retain_set_path is required to compute model utility.")
    if not Path(model_path).is_dir():
        raise ValueError(f"Unlearned model directory does not exist: {model_path}")

    evaluation_output_dir = api_config.get("evaluation_output_dir") or os.path.join(
        model_path, "evaluation"
    )
    os.makedirs(evaluation_output_dir, exist_ok=True)
    output_paths = {
        "pre_forget_scores_path": os.path.join(
            evaluation_output_dir, "pre_unlearning_forget_scores.parquet"
        ),
        "pre_retain_scores_path": os.path.join(
            evaluation_output_dir, "pre_unlearning_retain_scores.parquet"
        ),
        "post_forget_scores_path": os.path.join(
            evaluation_output_dir, "post_unlearning_forget_scores.parquet"
        ),
        "post_retain_scores_path": os.path.join(
            evaluation_output_dir, "post_unlearning_retain_scores.parquet"
        ),
    }

    _report(progress_callback, "loading_datasets", "Loading forget and retain sets.")
    forget_source = read_file(forget_set_path)
    retain_source = read_file(retain_set_path)
    max_new_tokens = api_config.get("max_new_tokens", 256)

    pre_benchmarks = _collect_model_outputs(
        gpu_ids=gpu_ids,
        batch_size=api_config.get("batch_size", 4),
        include_benchmarks=api_config.get("include_benchmarks", False),
        hf_key=model_config.get("hf_key"),
        label="Pre-unlearning",
        loader=lambda: _load_pre_unlearning_model(model_config),
        forget_source=forget_source,
        retain_source=retain_source,
        max_new_tokens=max_new_tokens,
        forget_output_path=output_paths["pre_forget_scores_path"],
        retain_output_path=output_paths["pre_retain_scores_path"],
        torch_module=torch,
        progress_callback=progress_callback,
    )
    post_benchmarks = _collect_model_outputs(
        gpu_ids=gpu_ids,
        batch_size=api_config.get("batch_size", 4),
        include_benchmarks=api_config.get("include_benchmarks", False),
        hf_key=model_config.get("hf_key"),
        label="Unlearnt",
        loader=lambda: _load_unlearned_model(
            model_path=model_path,
            base_model_path=model_config["model_name"],
            gpu_id=0,
            hf_key=model_config.get("hf_key"),
        ),
        forget_source=forget_source,
        retain_source=retain_source,
        max_new_tokens=max_new_tokens,
        forget_output_path=output_paths["post_forget_scores_path"],
        retain_output_path=output_paths["post_retain_scores_path"],
        torch_module=torch,
        progress_callback=progress_callback,
    )

    # Reload the persisted model outputs only after both language models have
    # been removed. The embedding model is never resident with either LLM.
    pre_forget = pd.read_parquet(output_paths["pre_forget_scores_path"])
    pre_retain = pd.read_parquet(output_paths["pre_retain_scores_path"])
    post_forget = pd.read_parquet(output_paths["post_forget_scores_path"])
    post_retain = pd.read_parquet(output_paths["post_retain_scores_path"])

    from eval.eval_utils import (
        compute_cosine_similarity_scores,
        compute_rouge_l_scores,
    )
    from sentence_transformers import SentenceTransformer

    embedding_model_name = api_config["embedding_model_name"]
    _report(
        progress_callback,
        "loading_embedding_model",
        f"Loading sentence-transformers model: {embedding_model_name}.",
    )
    embedding_kwargs: Dict[str, Any] = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }
    if model_config.get("hf_key"):
        embedding_kwargs["token"] = model_config["hf_key"]
    embedding_model = SentenceTransformer(
        embedding_model_name, **embedding_kwargs
    )
    try:
        _report(
            progress_callback,
            "calculating_cosine_similarity",
            "Calculating retain-set cosine similarity for pre- and post-unlearning outputs.",
        )
        batch_size = api_config.get("embedding_batch_size", 32)
        pre_retain = compute_cosine_similarity_scores(
            pre_retain, embedding_model, batch_size
        )
        post_retain = compute_cosine_similarity_scores(
            post_retain, embedding_model, batch_size
        )
    finally:
        del embedding_model
        _release_cuda(torch)
        _report(
            progress_callback,
            "removed_embedding_model",
            "Removed sentence-transformers model from memory.",
        )

    _report(
        progress_callback,
        "calculating_rouge_l",
        "Calculating ROUGE-L for both datasets and both model versions.",
    )
    pre_forget = compute_rouge_l_scores(pre_forget)
    pre_retain = compute_rouge_l_scores(pre_retain)
    post_forget = compute_rouge_l_scores(post_forget)
    post_retain = compute_rouge_l_scores(post_retain)

    pre_forget.to_parquet(output_paths["pre_forget_scores_path"], index=False)
    pre_retain.to_parquet(output_paths["pre_retain_scores_path"], index=False)
    post_forget.to_parquet(output_paths["post_forget_scores_path"], index=False)
    post_retain.to_parquet(output_paths["post_retain_scores_path"], index=False)

    _report(
        progress_callback,
        "calculating_scores",
        "Calculating forget quality and model utility scores.",
    )
    result = {
        "status": "success",
        "model_path": model_path,
        "embedding_model_name": embedding_model_name,
        "forget_set_path": forget_set_path,
        "retain_set_path": retain_set_path,
        "pre_unlearning": {**_summarize_model(pre_forget, pre_retain), "benchmarks": pre_benchmarks},
        "post_unlearning": {**_summarize_model(post_forget, post_retain), "benchmarks": post_benchmarks},
        "output_files": output_paths,
        "message": "Pre- and post-unlearning evaluation completed successfully.",
    }
    from eval.eval_utils import save_evaluation_jsonl
    from datetime import datetime, timezone
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    experiment_name = api_config.get("experiment_name") or orchestrator_config.get("experiment_name") or Path(model_path).parent.name
    _report(progress_callback, "saving_results", "Saving evaluation JSONL results.")
    import project_store
    project_id = orchestrator_config.get("project_id")
    save_evaluation_jsonl(result, experiment_name, results_root=evaluation_output_dir if project_id else None)
    project_store.record(project_id, "evaluation_result", result)
    _report(progress_callback, "completed", result["message"])
    return result
