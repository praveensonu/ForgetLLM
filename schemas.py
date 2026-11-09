from enum import Enum
import re
from typing import Annotated, Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unlearning import methods as registry


class LoadMethod(str, Enum):
    full = "full"
    lora = "lora"
    adaptor = "adaptor"


class UnlearningMethod(str, Enum):
    grad_ascent = "grad_ascent"
    grad_diff = "grad_diff"
    npo = "npo"
    dpo = "dpo"
    simnpo = "simnpo"
    rmu = "rmu"


# methods whose loss has a retain term and therefore need a retain set
RETAIN_REQUIRED_METHODS = frozenset(
    UnlearningMethod(name) for name in registry.RETAIN_REQUIRED_METHODS
)


class UnlearningMethodInfo(BaseModel):
    value: UnlearningMethod
    label: str
    requires_retain: bool


class UnlearningMethodsResponse(BaseModel):
    methods: List[UnlearningMethodInfo]
    default: UnlearningMethod


class LoadModelRequest(BaseModel):
    base_model_path: str
    adaptor_path: Optional[str] = None
    hf_key: Optional[str] = None
    method: LoadMethod


class LoadModelResponse(BaseModel):
    status: str
    method: str
    base_model_path: str
    adaptor_path: Optional[str] = None
    merged: bool
    message: str


class DatasetUploadResponse(BaseModel):
    status: str
    forget_set_path: str
    retain_set_path: Optional[str]
    forget_rows: int
    retain_rows: Optional[int]
    has_retain_set: bool
    prompt_template: str
    forget_preview: List[Dict[str, Any]]
    retain_preview: Optional[List[Dict[str, Any]]]
    message: str


class GradientCacheResponse(BaseModel):
    status: Literal["success"]
    experiment_name: str
    training_grads_path: str
    poison_grads_path: str
    training_data_path: str
    poison_data_path: str
    training_config_path: str
    poison_config_path: str
    training_rows: int
    poison_rows: int
    message: str


class DataSelectionMethod(str, Enum):
    raslik = "raslik"
    grace = "grace"


class DatasetExtractionResponse(GradientCacheResponse):
    selection_method: DataSelectionMethod
    forget_set_path: str
    retain_set_path: str
    selection_metadata_path: str
    forget_rows: int
    retain_rows: int
    has_retain_set: Literal[True]
    prompt_template: str
    forget_preview: List[Dict[str, Any]]
    retain_preview: List[Dict[str, Any]]
    gradients_retained: bool


class ExtractionStartResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"]
    message: str


class ExtractionProgressEvent(BaseModel):
    stage: str
    message: str
    timestamp: str


class ExtractionStatusResponse(BaseModel):
    job_id: str
    status: Literal[
        "queued", "running", "cancelling", "cancelled", "completed", "failed"
    ]
    current_stage: str
    message: str
    progress: List[ExtractionProgressEvent]
    result: Optional[DatasetExtractionResponse] = None
    error: Optional[str] = None


class JobCancelResponse(BaseModel):
    status: Literal["cancelling", "cancelled", "idle"]
    message: str


class LoraTargetModule(str, Enum):
    q_proj = "q_proj"
    v_proj = "v_proj"
    k_proj = "k_proj"
    o_proj = "o_proj"


class GeneralHyperParams(BaseModel):
    gamma: float = Field(1.0, ge=0, allow_inf_nan=False)
    alpha: float = Field(1.0, ge=0, allow_inf_nan=False)
    max_steps: Optional[int] = Field(None, ge=1)
    epochs: Optional[int] = Field(None, ge=1)
    learning_rate: Union[float, str]
    context_length: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_steps_or_epochs(self):
        if self.max_steps is None and self.epochs is None:
            raise ValueError("Either max_steps or epochs must be provided.")
        if self.max_steps is not None and self.epochs is not None:
            raise ValueError("Only one of max_steps or epochs can be provided.")
        return self

    @field_validator("context_length")
    @classmethod
    def validate_power_of_two(cls, value):
        if value & (value - 1) != 0:
            raise ValueError("context_length must be a power of 2.")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value):
        value = float(value)
        if value <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        return value


class LoraSettings(BaseModel):
    target_modules: List[LoraTargetModule]


class OptimizationHyperParams(BaseModel):
    batch_size: int = Field(..., ge=1)
    grad_accum: int = Field(..., ge=1)
    weight_decay: float = Field(..., ge=0)


class ScheduleHyperParams(BaseModel):
    save_steps: int = Field(..., ge=1)


class MemoryHyperParams(BaseModel):
    assistant_completions_only: bool = True

    @field_validator("assistant_completions_only")
    @classmethod
    def must_be_true(cls, value):
        if value is not True:
            raise ValueError("assistant_completions_only must always be true.")
        return value


class MethodHyperParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    beta: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    delta: Optional[float] = Field(None, allow_inf_nan=False)
    layers_to_transform: Optional[List[Annotated[int, Field(strict=True, ge=0)]]] = Field(None, min_length=1)
    module_regex: Optional[str] = Field(None, min_length=1)
    trainable_params_regex: Optional[str] = Field(None, min_length=1)
    steering_coeff: Optional[float] = Field(None, ge=0, allow_inf_nan=False)

    @field_validator("layers_to_transform")
    @classmethod
    def validate_layers(cls, value):
        if value is not None and len(set(value)) != len(value):
            raise ValueError("layers_to_transform must not contain duplicates.")
        return sorted(value) if value is not None else None

    @field_validator("module_regex", "trainable_params_regex")
    @classmethod
    def validate_regex(cls, value):
        if value is not None:
            if not value.strip():
                raise ValueError("Regular expressions must not be empty.")
            try:
                re.compile(value)
            except re.error as error:
                raise ValueError(f"Invalid regular expression: {error}") from error
        return value


class HyperParamsConfig(BaseModel):
    method: MethodHyperParams = Field(default_factory=MethodHyperParams)
    general: GeneralHyperParams
    optimization: OptimizationHyperParams
    schedule: ScheduleHyperParams
    memory: MemoryHyperParams
    lora_settings: Optional[LoraSettings] = None


class FinalTrainingConfigRequest(BaseModel):
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    model_name: str
    adaptor_path: Optional[str] = None
    hf_key: Optional[str] = None
    method: LoadMethod
    unlearning_method: UnlearningMethod = UnlearningMethod.simnpo
    gpu_id: Optional[int] = Field(None, ge=0)
    gpu_ids: Optional[List[int]] = None

    forget_set_path: str
    retain_set_path: Optional[str] = None

    hyperparams: HyperParamsConfig

    @model_validator(mode="after")
    def validate_method_specific_config(self):
        from gpu.gpu_utils import selected_gpu_ids
        self.gpu_ids = selected_gpu_ids(self.model_dump(exclude_none=True))
        self.gpu_id = self.gpu_ids[0]

        supplied = self.hyperparams.method.model_dump(exclude_none=True)
        allowed = registry.UNLEARNING_METHOD_ARGS[self.unlearning_method.value]
        unsupported = set(supplied) - set(allowed)
        if unsupported:
            raise ValueError(
                f"Unsupported hyperparameters for {self.unlearning_method.value}: {sorted(unsupported)}"
            )

        if self.method == LoadMethod.lora:
            if self.hyperparams.lora_settings is None:
                raise ValueError("lora_settings is required when method='lora'.")

        if self.method != LoadMethod.lora:
            if self.hyperparams.lora_settings is not None:
                raise ValueError("lora_settings is only allowed when method='lora'.")

        if self.method == LoadMethod.adaptor:
            if self.adaptor_path is None:
                raise ValueError("adaptor_path is required when method='adaptor'.")

        if (
            self.unlearning_method in RETAIN_REQUIRED_METHODS
            and self.retain_set_path is None
        ):
            raise ValueError(
                "retain_set_path is required when "
                f"unlearning_method='{self.unlearning_method.value}'."
            )

        return self


class FinalTrainingConfigResponse(BaseModel):
    status: str
    orchestrator_config: Dict[str, Any]
    message: str


class TrainRunSuccessResponse(BaseModel):
    status: Literal["success"]
    orchestrator_config: Dict[str, Any]
    result: Dict[str, Any]


class TrainRunStoppedResponse(BaseModel):
    status: Literal["stopped"]
    orchestrator_config: Dict[str, Any]
    result: None = None
    message: str


TrainRunResponse = Union[TrainRunSuccessResponse, TrainRunStoppedResponse]


class TrainStopResponse(BaseModel):
    status: Literal["stopped", "idle"]
    message: str


class EvaluationTrainingResult(BaseModel):
    """The part of a completed training response needed for evaluation."""

    model_config = ConfigDict(extra="allow")

    output_dir: str = Field(..., min_length=1)


class EvaluationRequest(BaseModel):
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    orchestrator_config: Dict[str, Any]
    training_result: EvaluationTrainingResult
    embedding_model_name: str = Field(..., min_length=1)
    include_benchmarks: bool = False
    embedding_batch_size: int = Field(default=32, ge=1)
    batch_size: int = Field(default=4, ge=1)
    experiment_name: Optional[str] = Field(default=None, min_length=1)
    max_new_tokens: int = Field(default=256, ge=1)
    evaluation_output_dir: Optional[str] = None

    @model_validator(mode="after")
    def validate_evaluation_inputs(self):
        model_config = self.orchestrator_config.get("model")
        dataset_config = self.orchestrator_config.get("dataset")
        gpu_config = self.orchestrator_config.get("gpu")

        if not isinstance(model_config, dict) or not model_config.get("model_name"):
            raise ValueError("orchestrator_config.model.model_name is required.")
        if not isinstance(dataset_config, dict):
            raise ValueError("orchestrator_config.dataset is required.")
        if not dataset_config.get("forget_set_path"):
            raise ValueError(
                "orchestrator_config.dataset.forget_set_path is required."
            )
        if not dataset_config.get("retain_set_path"):
            raise ValueError(
                "orchestrator_config.dataset.retain_set_path is required to "
                "compute model utility."
            )
        from gpu.gpu_utils import selected_gpu_ids
        if not isinstance(gpu_config, dict):
            raise ValueError("orchestrator_config.gpu is required.")
        ids = selected_gpu_ids(gpu_config)
        gpu_config.update(gpu_ids=ids, gpu_id=ids[0])

        return self


class ForgetQualityScores(BaseModel):
    evaluated_rows: int
    score: float
    average_perplexity: float
    mean_conditional_probability: float
    mean_rouge_l: float
    component_scores: List[float]


class ModelUtilityScores(BaseModel):
    evaluated_rows: int
    score: float
    average_perplexity: float
    mean_conditional_probability: float
    mean_rouge_l: float
    mean_cosine_similarity: float
    component_scores: List[float]


class BenchmarkScores(BaseModel):
    mmlu: float = Field(..., ge=0, le=1)
    gpqa: float = Field(..., ge=0, le=1)


class ModelEvaluationScores(BaseModel):
    benchmarks: Optional[BenchmarkScores] = None
    forget_quality: ForgetQualityScores
    model_utility: ModelUtilityScores


class EvaluationOutputFiles(BaseModel):
    results_jsonl_path: str
    pre_forget_scores_path: str
    pre_retain_scores_path: str
    post_forget_scores_path: str
    post_retain_scores_path: str


class EvaluationResponse(BaseModel):
    completed_at: str
    experiment_name: str
    status: Literal["success"]
    model_path: str
    embedding_model_name: str
    forget_set_path: str
    retain_set_path: str
    pre_unlearning: ModelEvaluationScores
    post_unlearning: ModelEvaluationScores
    output_files: EvaluationOutputFiles
    message: str


class EvaluationStartResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"]
    message: str


class EvaluationProgressEvent(BaseModel):
    stage: str
    message: str
    timestamp: str


class EvaluationStatusResponse(BaseModel):
    job_id: str
    status: Literal[
        "queued", "running", "cancelling", "cancelled", "completed", "failed"
    ]
    current_stage: str
    message: str
    progress: List[EvaluationProgressEvent]
    result: Optional[EvaluationResponse] = None
    error: Optional[str] = None


class GpuInfo(BaseModel):
    id: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    is_available: bool
    status: Literal["available", "busy"]


class GpuListResponse(BaseModel):
    gpus: List[GpuInfo]
    available_gpu_ids: List[int]
