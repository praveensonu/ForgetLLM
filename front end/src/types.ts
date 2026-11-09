export type Method = "full" | "lora" | "adaptor";
export type UnlearningMethod = "grad_ascent" | "grad_diff" | "npo" | "dpo" | "simnpo" | "rmu";

export type UnlearningMethodInfo = {
  value: UnlearningMethod;
  label: string;
  requires_retain: boolean;
};
export type StepMode = "max_steps" | "epochs";
export type LoraTarget = string;
export type ProjectStage = "gpu" | "data" | "model" | "hyperparameters" | "running" | "evaluation";
export type DataSourceMode = "upload" | "extract";
export type DataSelectionMethod = "raslik" | "grace";

export type BackendPreviewRow = {
  question?: string;
  answer?: string;
  [key: string]: unknown;
};

export type ProjectPreviewRow = {
  key: string;
  id: string;
  question: string;
  answer: string;
  raw: Record<string, unknown>;
};

export type DatasetUploadResponse = {
  status: string;
  forget_set_path: string;
  retain_set_path: string | null;
  forget_rows: number;
  retain_rows: number | null;
  has_retain_set: boolean;
  prompt_template: string;
  forget_preview: BackendPreviewRow[];
  retain_preview: BackendPreviewRow[] | null;
  message: string;
};

export type GradientCacheResponse = {
  status: "success";
  experiment_name: string;
  training_grads_path: string;
  poison_grads_path: string;
  training_data_path: string;
  poison_data_path: string;
  training_config_path: string;
  poison_config_path: string;
  training_rows: number;
  poison_rows: number;
  message: string;
};

export type DatasetExtractionResponse = DatasetUploadResponse & GradientCacheResponse & {
  selection_method: DataSelectionMethod;
  selection_metadata_path: string;
  retain_set_path: string;
  retain_rows: number;
  retain_preview: BackendPreviewRow[];
  has_retain_set: true;
  gradients_retained: boolean;
};

export type ProgressEvent = {
  stage: string;
  message: string;
  timestamp: string;
};

export type ExtractionStartResponse = {
  job_id: string;
  status: "queued" | "running";
  message: string;
};

export type ExtractionJobStatus = {
  job_id: string;
  status: "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
  current_stage: string;
  message: string;
  progress: ProgressEvent[];
  result: DatasetExtractionResponse | null;
  error: string | null;
};

export type JobCancelResponse = {
  status: "cancelling" | "cancelled" | "idle";
  message: string;
};

export type Hyperparams = {
  method: {
    beta?: number;
    delta?: number;
    layers_to_transform?: number[];
    module_regex?: string;
    trainable_params_regex?: string;
    steering_coeff?: number;
  };
  general: {
    max_steps: number | null;
    epochs: number | null;
    learning_rate: string;
    gamma: number;
    alpha: number;
    context_length: number;
  };
  optimization: {
    batch_size: number;
    grad_accum: number;
    weight_decay: number;
  };
  schedule: {
    save_steps: number;
  };
  memory: {
    assistant_completions_only: true;
  };
  lora_settings?: {
    target_modules: LoraTarget[];
  };
};

export type FinalTrainingConfigRequest = {
  project_id?: string;
  project_name?: string;
  model_name: string;
  adaptor_path: string | null;
  hf_key: string | null;
  method: Method;
  unlearning_method: UnlearningMethod;
  gpu_ids: number[];
  forget_set_path: string;
  retain_set_path: string | null;
  hyperparams: Hyperparams;
};

export type ConfigBuildResponse = {
  status: string;
  orchestrator_config: Record<string, unknown>;
  message: string;
};

type TrainResult = {
  status?: string;
  run_type?: string;
  merged_adapter?: boolean;
  output_dir?: string;
  trainable_parameters?: number;
  total_parameters?: number;
  metrics?: Record<string, unknown>;
  [key: string]: unknown;
};

export type TrainRunResponse =
  | {
      status: "success";
      orchestrator_config: Record<string, unknown>;
      result: TrainResult;
      message?: string | null;
    }
  | {
      status: "stopped";
      orchestrator_config: Record<string, unknown>;
      result: null;
      message: string;
    };

export type TrainStopResponse = {
  status: "stopped" | "idle";
  message: string;
};

export type EvaluationRequest = {
  project_id?: string;
  project_name?: string;
  orchestrator_config: Record<string, unknown>;
  training_result: Record<string, unknown>;
  embedding_model_name: string;
  include_benchmarks?: boolean;
  embedding_batch_size?: number;
  batch_size?: number;
  experiment_name?: string;
  max_new_tokens?: number;
};

export type ForgetQualityScores = {
  evaluated_rows: number;
  score: number;
  average_perplexity: number;
  mean_conditional_probability: number;
  mean_rouge_l: number;
  component_scores: number[];
};

export type ModelUtilityScores = {
  evaluated_rows: number;
  score: number;
  average_perplexity: number;
  mean_conditional_probability: number;
  mean_rouge_l: number;
  mean_cosine_similarity: number;
  component_scores: number[];
};

export type ModelEvaluationScores = {
  benchmarks?: { mmlu: number; gpqa: number } | null;
  forget_quality: ForgetQualityScores;
  model_utility: ModelUtilityScores;
};

export type EvaluationResponse = {
  status: "success";
  model_path: string;
  embedding_model_name: string;
  forget_set_path: string;
  retain_set_path: string;
  pre_unlearning: ModelEvaluationScores;
  post_unlearning: ModelEvaluationScores;
  output_files: Record<string, string>;
  message: string;
};

export type EvaluationProgressEvent = {
  stage: string;
  message: string;
  timestamp: string;
};

export type EvaluationStartResponse = {
  job_id: string;
  status: "queued" | "running";
  message: string;
};

export type EvaluationJobStatus = {
  job_id: string;
  status: "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
  current_stage: string;
  message: string;
  progress: EvaluationProgressEvent[];
  result: EvaluationResponse | null;
  error: string | null;
};

export type ProjectDataConfig = {
  sourceMode: DataSourceMode;

  // Direct upload mode.
  forgetFile: File | null;
  retainFile: File | null;

  // Extract mode: derive forget/retain from full data + poison gradients.
  fullFile: File | null;
  poisonFile: File | null;
  extractionModelName: string;
  extractionHfKey: string;
  extractionMaxLength: number;
  gradientBatchSize: number;
  extractionAdaptorPath: string;
  selectionMethod: DataSelectionMethod;
  forgetSize: number;
  retainSize: number;
  graceTopN: number;
  graceNumClusters: number;
  keepGradients: boolean;
  extractionJob: ExtractionJobStatus | null;
  extractionResponse: DatasetExtractionResponse | null;

  // Files actually sent to the backend after extraction / row filtering.
  preparedForgetFile: File | null;
  preparedRetainFile: File | null;

  // The UI collects an instruction; the dataset question is appended to it.
  promptTemplate: string;

  previewRows: ProjectPreviewRow[];
  selectedPreviewKeys: string[];
  previewReady: boolean;
  previewFilterable: boolean;

  uploadResponse: DatasetUploadResponse | null;
  backendUploadError: string | null;
};

export type ProjectModelConfig = {
  modelName: string;
  method: Method;
  adaptorPath: string;
  hfKey: string;
  gpuIds: number[];
  selectedTargets: LoraTarget[];
};

export type ProjectHyperparametersConfig = {
  unlearningMethod: UnlearningMethod;
  stepMode: StepMode;
  maxSteps: number;
  epochs: number;
  learningRate: string;
  contextLength: number;
  batchSize: number;
  gradAccum: number;
  dpoBeta: string;
  npoBeta: string;
  simnpoBeta: string;
  simnpoDelta: string;
  rmuLayersToTransform: string;
  rmuModuleRegex: string;
  rmuSteeringCoeff: string;
  forgettingStrength: string;
  retentionStrength: string;
  weightDecay: number;
  saveSteps: number;
};

export type ProjectRunState = {
  config: ConfigBuildResponse | null;
  training: TrainRunResponse | null;
  message: string | null;
  embeddingModelName: string;
  evaluationMaxNewTokens: number;
  evaluationBatchSize: number;
  includeBenchmarks: boolean;
  evaluationJob: EvaluationJobStatus | null;
  evaluation: EvaluationResponse | null;
  evaluationMessage: string | null;
};

export type CompletedStages = {
  gpu: boolean;
  data: boolean;
  model: boolean;
  hyperparameters: boolean;
  running: boolean;
  evaluation: boolean;
};

export type Project = {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  lastStage: ProjectStage;
  completedStages: CompletedStages;
  setupComplete: boolean;
  pendingResetFrom: ProjectStage | null;
  data: ProjectDataConfig;
  model: ProjectModelConfig;
  hyperparameters: ProjectHyperparametersConfig;
  run: ProjectRunState;
};
