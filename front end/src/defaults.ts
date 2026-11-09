import type {
  CompletedStages,
  Project,
  ProjectStage,
  UnlearningMethod,
  UnlearningMethodInfo
} from "./types";

export const defaultTemplate = "{question}";

export function buildPromptTemplate(userPrompt: string) {
  if (!userPrompt.includes("{question}")) {
    throw new Error("Prompt template must contain {question}.");
  }
  return userPrompt;
}

export const defaultLoraTargets = ["q_proj", "v_proj", "k_proj", "o_proj"];

export const fallbackUnlearningMethods: UnlearningMethodInfo[] = [
  { value: "grad_ascent", label: "Gradient Ascent", requires_retain: false },
  { value: "grad_diff", label: "Gradient Difference", requires_retain: true },
  { value: "npo", label: "NPO", requires_retain: false },
  { value: "dpo", label: "DPO", requires_retain: false },
  { value: "simnpo", label: "SimNPO", requires_retain: false },
  { value: "rmu", label: "RMU", requires_retain: false }
];

export const unlearningMethods: UnlearningMethod[] = fallbackUnlearningMethods.map(
  (method) => method.value
);

export const unlearningMethodLabels: Record<UnlearningMethod, string> = Object.fromEntries(
  fallbackUnlearningMethods.map((method) => [method.value, method.label] as const)
) as Record<UnlearningMethod, string>;

export const retainRequiredMethods: UnlearningMethod[] = fallbackUnlearningMethods
  .filter((method) => method.requires_retain)
  .map((method) => method.value);

export function createDefaultProject(name = "Unnamed project"): Project {
  const now = new Date().toISOString();

  return {
    id: crypto.randomUUID(),
    name,
    createdAt: now,
    updatedAt: now,
    lastStage: "gpu",
    completedStages: {
      gpu: false,
      data: false,
      model: false,
      hyperparameters: false,
      running: false,
      evaluation: false
    },
    setupComplete: false,
    pendingResetFrom: null,
    data: {
      sourceMode: "upload",
      forgetFile: null,
      retainFile: null,
      fullFile: null,
      poisonFile: null,
      extractionModelName: "meta-llama/Llama-3.2-1B-Instruct",
      extractionHfKey: "",
      extractionMaxLength: 512,
      gradientBatchSize: 2,
      extractionAdaptorPath: "",
      selectionMethod: "raslik",
      forgetSize: 100,
      retainSize: 100,
      graceTopN: 400,
      graceNumClusters: 10,
      keepGradients: false,
      extractionJob: null,
      extractionResponse: null,
      preparedForgetFile: null,
      preparedRetainFile: null,
      promptTemplate: defaultTemplate,
      previewRows: [],
      selectedPreviewKeys: [],
      previewReady: false,
      previewFilterable: true,
      uploadResponse: null,
      backendUploadError: null
    },
    model: {
      modelName: "meta-llama/Llama-2-7b-hf",
      method: "full",
      adaptorPath: "",
      hfKey: "",
      gpuIds: [],
      selectedTargets: [...defaultLoraTargets]
    },
    hyperparameters: {
      unlearningMethod: "simnpo",
      stepMode: "max_steps",
      maxSteps: 100,
      epochs: 1,
      learningRate: "1e-4",
      contextLength: 2048,
      batchSize: 1,
      gradAccum: 8,
      dpoBeta: "0.1",
      npoBeta: "0.1",
      simnpoBeta: "4.5",
      simnpoDelta: "0.0",
      rmuLayersToTransform: "22, 23, 24, 25, 26, 27, 28",
      rmuModuleRegex: String.raw`.*layers\.28$`,
      rmuSteeringCoeff: "20.0",
      forgettingStrength: "1.0",
      retentionStrength: "1.0",
      weightDecay: 0.01,
      saveSteps: 10
    },
    run: {
      config: null,
      training: null,
      message: null,
      embeddingModelName: "",
      evaluationMaxNewTokens: 256,
      evaluationBatchSize: 4,
      includeBenchmarks: false,
      evaluationJob: null,
      evaluation: null,
      evaluationMessage: null
    }
  };
}

function migratedCompletion(lastStage: ProjectStage, hasTraining: boolean, hasEvaluation: boolean): CompletedStages {
  const stageIndex = ["data", "model", "hyperparameters", "running", "evaluation"].indexOf(lastStage);
  return {
    gpu: false,
    data: stageIndex >= 1,
    model: stageIndex >= 2,
    hyperparameters: stageIndex >= 3,
    running: hasTraining,
    evaluation: stageIndex >= 4 && hasEvaluation
  };
}

export function normalizeProject(value: unknown): Project {
  const raw = (value ?? {}) as Partial<Project> & Record<string, unknown>;
  const base = createDefaultProject(typeof raw.name === "string" ? raw.name : "Unnamed project");
  const lastStage: ProjectStage =
    raw.lastStage === "gpu" ||
    raw.lastStage === "data" ||
    raw.lastStage === "model" ||
    raw.lastStage === "hyperparameters" ||
    raw.lastStage === "running" ||
    raw.lastStage === "evaluation"
      ? raw.lastStage
      : "gpu";

  const rawRun = (raw.run ?? {}) as Partial<Project["run"]>;
  const rawData = (raw.data ?? {}) as Partial<Project["data"]>;
  const rawModel = (raw.model ?? {}) as Partial<Project["model"]>;
  const rawHyperparameters = (raw.hyperparameters ?? {}) as Partial<Project["hyperparameters"]>;
  const completion = raw.completedStages ?? migratedCompletion(
    lastStage,
    rawRun.training !== null && rawRun.training !== undefined,
    rawRun.evaluation !== null && rawRun.evaluation !== undefined
  );

  return {
    ...base,
    ...raw,
    id: typeof raw.id === "string" ? raw.id : base.id,
    name: typeof raw.name === "string" ? raw.name : base.name,
    createdAt: typeof raw.createdAt === "string" ? raw.createdAt : base.createdAt,
    updatedAt: typeof raw.updatedAt === "string" ? raw.updatedAt : base.updatedAt,
    lastStage,
    completedStages: {
      ...base.completedStages,
      ...completion
    },
    setupComplete: typeof raw.setupComplete === "boolean" ? raw.setupComplete : true,
    pendingResetFrom: raw.pendingResetFrom ?? null,
    data: {
      ...base.data,
      ...rawData,
      sourceMode: rawData.sourceMode === "extract" ? "extract" : "upload",
      promptTemplate:
        typeof rawData.promptTemplate === "string"
          ? rawData.promptTemplate
          : defaultTemplate,
      previewRows: Array.isArray(rawData.previewRows) ? rawData.previewRows : [],
      selectedPreviewKeys: Array.isArray(rawData.selectedPreviewKeys) ? rawData.selectedPreviewKeys : []
    },
    model: {
      ...base.model,
      ...rawModel,
      selectedTargets:
        Array.isArray(rawModel.selectedTargets) && rawModel.selectedTargets.length > 0
          ? rawModel.selectedTargets
          : [...defaultLoraTargets]
    },
    hyperparameters: {
      ...base.hyperparameters,
      ...rawHyperparameters,
      unlearningMethod: unlearningMethods.includes(rawHyperparameters.unlearningMethod as UnlearningMethod)
        ? (rawHyperparameters.unlearningMethod as UnlearningMethod)
        : base.hyperparameters.unlearningMethod,
      epochs: Number(rawHyperparameters.epochs ?? base.hyperparameters.epochs) || 1
    },
    run: {
      ...base.run,
      ...rawRun
    }
  };
}
