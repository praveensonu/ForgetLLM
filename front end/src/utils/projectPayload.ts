import type { DatasetUploadResponse, FinalTrainingConfigRequest, Project } from "../types";
import { parseRmuLayers, rmuTrainableParamsRegex } from "./rmu";

function optionalText(value: string) {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function buildTrainingPayload(
  project: Project,
  dataset: DatasetUploadResponse
): FinalTrainingConfigRequest {
  const h = project.hyperparameters;
  const rmuLayers = h.unlearningMethod === "rmu" ? parseRmuLayers(h.rmuLayersToTransform) : null;
  if (h.unlearningMethod === "rmu" && !rmuLayers) {
    throw new Error("RMU layers must be distinct, non-negative integers separated by commas.");
  }

  return {
    project_id: project.id,
    project_name: project.name,
    model_name: project.model.modelName.trim(),
    adaptor_path:
      project.model.method === "adaptor" ? optionalText(project.model.adaptorPath) : null,
    hf_key: optionalText(project.model.hfKey),
    method: project.model.method,
    unlearning_method: h.unlearningMethod,
    gpu_ids: project.model.gpuIds,
    forget_set_path: dataset.forget_set_path,
    retain_set_path: dataset.retain_set_path,
    hyperparams: {
      method: h.unlearningMethod === "rmu" && rmuLayers
        ? {
            layers_to_transform: rmuLayers,
            module_regex: h.rmuModuleRegex.trim(),
            trainable_params_regex: rmuTrainableParamsRegex(rmuLayers),
            steering_coeff: Number(h.rmuSteeringCoeff),
          }
        : h.unlearningMethod === "simnpo"
        ? { beta: Number(h.simnpoBeta), delta: Number(h.simnpoDelta) }
        : h.unlearningMethod === "dpo"
          ? { beta: Number(h.dpoBeta) }
          : h.unlearningMethod === "npo" ? { beta: Number(h.npoBeta) } : {},
      general: {
        max_steps: h.stepMode === "max_steps" ? h.maxSteps : null,
        epochs: h.stepMode === "epochs" ? h.epochs : null,
        learning_rate: h.learningRate,
        gamma: Number(h.forgettingStrength),
        alpha: Number(h.retentionStrength),
        context_length: h.contextLength
      },
      optimization: {
        batch_size: h.batchSize,
        grad_accum: h.gradAccum,
        weight_decay: h.weightDecay
      },
      schedule: {
        save_steps: h.saveSteps
      },
      memory: {
        assistant_completions_only: true
      },
      ...(project.model.method === "lora"
        ? {
            lora_settings: {
              target_modules: project.model.selectedTargets
            }
          }
        : {})
    }
  };
}
