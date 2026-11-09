import { useEffect, useRef, useState } from "react";
import { getUnlearningMethods } from "../../api";
import { fallbackUnlearningMethods } from "../../defaults";
import { useProject } from "../../state/ProjectContext";
import type { StepMode, UnlearningMethod, UnlearningMethodInfo } from "../../types";
import { FieldLabel, fieldHelp } from "../ui/HelpTip";
import { defaultRmuModuleRegex, parseRmuLayers, rmuTrainableParamsRegex } from "../../utils/rmu";

const contextOptions = [512, 1024, 2048, 4096, 8192];

export default function HyperparametersStage() {
  const { project, updateHyperparameters } = useProject();
  const h = project.hyperparameters;
  const [methods, setMethods] = useState<UnlearningMethodInfo[]>(fallbackUnlearningMethods);

  useEffect(() => {
    let cancelled = false;
    getUnlearningMethods()
      .then((discovered) => {
        if (!cancelled && discovered.length > 0) {
          setMethods(discovered);
        }
      })
      .catch(() => {
        // The backend may be offline during frontend-only development.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const betaKey = h.unlearningMethod === "dpo" ? "dpoBeta"
    : h.unlearningMethod === "npo" ? "npoBeta"
      : h.unlearningMethod === "simnpo" ? "simnpoBeta" : null;
  const selected = methods.find((method) => method.value === h.unlearningMethod);
  const hasRetainSet = Boolean(project.data.preparedRetainFile ?? project.data.retainFile);
  const retainMissing = Boolean(selected?.requires_retain) && !hasRetainSet;
  const rmuLayers = parseRmuLayers(h.rmuLayersToTransform);
  const lastValidRmuLayers = useRef(rmuLayers);

  function changeRmuLayers(value: string) {
    const nextLayers = parseRmuLayers(value);
    const previousLayers = lastValidRmuLayers.current;
    const followsLayers = previousLayers && h.rmuModuleRegex === defaultRmuModuleRegex(previousLayers);
    if (nextLayers) lastValidRmuLayers.current = nextLayers;
    updateHyperparameters({
      rmuLayersToTransform: value,
      ...(nextLayers && followsLayers ? { rmuModuleRegex: defaultRmuModuleRegex(nextLayers) } : {}),
    });
  }

  function changeMode(mode: StepMode) {
    if (mode === "epochs") {
      updateHyperparameters({ stepMode: "epochs", epochs: h.epochs > 0 ? h.epochs : 1 });
    } else {
      updateHyperparameters({ stepMode: "max_steps", maxSteps: h.maxSteps > 0 ? h.maxSteps : 100 });
    }
  }

  return (
    <section className="stage-panel">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 4</span>
        <h1>Hyperparameters</h1>
        <p>Choose either max steps or epochs, then configure the remaining training parameters.</p>
      </div>

      <div className="stage-form">
        <div className="field-grid two">
          <label className="field">
            <span>Forgetting strength</span>
            <input type="number" min={0} step="any" value={h.forgettingStrength}
              onChange={(event) => updateHyperparameters({ forgettingStrength: event.target.value })} />
          </label>
          <label className="field">
            <span>Retention strength</span>
            <input type="number" min={0} step="any" value={h.retentionStrength}
              onChange={(event) => updateHyperparameters({ retentionStrength: event.target.value })} />
          </label>
        </div>
        <label className="field">
          <FieldLabel help={fieldHelp.unlearningMethod}>Unlearning method</FieldLabel>
          <select
            value={h.unlearningMethod}
            onChange={(event) =>
              updateHyperparameters({ unlearningMethod: event.target.value as UnlearningMethod })
            }
          >
            {methods.map((method) => (
              <option key={method.value} value={method.value}>{method.label}</option>
            ))}
          </select>
          <small>
            {retainMissing
              ? `${selected?.label ?? h.unlearningMethod} needs a retain set: go back to the data stage and add one.`
              : "The strengths apply across methods; retention strength is used when training includes a retain loss."}
          </small>
        </label>

        {betaKey && (
          <div className="field-grid two">
            {h.unlearningMethod === "simnpo" && (
              <label className="field">
                <span>Delta</span>
                <input type="number" step="any" value={h.simnpoDelta}
                  onChange={(event) => updateHyperparameters({ simnpoDelta: event.target.value })} />
              </label>
            )}
            <label className="field">
              <span>Beta</span>
              <input type="number" min={0} step="any" value={h[betaKey]}
                onChange={(event) => updateHyperparameters({ [betaKey]: event.target.value })} />
              <small>Must be greater than zero.</small>
            </label>
          </div>
        )}

        {h.unlearningMethod === "rmu" && (
          <div className="field-grid two">
            <label className="field">
              <span>Layers to transform</span>
              <input value={h.rmuLayersToTransform} onChange={(event) => changeRmuLayers(event.target.value)}
                aria-invalid={rmuLayers === null} placeholder="22, 23, 24, 25, 26, 27, 28" />
              <small>Comma-separated, zero-based layer indices. Also selects layers for new LoRA adapters.</small>
              {rmuLayers === null && <small role="alert">Enter distinct, non-negative integers separated by commas.</small>}
            </label>
            <label className="field">
              <span>Steering coefficient</span>
              <input type="number" min={0} step="any" value={h.rmuSteeringCoeff}
                onChange={(event) => updateHyperparameters({ rmuSteeringCoeff: event.target.value })} />
              <small>Magnitude of the fixed random activation target.</small>
            </label>
            <label className="field">
              <span>Module regex</span>
              <input value={h.rmuModuleRegex}
                onChange={(event) => updateHyperparameters({ rmuModuleRegex: event.target.value })} />
              <small>Must match exactly one activation module. Defaults to the highest selected layer.</small>
            </label>
            <label className="field">
              <span>Trainable parameters regex</span>
              <input readOnly value={rmuLayers ? rmuTrainableParamsRegex(rmuLayers) : ""} />
              <small>Generated from the layers above. Existing frozen base weights stay frozen when training adapters.</small>
            </label>
          </div>
        )}

        <div className="field">
          <FieldLabel help={fieldHelp.trainingLength}>Unlearning length</FieldLabel>
          <div className="segmented compact" role="group" aria-label="Training length mode">
            {(["max_steps", "epochs"] as StepMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={h.stepMode === mode ? "active" : ""}
                onClick={() => changeMode(mode)}
              >
                {mode === "max_steps" ? "Max steps" : "Epochs"}
              </button>
            ))}
          </div>
          <small>
            Only one is sent to the orchestrator: the other value is always <code>null</code>.
          </small>
        </div>

        <div className="field-grid three">
          {h.stepMode === "max_steps" ? (
            <label className="field">
              <FieldLabel help={fieldHelp.trainingLength}>Max steps</FieldLabel>
              <input
                type="number"
                min={1}
                value={h.maxSteps}
                onChange={(event) => updateHyperparameters({ maxSteps: Number(event.target.value) })}
              />
            </label>
          ) : (
            <label className="field">
              <FieldLabel help={fieldHelp.trainingLength}>Epochs</FieldLabel>
              <input
                type="number"
                min={1}
                value={h.epochs}
                onChange={(event) => updateHyperparameters({ epochs: Number(event.target.value) })}
              />
            </label>
          )}

          <label className="field">
            <FieldLabel help={fieldHelp.learningRate}>Learning rate</FieldLabel>
            <input
              value={h.learningRate}
              onChange={(event) => updateHyperparameters({ learningRate: event.target.value })}
            />
          </label>

          <label className="field">
            <FieldLabel help={fieldHelp.contextLength}>Context length</FieldLabel>
            <select
              value={h.contextLength}
              onChange={(event) => updateHyperparameters({ contextLength: Number(event.target.value) })}
            >
              {contextOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>

          <label className="field">
            <FieldLabel help={fieldHelp.batchSize}>Batch size</FieldLabel>
            <input
              type="number"
              min={1}
              value={h.batchSize}
              onChange={(event) => updateHyperparameters({ batchSize: Number(event.target.value) })}
            />
          </label>

          <label className="field">
            <FieldLabel help={fieldHelp.gradAccum}>Gradient accumulation</FieldLabel>
            <input
              type="number"
              min={1}
              value={h.gradAccum}
              onChange={(event) => updateHyperparameters({ gradAccum: Number(event.target.value) })}
            />
          </label>

          <label className="field">
            <FieldLabel help={fieldHelp.saveSteps}>Save steps</FieldLabel>
            <input
              type="number"
              min={1}
              value={h.saveSteps}
              onChange={(event) => updateHyperparameters({ saveSteps: Number(event.target.value) })}
            />
          </label>

          <label className="field">
            <FieldLabel help={fieldHelp.weightDecay}>Weight decay</FieldLabel>
            <input
              type="number"
              min={0}
              step="0.001"
              value={h.weightDecay}
              onChange={(event) => updateHyperparameters({ weightDecay: Number(event.target.value) })}
            />
          </label>
        </div>
      </div>
    </section>
  );
}
