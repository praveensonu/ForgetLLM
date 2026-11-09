import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu.gpu_utils import get_available_gpu_ids, selected_gpu_ids, validate_gpu_ids
from training_process import TrainingProcessManager
from evaluation_process import EvaluationProcessManager
from eval_orchestrator import _collect_model_outputs
from data_selection.caching import run_mp_main


def report_visible_gpus(config):
    return {"visible": os.environ.get("CUDA_VISIBLE_DEVICES")}


class GpuSelectionTests(unittest.TestCase):
    def test_order_and_legacy_selection(self):
        self.assertEqual(selected_gpu_ids({"gpu_ids": [6, 2]}), [6, 2])
        self.assertEqual(selected_gpu_ids({"gpu_id": 3}), [3])

    def test_invalid_selections(self):
        for ids in ([], [2, 2], [-1], [True], ["2"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                selected_gpu_ids({"gpu_ids": ids})

    @patch("gpu.gpu_utils.get_gpu_info")
    def test_only_free_gpus_are_available(self, info):
        info.return_value = [{"id": 2, "is_available": True}, {"id": 6, "is_available": False}]
        self.assertEqual(get_available_gpu_ids(), [2])
        self.assertEqual(validate_gpu_ids([6, 2]), [6, 2])
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_gpu_ids([7])

    def test_training_worker_visibility_and_server_isolation(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            outcome = TrainingProcessManager(runner=report_visible_gpus).run({"gpu": {"gpu_ids": [6, 2]}})
            self.assertEqual(outcome.result, {"visible": "6,2"})
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_reference_training_worker_selects_first_gpu_before_runner(self):
        for method in ("dpo", "npo", "rmu"):
            with self.subTest(method=method), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
                outcome = TrainingProcessManager(runner=report_visible_gpus).run({
                    "gpu": {"gpu_ids": [6, 2]}, "unlearning": {"method": method}
                })
                self.assertEqual(outcome.result, {"visible": "6"})
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_evaluation_worker_keeps_generation_selection(self):
        outcome = EvaluationProcessManager(runner=report_visible_gpus).run({"orchestrator_config": {"gpu": {"gpu_ids": [6, 2]}}})
        self.assertEqual(outcome, {"visible": "6,2"})

    @patch("data_selection.caching.subprocess.Popen")
    def test_extraction_sets_visibility_before_child_starts(self, popen):
        popen.return_value.poll.return_value = 0
        popen.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"gpu_ids": [6, 2]}))
            run_mp_main(config, Path(directory) / "run.log")
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6,2")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    @patch("eval_orchestrator._collect_model_outputs_phase")
    def test_scoring_single_gpu_generation_multiple(self, collect):
        calls = []
        collect.side_effect = lambda **kwargs: calls.append((kwargs["phase"], os.environ["UL_MODEL_DEVICE_MAP"]))
        with patch.dict(os.environ, {"UL_MODEL_DEVICE_MAP": "original"}):
            _collect_model_outputs(gpu_ids=[6, 2])
            self.assertEqual(calls, [("metrics", "cuda:0"), ("generation", "balanced")])
            self.assertEqual(os.environ["UL_MODEL_DEVICE_MAP"], "original")

    @patch("eval_orchestrator._collect_model_outputs_phase")
    def test_single_gpu_does_not_reload_for_generation(self, collect):
        _collect_model_outputs(gpu_ids=[6])
        collect.assert_called_once_with(gpu_ids=[6], phase="both")

    def test_evaluation_phases_do_not_mix_scoring_and_generation(self):
        import pandas as pd
        from eval.eval_utils import compute_model_outputs
        data = pd.DataFrame([{"question": "q", "answer": "a", "num_tokens": 2}])
        with patch("eval.eval_utils.get_probs_ppl_batch", return_value=([0.5], [2.0])) as score, patch(
            "eval.eval_utils.generate_outputs_batch", return_value=["generated"]
        ) as generate:
            metrics = compute_model_outputs(data, None, None, None, "test", phase="metrics")
            score.assert_called_once()
            generate.assert_not_called()
            self.assertEqual(metrics["perplexity"].tolist(), [2.0])
            score.reset_mock()
            outputs = compute_model_outputs(data, None, None, None, "test", phase="generation")
            score.assert_not_called()
            generate.assert_called_once()
            self.assertEqual(outputs["gen_answer"].tolist(), ["generated"])

    def test_gpu_endpoint_schema_and_failure(self):
        from fastapi import HTTPException
        from routes.gpu_routes import list_gpus
        from schemas import GpuListResponse
        with patch("routes.gpu_routes.get_gpu_info", return_value=[]):
            self.assertEqual(GpuListResponse.model_validate(list_gpus()).model_dump(), {
                "gpus": [], "available_gpu_ids": []
            })
        with patch("routes.gpu_routes.get_gpu_info", side_effect=RuntimeError("No driver")):
            with self.assertRaises(HTTPException) as error:
                list_gpus()
            self.assertEqual(error.exception.status_code, 503)



if __name__ == "__main__":
    unittest.main()
