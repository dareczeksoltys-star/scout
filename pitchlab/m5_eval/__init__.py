"""M5: Calibration evaluation module (test datasets, IoU metrics, reprojection error, and benchmark reports)."""
from pitchlab.m5_eval.dataset import EvalDataset, EvalSample
from pitchlab.m5_eval.metrics import EvaluationMetrics, evaluate_calibration
from pitchlab.m5_eval.reporter import run_benchmark_and_report

__all__ = [
    "EvalDataset",
    "EvalSample",
    "EvaluationMetrics",
    "evaluate_calibration",
    "run_benchmark_and_report",
]
