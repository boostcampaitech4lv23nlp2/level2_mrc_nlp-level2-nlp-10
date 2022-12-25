from datasets import load_metric
from transformers import EvalPrediction


class Metric:
    def __init__(self):
        self.metric = load_metric("squad")

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)
