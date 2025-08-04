import json
import logging
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RecommendationMetrics:
    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    average_hit_rate: float
    total_scenarios: int
    top_1_hits: int
    top_3_hits: int
    top_5_hits: int



class BaseEvaluator:
    """Base class for evaluation tools"""
    def __init__(self):
        self.metrics_history: List[RecommendationMetrics] = []

    def save_metrics(self, metrics: RecommendationMetrics):
        """Save metrics to history"""
        self.metrics_history.append(metrics)

    def get_metrics_history(self):
        """Get all historical metrics"""
        return self.metrics_history

class RecommendationEvaluator(BaseEvaluator):
    """Evaluator for recommendation tasks"""
    
    def __init__(self):
        super().__init__()
        self.n_values = [1, 3, 5]  # 预定义的n值数组

    def calculate_hr_at_n(
        self,
        ground_truth: List[str],
        predictions: List[List[str]]
    ) -> RecommendationMetrics:
        """Calculate Hit Rate at different N values"""
        total = len(ground_truth)
        hits = {n: 0 for n in self.n_values}
        
        for gt, pred in zip(ground_truth, predictions):
            for n in self.n_values:
                if gt in pred[:n]:
                    hits[n] += 1
        
        top_1_hit_rate = hits[1] / total if total > 0 else 0
        top_3_hit_rate = hits[3] / total if total > 0 else 0
        top_5_hit_rate = hits[5] / total if total > 0 else 0
        average_hit_rate = (top_1_hit_rate + top_3_hit_rate + top_5_hit_rate) / 3
        metrics = RecommendationMetrics(
            top_1_hit_rate=top_1_hit_rate,
            top_3_hit_rate=top_3_hit_rate,
            top_5_hit_rate=top_5_hit_rate,
            average_hit_rate=average_hit_rate,
            total_scenarios=total,
            top_1_hits=hits[1],
            top_3_hits=hits[3],
            top_5_hits=hits[5]
        )
        
        self.save_metrics(metrics)
        return metrics

