import numpy as np
from typing import Dict, Any, Optional

class CamouflageEffectivenessMetrics:
    """
    Computes quantitative metrics for camouflage effectiveness across spectral bands.
    Supports contrast, spectral similarity, and detection probability estimation.
    """
    def __init__(self, panel_shape: tuple, bands: Optional[list] = None):
        self.panel_shape = panel_shape
        self.bands = bands if bands is not None else ['visible', 'ir', 'uv']

    def spectral_contrast(self, env_map: np.ndarray, camo_pattern: np.ndarray) -> float:
        # Lower is better (perfect match = 0)
        diff = env_map - camo_pattern
        return float(np.mean(np.abs(diff)))

    def spectral_similarity(self, env_map: np.ndarray, camo_pattern: np.ndarray) -> float:
        # Higher is better (perfect match = 1)
        numerator = np.sum(env_map * camo_pattern)
        denominator = np.sqrt(np.sum(env_map**2)) * np.sqrt(np.sum(camo_pattern**2)) + 1e-8
        return float(numerator / denominator)

    def aggregate_metric(self, env_maps: Dict[str, np.ndarray], camo_patterns: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> float:
        # Weighted sum of spectral similarities (higher = better)
        if weights is None:
            weights = {b: 1.0 for b in self.bands}
        total = 0.0
        for band in self.bands:
            sim = self.spectral_similarity(env_maps[band], camo_patterns[band])
            total += sim * weights.get(band, 1.0)
        return total / max(1e-8, sum(weights.values()))

class CamouflageOptimizer:
    """
    Optimizes camouflage patterns to maximize effectiveness metrics.
    Supports gradient-free search and iterative improvement.
    """
    def __init__(self, metrics: CamouflageEffectivenessMetrics):
        self.metrics = metrics

    def optimize(self, env_maps: Dict[str, np.ndarray], initial_patterns: Dict[str, np.ndarray], steps: int = 50, lr: float = 0.1) -> Dict[str, np.ndarray]:
        # Simple iterative optimization (gradient-free, random perturbation)
        best_patterns = {b: p.copy() for b, p in initial_patterns.items()}
        best_score = self.metrics.aggregate_metric(env_maps, best_patterns)
        for _ in range(steps):
            candidate = {b: np.clip(p + np.random.normal(0, lr, p.shape), 0.0, 1.0) for b, p in best_patterns.items()}
            score = self.metrics.aggregate_metric(env_maps, candidate)
            if score > best_score:
                best_patterns, best_score = candidate, score
        return best_patterns
