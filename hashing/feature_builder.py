from typing import Dict, List, Tuple, Iterable, Any
import numpy as np
from .hasher import FeatureHasher

class TenDimProjector:
    """
    Example projector: combine sparse hashed vectors into fixed 10-d features.
    Option A: Trainable linear projection (load weights from model).
    Option B: Handcrafted summaries (mean/max/entropy/overlap/etc.).
    Here we show a simple handcrafted summary approach for clarity.
    """
    def __init__(self):
        pass

    @staticmethod
    def _entropy(values: List[float]) -> float:
        arr = np.array(values, dtype=np.float32)
        arr = np.clip(arr, 1e-6, None)
        p = arr / (np.sum(arr) + 1e-9)
        return float(-(p * np.log(p + 1e-9)).sum())

    def project(self,
                user_bias: Dict[str, float],
                item_tags: List[str],
                ctx: Dict[str, str],
                hasher: FeatureHasher) -> np.ndarray:
        # Normalize tags
        item_norm = [hasher.normalize_tag(t) for t in item_tags]
        # Summaries
        bias_vals = list(user_bias.values()) if user_bias else [0.0]
        mean_bias = float(np.mean(bias_vals))
        max_bias = float(np.max(bias_vals))
        entropy_bias = self._entropy(bias_vals)

        num_tags = float(len(item_norm))
        # overlap_sum: sum of bias[tag] for tags present in item
        overlap_sum = 0.0
        overlap_max = 0.0
        for t in item_norm:
            w = float(user_bias.get(t, 0.0))
            overlap_sum += w
            overlap_max = max(overlap_max, w)

        # simple context encoding: meal_slot if present
        meal_slot = ctx.get("meal_slot", "")
        meal_slot_code = {
            "breakfast": 0.0,
            "lunch": 0.5,
            "dinner": 1.0
        }.get(meal_slot.lower(), 0.0)

        # 10 features (example):
        # 1 mean_bias, 2 max_bias, 3 entropy_bias,
        # 4 num_tags, 5 overlap_sum, 6 overlap_max,
        # 7 meal_slot_code, 8 popularity (optional, here 0),
        # 9 item_tag_diversity (proxy = 1/num_tags), 10 bias_std
        bias_std = float(np.std(bias_vals))
        popularity = 0.0  # placeholder if you have it
        item_diversity_proxy = float(1.0 / (num_tags + 1e-6))

        features = np.array([
            mean_bias,
            max_bias,
            entropy_bias,
            num_tags,
            overlap_sum,
            overlap_max,
            meal_slot_code,
            popularity,
            item_diversity_proxy,
            bias_std
        ], dtype=np.float32)

        return features


def _flatten_tags(raw_tags: Iterable[Any]) -> List[str]:
    flat: List[str] = []
    for tag in raw_tags:
        if isinstance(tag, (list, tuple, set)):
            flat.extend(_flatten_tags(tag))
        else:
            flat.append(str(tag))
    return flat


def _extract_item_tags(candidate: Any) -> List[str]:
    if hasattr(candidate, "item_tags"):
        return _flatten_tags(getattr(candidate, "item_tags"))
    if isinstance(candidate, dict) and "item_tags" in candidate:
        return _flatten_tags(candidate["item_tags"])
    if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
        return _flatten_tags(candidate[1])
    raise ValueError("Candidate must provide item_tags field")


def build_batch_features(user_bias: Dict[str, float],
                         candidates: Iterable[Any],
                         ctx: Dict[str, str],
                         hasher: FeatureHasher,
                         projector: TenDimProjector) -> np.ndarray:
    """Build feature matrix for a batch of candidates."""
    X = []
    for candidate in candidates:
        item_tags = _extract_item_tags(candidate)
        x = projector.project(user_bias, item_tags, ctx, hasher)
        X.append(x)
    return np.stack(X, axis=0)