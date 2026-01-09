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
        # User bias stats capture user-side preference strength
        bias_vals = list(user_bias.values()) if user_bias else [0.0]
        mean_bias = float(np.mean(bias_vals))
        bias_std = float(np.std(bias_vals))

        num_tags = float(len(item_norm))
        num_tags_log = float(np.log1p(num_tags) / np.log1p(50.0))  # bounded in [0,1]

        # Overlap features encode how much this item's tags align with the user
        overlap_sum = 0.0
        overlap_max = 0.0
        for t in item_norm:
            w = float(user_bias.get(t, 0.0))
            overlap_sum += w
            overlap_max = max(overlap_max, w)
        overlap_mean = overlap_sum / (num_tags + 1e-6)

        # Hashed tag signature (order-invariant) injects item identity without using item_id
        # Each tag contributes a signed value derived from its hash index; stats keep dimension fixed.
        signed_vals = []
        for t in item_norm:
            idx, sign = hasher.index_and_sign(t)
            signed = sign * (idx / float(hasher.dim))  # normalize index into [-1,1]
            signed_vals.append(signed)

        if not signed_vals:
            signed_vals = [0.0]

        signed_vals = np.array(signed_vals, dtype=np.float32)
        hash_signed_mean = float(np.mean(signed_vals))          # differentiates by index distribution
        hash_signed_std = float(np.std(signed_vals))            # spread of hashed positions
        abs_vals = np.abs(signed_vals)
        hash_abs_mean = float(np.mean(abs_vals))                # collision-resistant magnitude signal
        hash_abs_max = float(np.max(abs_vals))                  # highlights any single dominant tag hash
        hash_l2 = float(np.linalg.norm(signed_vals))            # overall energy, scales with count and positions

        # 10 features total; balance user stats, overlap, and hashed signatures to avoid collapse.
        features = np.array([
            mean_bias,          # user preference center
            bias_std,           # user preference spread
            overlap_mean,       # average alignment per tag
            overlap_max,        # strongest aligned tag
            num_tags_log,       # tag count (log scaled)
            hash_signed_mean,   # signed hash center
            hash_signed_std,    # signed hash spread
            hash_abs_mean,      # magnitude center (order-invariant)
            hash_abs_max,       # strongest hash magnitude
            hash_l2             # overall hash energy
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