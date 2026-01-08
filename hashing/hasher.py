import mmh3
import re
from typing import Tuple

class FeatureHasher:
    def __init__(self, dim: int = 200_000, seed: int = 2025):
        self.dim = dim
        self.seed = seed

    @staticmethod
    def normalize_tag(tag: str) -> str:
        # simple normalization: lowercase, strip, collapse spaces to '-'
        t = tag.strip().lower()
        t = re.sub(r"\s+", "-", t)
        return t

    def index_and_sign(self, key: str) -> Tuple[int, int]:
        """
        Returns (index, sign) for a given key using mmh3 hashing.
        sign in {+1, -1}
        """
        h = mmh3.hash(key, self.seed, signed=True)
        idx = abs(h) % self.dim
        sign = 1 if h >= 0 else -1
        return idx, sign

    def featurize_sparse(self, keys_with_values):
        """
        Build a sparse dict {index: value} from iterable of (key, value).
        keys_with_values: Iterable[Tuple[str, float]]
        """
        vec = {}
        for key, val in keys_with_values:
            idx, sign = self.index_and_sign(key)
            vec[idx] = vec.get(idx, 0.0) + sign * float(val)
        return vec