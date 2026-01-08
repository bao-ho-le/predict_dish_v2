import pickle
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Dict, Tuple

REPLAY_BUFFER_FILE = "replay_buffer.pkl"

# Transition mới
Transition = namedtuple('Transition', ('user_bias','item_tags','reward','next_user_bias'))

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buf = deque(maxlen=capacity)

    def push(self, user_bias: Dict[str, float], item_tags: List[str], reward: float, next_user_bias: Dict[str, float]):
        """Push một transition mới"""
        self.buf.append(Transition(user_bias, item_tags, reward, next_user_bias))

    def sample(self, batch_size: int) -> List[Transition]:
        """Lấy batch random các transition"""
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


# ======= Save / Load =======
def save_replay_buffer(buffer: ReplayBuffer, filename: str = REPLAY_BUFFER_FILE):
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"✅ Replay buffer saved to {filename}")
    

def load_replay_buffer(filename: str = REPLAY_BUFFER_FILE) -> ReplayBuffer:
    with open(filename, 'rb') as f:
        try:
            buffer = pickle.load(f)
        except TypeError as exc:
            print(f"⚠️ Replay buffer file không tương thích ({exc}). Khởi tạo buffer mới.")
            return ReplayBuffer()
    print(f"✅ Replay buffer loaded from {filename}, contains {len(buffer)} transitions")
    return buffer
