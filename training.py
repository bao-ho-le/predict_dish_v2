from config import *
from model.model import *
import random, numpy as np
import tensorflow as tf
from replay_buffer.replaybuffer import (
    ReplayBuffer,
    load_replay_buffer,
    REPLAY_BUFFER_FILE as RAW_REPLAY_FILE
)
from utils import load_models, save_models, fetch_items, fetch_item_rating
from hashing.feature_builder import TenDimProjector
from hashing.hasher import FeatureHasher
import os
from typing import Optional


REPLAY_BUFFER_PATH = os.path.join("replay_buffer", RAW_REPLAY_FILE)
_replay_buffer: Optional[ReplayBuffer] = None


def _init_replay_buffer() -> ReplayBuffer:
    global _replay_buffer
    if _replay_buffer is not None:
        return _replay_buffer

    try:
        _replay_buffer = load_replay_buffer(REPLAY_BUFFER_PATH)
        print(f"✅ Đã load replay buffer từ file, kích thước: {len(_replay_buffer)}")
    except FileNotFoundError:
        _replay_buffer = ReplayBuffer(REPLAY_CAP)
        print("🆕 Tạo replay buffer mới")
    return _replay_buffer


def get_replay_buffer() -> ReplayBuffer:
    return _init_replay_buffer()


def set_replay_buffer(buffer: ReplayBuffer) -> None:
    global _replay_buffer
    _replay_buffer = buffer


replay = get_replay_buffer()

ITEMS = fetch_items()
ITEM_RATING_NORM = fetch_item_rating()


# ========== XÁC ĐỊNH SỐ TAGS THỰC TẾ ==========
def get_actual_num_tags():

    print(f"replay.buf length: {len(replay.buf)}")

    if len(replay.buf) > 0:
        sample = replay.buf[0]
        if isinstance(sample.user_bias, dict):
            length = len(sample.user_bias)
        else:
            length = len(getattr(sample, "user_tags", []))
        print(f"Sample user_bias length: {length}")
        return length if length > 0 else NUM_TAGS
    else:
        print("🚨 REPLAY BUFFER RỖNG - DÙNG SỐ TAGS MẶC ĐỊNH 21")
        return NUM_TAGS


ACTUAL_NUM_TAGS = get_actual_num_tags()
print(f"✅ Số tags thực tế: {ACTUAL_NUM_TAGS}")


# ========== Model Saving & Loading ==========
MODEL_DIR = "model"
POLICY_NET_PATH = os.path.join(MODEL_DIR, "policy_net")
TARGET_NET_PATH = os.path.join(MODEL_DIR, "target_net")

policy_net = RecommenderDQN()
target_net = RecommenderDQN()

# Đồng bộ weights
target_net.set_weights(policy_net.get_weights())

# Load weights nếu có
try:
    load_models(policy_net, target_net, MODEL_DIR, POLICY_NET_PATH, TARGET_NET_PATH)
except Exception as e:
    print(f"⚠️ Cannot load models: {e}. Using initialized models.")
    


optimizer = tf.keras.optimizers.Adam(LR)
huber_loss = tf.keras.losses.Huber()



# ========== Training Step ==========
@tf.function
def train_step(batch_user_biases, batch_item_tags, batch_rewards, batch_next_user_biases):
    """
    batch_user_biases: list[dict]
    batch_item_tags: list[list[str]]
    batch_rewards: tensor (B,)
    batch_next_user_biases: list[dict]
    """
    hasher = FeatureHasher(dim=200_000, seed=2025)
    projector = TenDimProjector()
    batch_size = len(batch_rewards)

    # build features batch
    X = []
    X_next = []
    for i in range(batch_size):
        f = projector.project(batch_user_biases[i], batch_item_tags[i], {}, hasher)
        f_next = projector.project(batch_next_user_biases[i], batch_item_tags[i], {}, hasher)
        X.append(f)
        X_next.append(f_next)
    
    X = tf.constant(np.stack(X, axis=0), dtype=tf.float32)
    X_next = tf.constant(np.stack(X_next, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        q_pred = tf.squeeze(policy_net(X, training=True), axis=1)
        q_next = tf.squeeze(target_net(X_next, training=False), axis=1)
        target = batch_rewards + GAMMA * q_next
        loss = huber_loss(target, q_pred)

    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    return loss



# ========== Train from replay ==========
def train_from_replay(batch):
    # batch = list of Transition(user_bias, item_tags, reward, next_user_bias)
    batch_user_biases = [t.user_bias for t in batch]
    batch_item_tags = [t.item_tags for t in batch]
    batch_rewards = tf.constant([t.reward for t in batch], dtype=tf.float32)
    batch_next_user_biases = [t.next_user_bias for t in batch]

    loss = train_step(batch_user_biases, batch_item_tags, batch_rewards, batch_next_user_biases)

    save_models(policy_net, target_net, MODEL_DIR, POLICY_NET_PATH, TARGET_NET_PATH)
    return loss









# epsilon schedule
# def epsilon_by_step(step):
#     return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)

# # helper to select top-k with exploration
# def recommend_for_user(user_bias, candidates, eps):
#     features = build_batch_features(
#         user_bias,
#         candidates,
#         feature_hasher,
#         projector
#     )






#     user_id_tf = tf.constant([[user_id]], dtype=tf.int32)
#     user_tags_tf = tf.constant(user_tags_vec.reshape(1,-1), dtype=tf.float32)

#     items_tf = tf.constant(ITEMS, dtype=tf.int32) 
#     items_tf = tf.reshape(items_tf, [1, -1])

#     qvals = policy_net.call_all_q(
#         user_id_tf, 
#         user_tags_tf, 
#         items_tf,
#         item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32)
#     )
    
#     q_np = qvals.numpy().flatten()
#     topk_indices = list(np.argsort(-q_np)[:TOP_K])

#     # SỬA: Dùng .tolist() để chuyển numpy array sang Python list
#     topk_item_ids = ITEMS[topk_indices].tolist()

#     if random.random() < eps:
#         candidate_indices = list(set(range(len(ITEMS))) - set(topk_indices))
#         if len(candidate_indices) >= NUM_EXPLORE:
#             explore_indices = random.sample(candidate_indices, NUM_EXPLORE)
#             explore_item_ids = ITEMS[explore_indices].tolist()
#             topk_item_ids[-NUM_EXPLORE:] = explore_item_ids
    
#     # SỬA: Chuyển q_np sang Python list
#     q_values_list = q_np.tolist()
    
#     return topk_item_ids, q_values_list