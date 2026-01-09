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

# Target network synchronization (soft-update parameter for stability)
TARGET_UPDATE_RATE = 0.001  # τ in Polyak averaging: target_weights = (1-τ)*target_weights + τ*policy_weights


def soft_update_target_network():
    """Polyak averaging: gradually move target_net toward policy_net. Improves stability."""
    for target_var, policy_var in zip(target_net.trainable_variables, policy_net.trainable_variables):
        target_var.assign(TARGET_UPDATE_RATE * policy_var + (1.0 - TARGET_UPDATE_RATE) * target_var)


# ========== Training Step ==========
# NOTE: This is VALUE REGRESSION with target network (stable Q-learning), NOT Double DQN.
# 
# Why NOT Double DQN?
#   - Double DQN requires discrete action dimension [state → Q(state, action1), Q(state, action2), ...]
#   - Your model outputs a single scalar: [state → scalar Q-value]
#   - Problem domain: contextual bandit (rank items), not MDP (select actions)
#   - Double DQN's benefit (reduce action overestimation) is irrelevant here
#
# Current approach (Stable Value Regression):
#   1. policy_net: learns from training data (Bellman target via target_net)
#   2. target_net: fixed reference for Q-value targets (soft-updated each step)
#   3. This decouples learning target from learning policy → reduces divergence
#

@tf.function
def train_step(batch_user_biases, batch_item_tags, batch_rewards, batch_next_user_biases):
    """
    Stable Q-learning with target network (NOT Double DQN, which doesn't apply to value regression).
    
    Args:
        batch_user_biases: list[dict] - user preference vectors
        batch_item_tags: list[list[str]] - item tag lists
        batch_rewards: tensor (B,) - scalar rewards
        batch_next_user_biases: list[dict] - next user state
    
    Returns:
        loss: scalar training loss
    """
    hasher = FeatureHasher(dim=200_000, seed=2025)
    projector = TenDimProjector()
    batch_size = len(batch_rewards)

    # Build feature vectors for current and next states
    X = []
    X_next = []
    for i in range(batch_size):
        # Current state: (user_bias, item_tags) → 10D feature vector
        f = projector.project(batch_user_biases[i], batch_item_tags[i], {}, hasher)
        # Next state: (next_user_bias, same_item_tags) → 10D feature vector
        f_next = projector.project(batch_next_user_biases[i], batch_item_tags[i], {}, hasher)
        X.append(f)
        X_next.append(f_next)
    
    X = tf.constant(np.stack(X, axis=0), dtype=tf.float32)
    X_next = tf.constant(np.stack(X_next, axis=0), dtype=tf.float32)

    # ============ Forward Pass ============
    with tf.GradientTape() as tape:
        # Predicted Q-value for current state (updated each iteration)
        q_pred = tf.squeeze(policy_net(X, training=True), axis=1)  # shape [B]
        
        # Target Q-value estimate from NEXT state (using target_net, NOT policy_net)
        # This is stable because target_net changes slowly (soft-updated, not trained directly)
        q_next = tf.squeeze(target_net(X_next, training=False), axis=1)  # shape [B]
        
        # Bellman target: r + γ * Q(s', π)
        # Note: NO action selection here because there are no discrete actions (value prediction task)
        target = batch_rewards + GAMMA * q_next
        
        # Temporal difference loss: minimize (policy_output - target)²
        loss = huber_loss(target, q_pred)

    # ============ Backward Pass ============
    grads = tape.gradient(loss, policy_net.trainable_variables)
    
    # Clip gradients to prevent exploding gradients
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    
    return loss



# ========== Train from replay ==========
def train_from_replay(batch):
    """
    Sample a batch from replay buffer and perform one gradient step.
    Automatically soft-updates target_net after training.
    
    Args:
        batch: list of Transition(user_bias, item_tags, reward, next_user_bias)
    
    Returns:
        loss: scalar training loss
    """
    batch_user_biases = [t.user_bias for t in batch]
    batch_item_tags = [t.item_tags for t in batch]
    batch_rewards = tf.constant([t.reward for t in batch], dtype=tf.float32)
    batch_next_user_biases = [t.next_user_bias for t in batch]

    loss = train_step(batch_user_biases, batch_item_tags, batch_rewards, batch_next_user_biases)
    
    # Soft-update target network toward policy network (Polyak averaging)
    # This ensures the Q-target is stable and changes slowly
    soft_update_target_network()
    
    # Save models after training
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