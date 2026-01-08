from config import *
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from model.model import RecommenderDQN
import os
import tensorflow as tf
from typing import Dict, List, Tuple


# ========== Utils ==========
def update_user_tags(user_tags, dish_tags, action: Action, alpha=0.1):
    
    if len(user_tags) != len(dish_tags):
        raise ValueError(f"Số tags không khớp: user_tags={len(user_tags)}, dish_tags={len(dish_tags)}")
    
    delta = alpha * reward(action) * dish_tags
    next_state = user_tags + delta
    next_state = np.clip(next_state, 0.0, 5.0)
    return next_state


def reward(action: Action) -> float:
    rewards = {
        Action.DETAILS: 0.2,
        Action.ADD_TO_CART: 0.6,
        Action.ORDER: 1.0,
        Action.CANCEL_ORDER: -0.8,
        Action.REMOVE_FROM_CART: -0.2
    }
    return rewards.get(action, 0.0)


def align_tag_vectors(user_bias: Dict[str, float], dish_tags: List[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    tag_order = list(user_bias.keys())
    for tag in dish_tags:
        if tag not in user_bias:
            tag_order.append(tag)

    dish_set = set(dish_tags)
    user_vec = np.array([float(user_bias.get(tag, 0.0)) for tag in tag_order], dtype=np.float32)
    dish_vec = np.array([1.0 if tag in dish_set else 0.0 for tag in tag_order], dtype=np.float32)
    return tag_order, user_vec, dish_vec


def vector_to_named_dict(tag_order: List[str], values: np.ndarray) -> Dict[str, float]:
    return {tag: float(val) for tag, val in zip(tag_order, values.tolist())}


def validate_user_tags(user_tags, expected_length=None):
    from training import get_actual_num_tags

    """Validate và chuẩn hóa user_tags"""
    if expected_length is None:
        expected_length = get_actual_num_tags()
    
    if len(user_tags) != expected_length:
        print(f"⚠️ Cảnh báo: user_tags có {len(user_tags)} tags, nhưng model expect {expected_length}")
        # Tự động điều chỉnh: pad hoặc crop
        if len(user_tags) < expected_length:
            # Pad với 0
            padded = np.zeros(expected_length)
            padded[:len(user_tags)] = user_tags
            return padded
        else:
            # Crop bớt
            return user_tags[:expected_length]
    return user_tags


def fetch_item_rating():
    try:
        resp = requests.get(
            "http://localhost:8080/api/v1/dishes/average_rating"
            # auth=HTTPBasicAuth("admin@gmail.com", "12345678")
        )
        resp.raise_for_status()
        return np.array(resp.json(), dtype=np.float32)
    except requests.RequestException as exc:
        print(f"⚠️ Không thể gọi API average_rating: {exc}. Dùng mặc định 1.0.")
        return np.ones(NUM_ITEMS, dtype=np.float32)


def fetch_items():
    try:
        resp = requests.get(
            "http://localhost:8080/api/v1/dishes/allIds"
            # auth=HTTPBasicAuth("admin@gmail.com", "12345678")
        )
        resp.raise_for_status()
        return np.array(resp.json(), dtype=np.int32)
    except requests.RequestException as exc:
        print(f"⚠️ Không thể gọi API allIds: {exc}. Dùng dãy 1..{NUM_ITEMS}.")
        return np.arange(1, NUM_ITEMS + 1, dtype=np.int32)


def save_models(policy_net, target_net, MODEL_DIR, POLICY_NET_PATH, TARGET_NET_PATH):
    """Lưu cả policy_net và target_net"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # SỬA: Thêm .keras vào filepath trước khi gọi save()
    policy_net_path = POLICY_NET_PATH + ".keras"
    target_net_path = TARGET_NET_PATH + ".keras"
    
    policy_net.save(policy_net_path)
    target_net.save(target_net_path)
    print(f"✅ Models saved to {MODEL_DIR}")
    

def load_models(policy_net, target_net, MODEL_DIR, POLICY_NET_PATH, TARGET_NET_PATH):
    # SỬA: Thêm .keras vào filepath
    policy_net_path = POLICY_NET_PATH + ".keras"
    target_net_path = TARGET_NET_PATH + ".keras"
    
    if os.path.exists(policy_net_path) and os.path.exists(target_net_path):
        try:
            # Load weights vào models đã khởi tạo
            policy_net_load = tf.keras.models.load_model(
                policy_net_path,  # Đã có .keras
                custom_objects={'RecommenderDQN': RecommenderDQN}
            )
            target_net_load = tf.keras.models.load_model(
                target_net_path,  # Đã có .keras
                custom_objects={'RecommenderDQN': RecommenderDQN}
            )
            
            # Copy weights từ loaded models sang current models
            policy_net.set_weights(policy_net_load.get_weights())
            target_net.set_weights(target_net_load.get_weights())
            
            print(f"✅ Models loaded from {MODEL_DIR}")
            return True
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            return False
    else:
        print("⚠️ No saved models found. Using initialized models.")
        return False