from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from pathlib import Path

from config import RecommendRequest, UserAction, BATCH_SIZE
from utils import update_user_tags, reward, align_tag_vectors, vector_to_named_dict
from replay_buffer.replaybuffer import *
from training import train_from_replay, ITEMS, ITEM_RATING_NORM, set_replay_buffer
from model.model import RecommenderDQN
from hashing.hasher import FeatureHasher
from hashing.feature_builder import TenDimProjector, build_batch_features


app = FastAPI()

MODEL_PATH = Path("model/policy_net.keras")


def _load_or_init_model() -> RecommenderDQN:
    if MODEL_PATH.exists():
        try:
            return tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={"RecommenderDQN": RecommenderDQN}
            )
        except Exception as exc:
            print(f"⚠️ Không thể load model đã lưu: {exc}. Khởi tạo model mới.")

    model = RecommenderDQN()
    _ = model(tf.zeros((1, 10), dtype=tf.float32))
    return model



inference_model = _load_or_init_model()

# Load replay buffer nếu file tồn tại
try:
    replay_buffer = load_replay_buffer("replay_buffer/" + REPLAY_BUFFER_FILE)
except FileNotFoundError:
    replay_buffer = ReplayBuffer(capacity=10000)
    print("Không có file cũ → tạo buffer mới.")

set_replay_buffer(replay_buffer)


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    if not req.candidates:
        return {}
    
    ctx = req.ctx or {}
    hasher = FeatureHasher(dim=200_000, seed=2025)
    projector = TenDimProjector()

    X = build_batch_features(
        req.user_bias,
        req.candidates,
        ctx,
        hasher,
        projector
    )

    predictions = inference_model(tf.constant(X, dtype=tf.float32), training=False)
    scores = predictions.numpy().flatten().tolist()

    scored = list(zip([c.dish_id for c in req.candidates], scores))
    scored.sort(key=lambda item: item[1], reverse=True)
    limit = min(max(req.top_k, 1), len(scored))
    top_candidates = scored[:limit]

    return {str(dish_id): float(score) for dish_id, score in top_candidates}
    

@app.put("/push_replay_buffer")
async def push_replay_buffer(req: UserAction):
    # Cần align tag vector để tính next_state, sau đó chuyển ngược về dict (vector_to_named_dict) để có thể đưa vào hash

    tag_order, user_vec, dish_vec = align_tag_vectors(req.user_bias, req.dish_tags)
    action = req.action

    next_state = update_user_tags(user_vec, dish_vec, action)
    r = reward(action)

    user_bias_update = vector_to_named_dict(tag_order, next_state)
    # kết quả: {"spicy": 2.5, "vegan": 1.0, "grill": 0.0}

    transition = Transition(
        user_bias= vector_to_named_dict(tag_order, user_vec),
        item_tags=list(req.dish_tags),
        reward=r,
        next_user_bias=user_bias_update
    )

    # đẩy vào replay buffer
    replay_buffer.push(*transition)
    save_replay_buffer(replay_buffer, "replay_buffer/" + REPLAY_BUFFER_FILE)

    if len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        loss = train_from_replay(batch)
        print("Model trained, loss =", loss)


    return user_bias_update



# @app.post("/reload_model")
# async def reload_model():
#     """Endpoint để reload model khi có thay đổi số tags"""
#     from training import policy_net, target_net, ACTUAL_NUM_TAGS
    
#     # Build lại model với số tags mới
#     _dummy_user_id = tf.constant([[0]])
#     _dummy_tags = tf.constant(np.zeros((1, ACTUAL_NUM_TAGS), dtype=np.float32))
#     _dummy_items = tf.constant([ITEMS.tolist()], dtype=tf.int32)
    
#     _ = policy_net.call_all_q(_dummy_user_id, _dummy_tags, _dummy_items, 
#                              item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32))
#     _ = target_net.call_all_q(_dummy_user_id, _dummy_tags, _dummy_items, 
#                              item_rating_norm=tf.constant(ITEM_RATING_NORM, dtype=tf.float32))
    
#     return {"status": "Model reloaded", "num_tags": ACTUAL_NUM_TAGS}


# @app.post("/recommend")
# def recommend(user: UserData):
#     topk, qvals = recommend_for_user(user.user_id, np.array(user.user_tags))
#     return {"recommended_items": topk, "scores": qvals}



