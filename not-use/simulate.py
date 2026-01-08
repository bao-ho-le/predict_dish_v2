import math, random, numpy as np
from config import *

# ========== Simple simulator ==========
random.seed(0)
np.random.seed(0)

ITEM_TAGS = np.random.uniform(low=1.0, high=5.0, size=(NUM_ITEMS, NUM_TAGS)).astype(np.float32)
ITEM_TAGS_NORM = (ITEM_TAGS - 1.0) / (5.0 - 1.0)

ITEM_RATINGS = np.clip(np.random.normal(loc=3.5, scale=0.7, size=(NUM_ITEMS,)), 1.0, 5.0)
ITEM_RATING_NORM = (ITEM_RATINGS - ITEM_RATINGS.min()) / (ITEM_RATINGS.max() - ITEM_RATINGS.min())

USER_TRUE_PREF = np.random.rand(NUM_USERS, NUM_TAGS).astype(np.float32)


def simulator_reward(user_id, item_idx):
    pref = USER_TRUE_PREF[user_id]
    item_t = ITEM_TAGS_NORM[item_idx]

    sim = np.dot(pref, item_t) / (np.linalg.norm(pref) * (np.linalg.norm(item_t)+1e-8) + 1e-8)  
    score = sim + 0.2 * (ITEM_RATING_NORM[item_idx])
    p = 1/(1+math.exp(-10*(score-0.1)))  
    
    return 1.0 if random.random() < p else 0.0

def update_user_tags_online(user_tags_vec, item_idx, positive):
    if positive:
        new = user_tags_vec + 0.2 * ITEM_TAGS_NORM[item_idx]
        return np.clip(new, 0.0, 1.0)
    else:
        return user_tags_vec  