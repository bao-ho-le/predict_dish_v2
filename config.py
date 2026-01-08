from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional

# ========== Config ==========
NUM_USERS = 100
NUM_ITEMS = 100
NUM_TAGS  = 100
USER_EMB_DIM = 16
ITEM_EMB_DIM = 16
HIDDEN = 128
GAMMA = 0.99
LR = 1e-3
REPLAY_CAP = 20000
TARGET_UPDATE_EVERY = 500  
MAX_STEPS = 20000
EPS = 0.2
TOP_K = 10
NUM_EXPLORE = 2  
# BATCH_SIZE = 64
BATCH_SIZE = 10



# ========== Type ==========
class Action(str, Enum):
    DETAILS = "DETAILS"
    ADD_TO_CART = "ADD_TO_CART"
    ORDER = "ORDER"
    REMOVE_FROM_CART = "REMOVE_FROM_CART"
    CANCEL_ORDER = "CANCEL_ORDER"
    

action_map = {
    Action.DETAILS: 0,
    Action.ADD_TO_CART: 1,
    Action.ORDER: 2,
    Action.CANCEL_ORDER: 3,
    Action.REMOVE_FROM_CART: 4
}


class Candidate(BaseModel):
    dish_id: int
    item_tags: List[str]


class RecommendRequest(BaseModel):
    candidates: List[Candidate]
    user_bias: Dict[str, float]
    top_k: int = TOP_K
    ctx: Optional[Dict[str, str]] = None  
              


class UserAction(BaseModel):
    user_bias: Dict[str, float]
    dish_tags: List[str]
    action: Action
