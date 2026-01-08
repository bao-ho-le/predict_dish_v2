import numpy as np, tensorflow as tf

from config import *
from model import *
from simulate import *
from training import *


# ========== Main loop demo ==========
# initialize simple user_tags state (start from zeros)
user_states = np.zeros((NUM_USERS, NUM_TAGS), dtype=np.float32)
# seed user_states a bit from true pref to speed up
for u in range(NUM_USERS):
    user_states[u] = (USER_TRUE_PREF[u] > 0.6).astype(np.float32)

steps = 0
loss_avg = 0.0
while steps < MAX_STEPS:
    # sample a random user to interact
    u = random.randrange(NUM_USERS)
    state = user_states[u].copy()
    eps = epsilon_by_step(steps)
    recs, q_np = recommend_for_user(u, state, eps)

    # simulate user interacting with top-1 (or user may choose any among recs)
    chosen = recs[0]  # for simplicity assume user considers the first recommended
    reward = simulator_reward(u, chosen)
    next_state = update_user_tags_online(state, chosen, reward)

    # push to replay
    replay.push(u, state, int(chosen), float(reward), next_state)

    # train if enough samples
    if len(replay) >= BATCH_SIZE:
        batch = replay.sample(BATCH_SIZE)
        b_user_ids = tf.constant(np.array(batch.user_id).reshape(-1,1), dtype=tf.int32)
        b_user_tags = tf.constant(np.array(batch.user_tags), dtype=tf.float32)
        b_actions = tf.constant(np.array(batch.action), dtype=tf.int32)
        b_rewards = tf.constant(np.array(batch.reward), dtype=tf.float32)
        b_next_user_tags = tf.constant(np.array(batch.next_user_tags), dtype=tf.float32)
        b_dones = tf.constant(np.array(batch.done), dtype=tf.float32)
        loss = train_step(b_user_ids, b_user_tags, b_actions, b_rewards, b_next_user_tags, b_dones)
        loss_avg = 0.99 * loss_avg + 0.01 * float(loss)

    # periodically update target network
    if steps % TARGET_UPDATE_EVERY == 0 and steps > 0:
        target_net.set_weights(policy_net.get_weights())

    # apply next_state
    user_states[u] = next_state

    if steps % 1000 == 0:
        print(f"step {steps} eps {eps:.3f} replay {len(replay)} loss_avg {loss_avg:.4f}")

    steps += 1

print("Training demo finished. Example recommendations for user 0:")
for uid in range(5):
    recs, q = recommend_for_user(uid, user_states[uid], eps=0.05)
    print("User", uid, "top10:", recs[:10])
