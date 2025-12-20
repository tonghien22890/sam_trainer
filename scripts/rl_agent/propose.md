def phi(c_opp, c_me, k=0.2):
    return k * (c_opp - c_me)

def step_reward(c_me_before, c_me_after, c_opp_before, c_opp_after,
                w1=0.5, w2=0.7, k=0.2, gamma=0.99, time_penalty=0.01):
    d_me = c_me_before - c_me_after       # >=0 nếu bạn đánh
    d_opp = c_opp_before - c_opp_after    # >=0 nếu đối thủ đánh
    shaping = gamma * phi(c_opp_after, c_me_after, k) - phi(c_opp_before, c_me_before, k)
    r_step = w1 * d_me - w2 * d_opp + shaping - time_penalty
    # clip để ổn định
    return max(-1.0, min(1.0, r_step))

def final_reward(c_opp_0, c_opp_T, beta=5.0):
    # normalized final reward in [0, beta]
    return beta * (1.0 - c_opp_T / max(1, c_opp_0))