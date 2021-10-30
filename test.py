from environments import \
    LavaWorld5x7Determ, LavaWorld5x7StochMovement, LavaWorld5x7StochRewards,\
    CliffWorld4x12Determ
    
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import q_learning, double_q_learning

import numpy as np


env = CliffWorld4x12Determ()
Q = np.zeros((env.nS, env.nA))
policy = EpsilonGreedyPolicy(Q, 0.1, 0)
Q_table, metrics, policy, Q_tables = q_learning(
    env, policy, Q, 200, discount_factor=1., alpha_0=0.1, alpha_decay=0)
k=0