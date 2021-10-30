from environments import \
    LavaWorld5x7Determ, LavaWorld5x7StochMovement, LavaWorld5x7StochRewards,\
    CliffWorld4x12Determ
    
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import q_learning, double_q_learning

import numpy as np


def run_setup(config, q_learning_variant):
    policy = config['policy']
    epsilon_0 = config['epsilon_0']
    epsilon_decay = config['epsilon_decay']
    gamma = config['gamma']
    alpha_0 = config['alpha_0']
    alpha_decay = config['alpha_decay']
    num_iter = config['num_iter']

    env = config['env']
    if env == "LavaWorld5x7Determ":
        env = LavaWorld5x7Determ()
    elif env == "LavaWorld5x7StochMovement":
        env = LavaWorld5x7StochMovement()
    elif env == "LavaWorld5x7StochRewards":
        env = LavaWorld5x7StochRewards()
    elif env == "CliffWorld4x12Determ":
        env = CliffWorld4x12Determ()

    else:
        raise NotImplementedError

    if policy == "EpsilonGreedy":
        if q_learning_variant == "vanilla":
            Q = np.zeros((env.nS, env.nA))
            policy = EpsilonGreedyPolicy(Q, epsilon_0, epsilon_decay)
            Q_table, metrics, policy, Q_tables = q_learning(
                env, policy, Q, num_iter, discount_factor=gamma, alpha_0=alpha_0, alpha_decay=alpha_decay)
            return Q_table, np.array(metrics), policy, Q_tables, env
        elif q_learning_variant == "double":
            Q1 = np.zeros((env.nS, env.nA))
            Q2 = np.zeros((env.nS, env.nA))
            policy = EpsilonGreedyPolicy_Double_Q(
                Q1, Q2, epsilon_0, epsilon_decay)
            Q_table1, Q_table2, metrics, policy, Q_tables = double_q_learning(
                env, policy, Q1, Q2, num_iter,  discount_factor=gamma, alpha_0=alpha_0, alpha_decay=alpha_decay)
            return Q_table1, Q_table2, np.array(metrics), policy, Q_tables, env
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
