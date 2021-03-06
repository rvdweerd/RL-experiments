from itertools import product

import gym
import numpy as np
from gym import spaces


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class BaseGrid(gym.Env):
    """"""

    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def __init__(
        self,
        nS,
        nA,
        rows,
        cols,
        start,
        goal,
        final_reward,
        lava_cells=[],
        cliff_cells=[],
        wall_cells=[],
        p_forward=1.,
        stoch_reward=False,
    ):
        self.nS = nS
        self.nA = nA
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.final_reward = final_reward
        self.lava_cells = lava_cells
        self.cliff_cells = cliff_cells
        self.wall_cells = wall_cells
        self.p_forward = p_forward
        self.stoch_reward = stoch_reward

        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        # transition_probabilities
        self.P = self._get_transitions()

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def get_state_color(self, state_number):
        if state_number in [self.observation(cell) for cell in self.wall_cells]:
            return "black"  # wall
        if state_number in [self.observation(cell) for cell in self.cliff_cells]:
            return "grey"  # cliff
        elif state_number in [self.observation(cell) for cell in self.lava_cells]:
            return "orange"  # lava
        elif state_number == self.observation(self.start):
            return "blue"  # start position
        elif state_number == self.observation(self.goal):
            return "green"  # goal position
        else:
            return "white"

    def _new_loc(self, r, c, action):
        if action == UP:  # 0
            r2 = min(r + 1, self.rows - 1)
            return [r2, c]
        elif action == RIGHT:  # 1
            c2 = min(c + 1, self.cols - 1)
            return [r, c2]
        elif action == DOWN:  # 2
            r2 = max(r - 1, 0)
            return [r2, c]
        elif action == LEFT:  # 3
            c2 = max(c - 1, 0)
            return [r, c2]
        else:
            raise Exception("Invalid action.")

    def _get_transitions(self):
        """transition matrix"""
        # initialise
        transitions = {
            (r, c): {a: [] for a in range(self.nA)}
            for r, c in product(range(self.rows), range(self.cols))
        }

        # loop over state/action pairs
        for r, c, a in product(range(self.rows), range(self.cols), range(self.nA)):

            possible_a = [(a + d) % 4 for d in [-1, 0, 1]]
            prob_side = (1 - self.p_forward) / 2
            probs = [prob_side, self.p_forward, prob_side]
            new_locs = [self._new_loc(r, c, pa) for pa in possible_a]

            if self.stoch_reward:
                rewards = [-12, 10]
            else:
                rewards = [-1]

            for new_loc, loc_prob in zip(new_locs, probs):
                for reward in rewards:
                    done = False
                    if new_loc in self.lava_cells:
                        reward = -100
                        done = True
                    if new_loc in self.cliff_cells:
                        reward = -100
                        new_loc = self.start
                    if new_loc in self.wall_cells:
                        new_loc = [r,c]
                    if new_loc == self.goal:
                        reward = self.final_reward
                        done = True
                    transitions[(r, c)][a].append(
                        (loc_prob / len(rewards), new_loc, reward, done)
                    )

        # 0 reward after reaching goal state
        for a in range(self.nA):
            transitions[tuple(self.goal)][a] = [
                (p, tuple(self.goal), 0, True)
                for (p, l, r, d) in transitions[tuple(self.goal)][a]
            ]

        return transitions

    def step(self, action):
        next_steps = self.P[tuple(self.current_state)][action]
        probs = [p for p, _, _, _ in next_steps]
        chosen = np.random.choice(range(len(next_steps)), p=probs)
        _, new_state, reward, is_terminal = next_steps[chosen]

        self.current_state = new_state
        return self.observation(new_state), reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start
        return self.observation(self.current_state)

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class LavaWorld5x7Determ(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=35,
            nA=4,
            rows=5,
            cols=7,
            start=[4, 0],
            goal=[4, 6],
            final_reward=8,
            lava_cells=[[4, 2], [4, 3], [4, 4], [0, 2], [0, 3], [0, 4]],
            p_forward=1.0,
            stoch_reward=False,
        )


class LavaWorld5x7StochMovement(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=35,
            nA=4,
            rows=5,
            cols=7,
            start=[4, 0],
            goal=[4, 6],
            final_reward=10,
            lava_cells=[[4, 2], [4, 3], [4, 4], [0, 2], [0, 3], [0, 4]],
            p_forward=0.8,
            stoch_reward=False,
        )


class LavaWorld5x7StochRewards(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=35,
            nA=4,
            rows=5,
            cols=7,
            start=[4, 0],
            goal=[4, 6],
            final_reward=10,
            lava_cells=[[4, 2], [4, 3], [4, 4], [0, 2], [0, 3], [0, 4]],
            p_forward=1,
            stoch_reward=True,
        )



class CliffWorld4x12Determ(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=4*12,
            nA=4,
            rows=4,
            cols=12,
            start=[0, 0],
            goal=[0, 11],
            final_reward=13,
            lava_cells=[],
            cliff_cells=[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10]],
            p_forward=1,
            stoch_reward=False,
        )

class MazeWorld5x7Determ(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=5*7,
            nA=4,
            rows=5,
            cols=7,
            start=[0, 0],
            goal=[4, 6],
            final_reward=22,
            lava_cells=[[1,2]],
            cliff_cells=[[1,5]],
            wall_cells=[[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[1,0],[1,1],[1,3],[1,4]],
            p_forward=1,
            stoch_reward=False,
        )

