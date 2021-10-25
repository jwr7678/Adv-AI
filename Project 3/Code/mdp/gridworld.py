#!/usr/bin/env python3

from enum import Enum
from .mdp import FiniteStateMDP, MDPState
import itertools
import numpy as np


class Actions(Enum):
    UP=1
    DOWN=2
    LEFT=3
    RIGHT=4


_TXT = {
    Actions.UP: "^^",
    Actions.DOWN: "vv",
    Actions.LEFT: "<<",
    Actions.RIGHT: ">>",
}


_UP = np.array([0, 1])
_DOWN = np.array([0, -1])
_LEFT = np.array([-1, 0])
_RIGHT = np.array([1, 0])


class GridState(MDPState):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self._width = width
        self._height = height

    def clone(self):
        return GridState(self.x, self.y, self._width, self._height)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def coords(self):
        return (self.x, self.y)

    @property
    def i(self):
        return self._height * self.x + self.y

    def __repr__(self):
        return '({x}, {y})'.format(**self.__dict__)
    
    def __eq__(self, other):
        if isinstance(other, GridState):
            if self.x == other.x and self.y == other.y and self._width == other._width and self._height == other._height:
                return True
        elif isinstance(other, tuple):
            if self.x == other[0] and self.y == other[1]:
                return True
        return False

    def __hash__(self):
        return hash((self.x, self.y))

#Changed max[,] to have zero has the smallest option
def _clip(p, max_x, max_y):
    p = np.array([max(min(p[0], max_x-1), 0),
                  max(min(p[1], max_y-1), 0)])
    return p


_OBS_KEYS = ['pit', 'goal']
_OBS_REWARDS = {
    'pit': -1.0,
    'goal': 1.0
}

_OBJ_KEYS = []
class DiscreteGridWorldMDP(FiniteStateMDP):
    def __init__(self, w, h, move_cost=-0.1, gamma = 0.9, noise = .2):
        self._w = w
        self._h = h
        self.move_cost = move_cost
        self._obs = {k:{} for k in _OBS_KEYS}
        self.gamma = gamma
        self.walls = [] #List of tuples
        self.probs = [1 - noise, noise/2, noise/2]
        # self.transitions = {}
    

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    @property
    def num_states(self):
        return self.width * self.height * 4

    # @property
    # def states(self):
    #     # return itertools.product(
    #     #     range(self.width), range(self.height), (True, False), (True, False))
    #     states = list(itertools.product(range(self.width), range(self.height)))
    #     states = [state for state in states if state not in self.walls]
    #     return states

    @property
    def states(self):
        states = [GridState(*x, self.width, self.height) for x in
            itertools.product(range(self.width), range(self.height))]
        states = [state for state in states if state.coords not in self.walls]
        return states

    @property
    def actions(self):
        return Actions

    @property
    def initial_state(self):
        # return WumpusState(0, 0, False, False, self.width, self.height)
        return GridState(0, 0, self.width, self.height)


    def actions_at(self, state):
        # if isinstance(state, tuple):
        #     state = GridState(state[0], state[1], self.width, self.height)
        if self.is_terminal(state):
            return [None]
        a = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
        return a

    def p(self, state, action):
        # if isinstance(state, tuple):
        #     state = GridState(state[0], state[1], self.width, self.height)
        if action is None:
            return zip([state], [0.0]) #[state, 0.0]
        if action in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]:
            return self.move(state, action)
        else:
            raise Exception("Invalid action specified: {}".format(action))

    # def coordsToState(self, pos):
    #     if isinstance(pos, GridState):
    #         return pos
    #     elif isinstance(pos, tuple):
    #         return GridState(pos[0], pos[1], self.width, self.height)
    #     else:
    #         raise Exception('Invalid input, neither position coords or GridState')

    def r(self, s2):
        ## if it's the pit, return the default pit bad reward
        # s1 = self.coordsToState(s1)
        # s2 = self.coordsToState(s2)
        if self.obs_at('pit', s2.pos):
            return self._obs['pit'][tuple(s2.pos)]

        ## if it's the goal, return the goal reward + gold reward if applicable
        if self.obs_at('goal', s2.pos):
            return self._obs['goal'][tuple(s2.pos)]

        ## otherwise return the move cost
        return self.move_cost

    # def getReward(self, s2):
    #     ## if it's the pit, return the default pit bad reward
    #     # s2 = self.coordsToState(s2)

    #     if self.obs_at('pit', s2.pos):
    #         return self._obs['pit'][tuple(s2.pos)]

    #     ## if it's the goal, return the goal reward + gold reward if applicable
    #     if self.obs_at('goal', s2.pos):
    #         return self._obs['goal'][tuple(s2.pos)]

    #     ## otherwise return the move cost
    #     return self.move_cost

    def is_terminal(self, state):
        ## if we're at the wumpus and have no immunity, we die
        # state = self.coordsToState(state)
        if self.obs_at('pit', state.pos):
            return True

        ## if we're at the goal, we win
        if self.obs_at('goal', state.pos):
            return True

        return False

    def obs_at(self, kind, pos):
        return tuple(pos) in self._obs[kind].keys()

    def move(self, state, action):
        # probs = [0.8, 0.1, 0.1]
        # probs = [0.9, 0.05, 0.05]
        # probs = [1.0, 0, 0]

        if action == Actions.UP:
            alst = [_UP, _LEFT, _RIGHT]
        elif action == Actions.DOWN:
            alst = [_DOWN, _RIGHT, _LEFT]
        elif action == Actions.LEFT:
            alst = [_LEFT, _UP, _DOWN]
        elif action == Actions.RIGHT:
            alst = [_RIGHT, _DOWN, _UP]

        x = []
        for a in alst:
            test_pos = state.pos + a
            if self.checkForWall(test_pos):
                test_pos = state.pos
            new_state = state.clone()
            new_pos = _clip(test_pos, self.width, self.height)
            # print('np', new_pos)
            new_state.x = new_pos[0]
            new_state.y = new_pos[1]
            x += [new_state]

        return zip(x, self.probs)

    def add_obstacle(self, kind, pos, reward=None):
        ## default rewards
        if not reward:
            reward = _OBS_REWARDS[kind]
        self._obs[kind][tuple(pos)] = reward

    def setWalls(self, pos):
        if isinstance(pos[0], tuple):
            self.walls += pos
        else:
            self.walls.append(pos)

    def checkForWall(self, pos):
        if not isinstance(pos, tuple):
            pos = tuple(pos)
        if pos in self.walls:
            return True
        return False

    def display(self, pos = None):
        obs_lab = lambda p, lab, kind: lab if self.obs_at(kind, p) else ' '
        print('      ', end='')
        for i in range(self.width):
            print(' {:5d} '.format(i),end='')
        print()
        for j in reversed(range(self.height)):
            print('{:5d} '.format(j), end='')
            for i in range(self.width):
                p = tuple([i, j])
                l_s = 'S' if p == (0, 0) else ' '
                l_p = obs_lab(p, 'P', 'pit')
                l_gl = obs_lab(p, 'G', 'goal')
                l_w = 'W' if p in self.walls else ' '
                l_gd = ' '
                l_x = 'X' if p == pos else ' '
                print('|' + l_s+l_w+l_p+l_gl+l_gd+l_x,end='')
            print('|')

