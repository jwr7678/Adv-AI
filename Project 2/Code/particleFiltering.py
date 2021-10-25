from CS5313_Localization_Env import localization_env as le
import numpy as np
import math
import random
import itertools
import bisect

class HMM:
    def __init__(self, env, priorDist=None):
        self._location_transitions = env.location_transitions
        self._heading_transitions = env.heading_transitions
        self._observations = env.observation_tables
        self.priorDist = priorDist or [0.5, 0.5]
        self.env = env

    def sensorDist(self, state, evidence):
            return self._observations[state[0]][state[1]][evidence]

    


env = le.Environment(
    action_bias=0.1,
    observation_noise=0.2,
    action_noise=0.1,
    dimensions=(5,5),
    seed = 10,
    # seed=11,
)
import time
location, heading = env.dummy_location_and_heading_probs()
# print('Location',location)
# print('Heading',heading)

def probability(p):
    return p > random.uniform(0.0, 1.0)

def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)
    return [sample() for _ in range(n)]

def particleFilter(e, N, env):
    print('PARTICLE')
    # print('LP', list(env.location_priors.items()))
    # print('Head', env.heading_priors)
    # states = itertools.product(list(env.location_priors.items()), list(env.heading_priors.items()))
    # states = list(states)
    # distLoc = []
    # distHead = []
    # for val in states:
    #     x, y = val[0][0]
    #     priorLoc = val[0][1]
    #     heading = val[1][0]
    #     priorHead = val[1][1]
    #     temp = list(env.location_transitions[x][y][heading].values())
    #     # print(temp)
    #     locProduct = np.multiply(priorLoc, temp)
    #     distLoc.append(locProduct)
    #     temp = list(env.headings_transitions[x][y][heading].values())
    #     headProduct = np.multiply(priorHead, temp)
    #     distHead.append(headProduct)
    # random.seed(110)
    # print(list(env.location_priors.items()))
    sLoc = random.choices(list(env.location_priors.items()), k=N)
    sHead = []
    for value in sLoc:
        x, y = value[0]
        # sHead = np.random.choice(env.headings_transitions[x][y][hea])
        # print(list(env.heading_priors.items()))
        priors, weights = zip(*list(env.heading_priors.items()))
        sHead.append(np.random.choice(priors, 1, p=weights))
        # print('p', priors, weights)
        # print(np.random.choice(priors, 1, p=weights))
    # print(sLoc)

    s = []
    for i, val in enumerate(sLoc):
        s.append((val[0],sHead[i][0]))

    w = [0 for _ in range(N)]
    w_total = 0
    for i in range(N):
        x, y = s[i][0]
        # print(env.observation_tables[x][y])
        w_i = env.observation_tables[x][y][e]
        w[i] = w_i
        w_total += w_i
    
    #Norm
    for i in range(N):
        w[i] = w[i] / w_total

    s = weighted_sample_with_replacement(N, s, w)
    print(w)

    return s





sList = particleFilter((1,1,0,1), 100, env)
counted = {}
for s in sList:
    if s not in counted:
        counted[s] = sList.count(s)


sorted_d = sorted(counted.items(), key=lambda x: -x[1])
print(sorted_d)
for value in sorted_d:
    print(*value)

# dist = [1/len(env.free_cells) for state in env.free_cells]

# print(env.heading_priors)
# dist = []
stateDist = {}
for locKey, locValue in env.location_priors.items():
    # print('LOC', locKey, locValue)
    for headKey, headValue in env.heading_priors.items():
        # dist.append(locValue*headValue)
        stateDist[(locKey[0],locKey[1], headKey)] = [locValue, headValue]

# print('Dist', len(dist))
# for key, value in stateDist.items():
#     locT = list(env.location_transitions[key[0]][key[1]][key[2]].values())
#     headT = list(env.headings_transitions[key[0]][key[1]][key[2]].values())
#     print(key, value)
#     print('LOC', locT)
#     print('HEAD', headT)
    # print('Multiply', np.dot(locT, np.array([headT])))
    # stateDist[key] = np.multiply(value, env.location_transitions[key[0]][key[1]][key[2]]
    
    # dist[i] = np.dot(dist[i],env.location_transitions[state[0]][state[1]])

# print('DIST', dist)

done = False
printouts = True
print("LENGTH", env.free_cells)
env.running=False
while env.running == True and env.steps < 20:
    # location, heading = env.dummy_location_and_heading_probs()
    observation = env.move()
    env.update(location, heading)
    # print('HEADING TRANSITIONS')
    # for i in env.headings_transitions:
    #     print(i)
    #     print('')
    print('OBSERVATION', env.observation_tables[1][1])

    if printouts:
        print(observation)
    input('Press Enter to continue')
# while True:
#     env.running = True


#np.savetxt('PFText.txt', array, fmt='%s')