# Written by Jacob Regan
# For CS 5313/7313: Advanced Artificial Intelligence
# Solving Markov Decision Processes
from collections import defaultdict
from mdp.gridworld import *
import random
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, mdp, gamma = 0.9, alpha=None, Ne = 5, Rplus = 0.1):
        self.mdp = mdp
        self.gamma = gamma
        self.mdp.gamma = gamma
        if alpha:
            if isinstance(alpha, float):
                self.alpha = lambda n: alpha
            else:
                self.alpha = alpha
        else:
            self.alpha = lambda n: 1 /(1 + n) #See pg. 837 in textbook
        self.Ne = Ne
        self.Rplus = Rplus
        self.Qtable = defaultdict(float)
        self.Nsa = defaultdict(int)
        self.s, self.a, self.r = None, None, None

    def isTerminal(self, state):
        if self.mdp.is_terminal(state):
            return True
        else:
            return False

    def f(self, u, n):
        return self.Rplus if n < self.Ne else u

    @property
    def previousArgs(self):
        return self.s, self.a, self.r
    
    def resetArgs(self):
        self.s = self.a = self.r = None
    
#Percept is assumed to be tuple (state:State, reward)
def qLearningStep(agent: Agent, percept):
    s, a, r = agent.previousArgs
    Qtable = agent.Qtable
    # print(Qtable)
    Nsa = agent.Nsa
    s1, r1 = percept
    if agent.isTerminal(s1):
        Qtable[s1, None] = r1
    if s is not None:
    #     if s.coords == (2, 1):
    #         print([Qtable[s1, a1] for a1 in agent.mdp.actions_at(s1)], ':', s, a)
        Nsa[s, a] += 1
        Qtable[s,a] = Qtable[s, a] + agent.alpha(Nsa[s, a]) * (r + agent.gamma 
            * max(Qtable[s1, a1] for a1 in agent.mdp.actions_at(s1)) - Qtable[s, a]
            )

    if agent.isTerminal(s1):
        agent.resetArgs()
        # print('IS TERMINAL', agent.a)
    else:
        agent.s = s1
        agent.a = max(agent.mdp.actions_at(s1), key=lambda a1: agent.f(Qtable[s1, a1], Nsa[s1,a1]))
        agent.r = r1
    # print(s1, agent.a)
    return agent.a

#Percept include is state, reward, action
def sarsaLearningStep(agent: Agent, percept):
    s, a, r = agent.previousArgs
    Qtable = agent.Qtable
    Nsa = agent.Nsa
    s1, r1 = percept
    a1 = max(agent.mdp.actions_at(s1), key=lambda a1: agent.f(Qtable[s1, a1], Nsa[s1,a1]))
    if agent.isTerminal(s1):
        Qtable[s1, None] = r1
    if s is not None:
    #     if s.coords == (2, 1):
    #         print([Qtable[s1, a1] for a1 in agent.mdp.actions_at(s1)], ':', s, a)
        Nsa[s, a] += 1
        Qtable[s,a] = Qtable[s, a] + agent.alpha(Nsa[s, a]) * (r + agent.gamma 
            * Qtable[s1, a1] - Qtable[s, a]
            )

    if agent.isTerminal(s1):
        agent.resetArgs()
        # print('IS TERMINAL', agent.a)
    else:
        agent.s = s1
        # agent.a = max(agent.mdp.actions_at(s1), key=lambda a1: agent.f(Qtable[s1, a1], Nsa[s1,a1]))
        agent.a = a1
        agent.r = r1
    # print(s1, agent.a)
    return agent.a

#Perform n episodes
def runQTrials(agent: Agent, n):
    episodeRewardList = []
    for i in range(n):
        mdp = agent.mdp
        x = mdp.initial_state
        total_reward = 0
        stepCount = 0
        while True:
            # r = mdp.getReward(x)
            r = mdp.r(x)
            total_reward += r
            stepCount += 1
            percept = (x, r)
            # s, a0, _ = agent.previousArgs
            # if a0 is not None:
            #     if agent.Qtable[s, a0] > 0:
            #         print('Greater than', agent.Qtable[s, a0])
            a1 = qLearningStep(agent, percept)
            if a1 is None:
                break
            x, _ = mdp.act(x, a1)
        # print('Step:', i, 'Reward:', total_reward)
        episodeRewardList.append([i+1, total_reward, stepCount])
        # episodeRewardList.append([i+1, stepCount])
    return episodeRewardList


def runSARSATrials(agent: Agent, n):
    episodeRewardList = []
    for i in range(n):
        mdp = agent.mdp
        x = mdp.initial_state
        total_reward = 0
        stepCount = 0
        while True:
            # r = mdp.getReward(x)
            r = mdp.r(x)
            total_reward += r
            stepCount += 1
            percept = (x, r)
            a1 = sarsaLearningStep(agent, percept)
            if a1 is None:
                break
            x, _ = mdp.act(x, a1)
        episodeRewardList.append([i+1, total_reward, stepCount])
        # episodeRewardList.append([i+1, total_reward])
    return episodeRewardList

def getTransform(mdp, policies):
    tPolicies = [[None] * mdp.width for _ in range(mdp.height)]
    for key, value in policies.items():
        key = key.coords
        if value == None:
            continue
        value = value.value
        if value == Actions.UP.value:
            # print('UP')
            tPolicies[key[1]][key[0]] = (0, 0.1)
        elif value == Actions.DOWN.value:
            tPolicies[key[1]][key[0]] = (0, -.1)
        elif value == Actions.LEFT.value:
            tPolicies[key[1]][key[0]] = (-.1, 0)
        elif value == Actions.RIGHT.value:
            tPolicies[key[1]][key[0]] = (.1, 0)
    # tPolicies.reverse()
    return np.array(tPolicies)

def printPolicies_Grid(mdp, policies, utilities=None, isQLearning=True):
    # import matplotlib
    # import matplotlib.pyplot as plt

    data = [[np.nan] * mdp.width for _ in range(mdp.height)]
    if utilities != None:
        for key, value in utilities.items():
            key = key.coords
            data[key[1]][key[0]] = round(value,4)
    dataArray = np.array(data)
    
    fig, ax = plt.subplots()
    im = ax.imshow(dataArray, origin = 'lower', extent = (0, dataArray.shape[1], 0, dataArray.shape[0]), cmap = 'cool')
    if utilities != None:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Q Value', rotation=-90, va="bottom")

    ax.set_xticks(range(mdp.width))
    ax.set_xticks(np.arange(0.5, mdp.width, 1), minor=True)

    ax.set_yticks(range(mdp.height))
    ax.set_yticks(np.arange(0.5, mdp.height, 1), minor=True)
    # ... and label them with the respective list entries
    ax.set_xticklabels([])
    ax.set_xticklabels(range(mdp.width), minor=True)

    ax.set_yticklabels([])
    ax.set_yticklabels(range(mdp.height), minor=True)

    plt.setp(ax.get_xticklabels(minor=True), rotation=15, ha="center",
            rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(minor=True), rotation=15, ha="center",
            rotation_mode="anchor")

    tPolicies = getTransform(mdp, policies)
    # Loop over data dims and create arrows for policies
    for i in range(len(dataArray)):
        for j in range(len(dataArray[0])):
            obs_lab = lambda p, lab, kind: lab if mdp.obs_at(kind, p) else ''
            p = (j, i)
            l_s = 'Start\n' if p == (0, 0) else ''
            l_gl = obs_lab(p, 'Goal\n', 'goal')
            l_p = obs_lab(p, 'Pit\n', 'pit')
            text = ax.text(j+.5, i+.6, dataArray[i, j],
                    ha="center", va="center", color="w", fontsize = 'medium', fontweight = 'bold')
            roomInfo = l_s+l_p+l_gl
            text = ax.text(j+ 0.5, i+ 0.9, roomInfo, #abs(i - len(dataArray) + 1)
                        ha="center", va="top", color="b", fontweight = 'bold', fontsize = 'medium')
            if tPolicies[i, j] != None:
                xOffset, yOffset = tPolicies[i, j]
                plt.arrow(j+0.5 - xOffset, i+0.3 - yOffset, xOffset, yOffset, head_width = 0.1, head_length = 0.1, width = 0.02)
    if isQLearning:
        ax.set_title("Q Learning Results")
    else:
        ax.set_title("SARSA Results")
    ax.grid(which='major', color='k', linewidth=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    fig.text(0.05,0.2, 'Gamma: ' + str(mdp.gamma))
    # fig.text(0.05, 0.15, 'Number of iterations: ' + str(iCount))
    

#Returns the policies and qValues list for printing
def getResults(agent):
    bestSteps = {}
    for key, value in dict(agent.Qtable).items():
        # print(key, value)
        if key[0] in bestSteps:
            # print(key[0], bestSteps[key[0]][1], value)
            if value > bestSteps[key[0]][1]:
                bestSteps[key[0]] = (key[1], value)
        else:
            bestSteps[key[0]] = (key[1], value)
            # print(key[0], bestSteps[key[0]][1], value)

    # print('\nBest Choices')
    policies = {}
    qValues = {}
    for key, value in bestSteps.items():
        policies[key] = value[0]
        qValues[key] = value[1]
        # print(key, value)
    return policies, qValues


def GridTest():
    mdp = DiscreteGridWorldMDP(4, 3, move_cost=-0.04, noise = 0.2)
    mdp.setWalls((1,1))
    mdp.add_obstacle('goal', [3, 2], reward=1)
    # mdp.add_obstacle('goal', [2, 0])
    mdp.add_obstacle('pit', [3, 1], reward=-1)
    return mdp

def GridTest2():
    mdp = DiscreteGridWorldMDP(4, 3, move_cost=-0.04, noise = 0)
    mdp.setWalls((1,1))
    mdp.add_obstacle('goal', [3, 2], reward=1)
    # mdp.add_obstacle('goal', [2, 0])
    mdp.add_obstacle('pit', [3, 1], reward=-1)
    return mdp

mdp = GridTest()
mdp.display()
# lambda n: 60. / (59 + n)
agent = Agent(mdp, alpha=0.5)
# agent = Agent(mdp, gamma = .9, alpha = lambda n: 60. / (59 + n))
# g1 = GridState(2, 2, 4, 3)
# print(qLearningAgent(agent, (g1,mdp.getReward(g1))))

# runQTrials(agent, 100)

# runSARSATrials(agent, 500)
qResultsList = []
def testQLearning():
    global agent, qResultsList, EPISODES, currentTrial
    print('Q Learning Trial:', currentTrial)
    currentTrial += 1
    # agent = Agent(mdp, gamma = .9, alpha = lambda n: 60. / (59 + n))#, alpha = 0.5  
    agent = Agent(mdp, gamma = .9, alpha = 0.5)
    #, alpha=0.5)
    resultsList = runQTrials(agent, EPISODES)
    if qResultsList:
        for i in range(len(resultsList)):
            qResultsList[i] = [resultsList[i][0], qResultsList[i][1] + resultsList[i][1], qResultsList[i][2] + resultsList[i][2]]
    else:
        qResultsList = resultsList.copy()

    # plt.show()
# sarsaResultsList = []
# def testSARSA():
#     global agent, sarsaResultsList, EPISODES, currentTrial
#     print('Q Learning Trial:', currentTrial)
#     currentTrial += 1
#     agent = Agent(mdp, gamma = .9, 0.5)#, alpha = 0.5  
#     resultsList = runQTrials(agent, EPISODES)
#     if sarsaResultsList:
#         for i in range(len(resultsList)):
#             sarsaResultsList[i] = [resultsList[i][0], sarsaResultsList[i][1] + resultsList[i][1], sarsaResultsList[i][2] + resultsList[i][2]]
#     else:
#         sarsaResultsList = resultsList.copy()

sarsaResultsList = []
def testSARSA():
    global agent, sarsaResultsList, EPISODES, currentTrial
    print('Q Learning Trial:', currentTrial)
    currentTrial += 1
    # agent = Agent(mdp, gamma = .9, alpha = lambda n: 60. / (59 + n))#, alpha = 0.5  
    agent = Agent(mdp, gamma = .9, alpha = 0.5)
    resultsList = runSARSATrials(agent, EPISODES)
    if sarsaResultsList:
        for i in range(len(resultsList)):
            sarsaResultsList[i] = [resultsList[i][0], sarsaResultsList[i][1] + resultsList[i][1], sarsaResultsList[i][2] + resultsList[i][2]]
    else:
        sarsaResultsList = resultsList.copy()

def graphResults(qResultsList, sarsaResultsList, trialCount):
    
    for i in range(len(qResultsList)):
        qResultsList[i][1] /= trialCount
        sarsaResultsList[i][1] /= trialCount
        qResultsList[i][2] /= trialCount
        sarsaResultsList[i][2] /= trialCount
    episodeListQ, rewardListQ, stepListQ = zip(*qResultsList)
    episodeListSARSA, rewardListSARSA, stepListSARSA = zip(*sarsaResultsList)

    plt.figure('4x3 Reward vs Episodes')
    qPlot, = plt.plot(episodeListQ, rewardListQ, label= 'Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, rewardListSARSA, label = 'SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    # plt.ylabel('Avg Steps Per Episode')
    plt.title('Q Learning - Avg Reward vs Episode')
    # plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot], ['Q Learning', 'SARSA'], loc='upper right')

    plt.figure('4x3 Steps vs Episodes')
    qPlot, = plt.plot(episodeListQ, stepListQ, label= 'Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, stepListSARSA, label = 'SARSA')
    plt.xlabel('Episode')
    # plt.ylabel('Avg Reward')
    plt.ylabel('Avg Steps Per Episode')
    # plt.title('Q Learning - Avg Reward vs Episode')
    plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot], ['Q Learning', 'SARSA'], loc='upper right')

def graphResults_Expanded(qResultsList, qResultsList2, sarsaResultsList, sarsaResultsList2, trialCount):
    
    for i in range(len(qResultsList)):
        qResultsList[i][1] /= trialCount
        sarsaResultsList[i][1] /= trialCount
        qResultsList[i][2] /= trialCount
        sarsaResultsList[i][2] /= trialCount
    episodeListQ, rewardListQ, stepListQ = zip(*qResultsList)
    episodeListSARSA, rewardListSARSA, stepListSARSA = zip(*sarsaResultsList)
    episodeListQ2, rewardListQ2, stepListQ2 = zip(*qResultsList2)
    episodeListSARSA2, rewardListSARSA2, stepListSARSA2 = zip(*sarsaResultsList2)

    plt.figure('4x3 Reward vs Episodes')
    qPlot, = plt.plot(episodeListQ, rewardListQ, label= 'Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, rewardListSARSA, label = 'SARSA')

    qPlot2, = plt.plot(episodeListQ2, rewardListQ2, label= 'Q Learning (No Noise)')
    sarsaPlot2, = plt.plot(episodeListSARSA2, rewardListSARSA2, label = 'SARSA (No Noise)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    # plt.ylabel('Avg Steps Per Episode')
    plt.title('Q Learning - Avg Reward vs Episode')
    # plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot, qPlot2, sarsaPlot2], ['Q Learning', 'SARSA','Q Learning (No Noise)', 'SARSA (No Noise)'], loc='upper right')

    plt.figure('4x3 Steps vs Episodes')
    qPlot, = plt.plot(episodeListQ, stepListQ, label= 'Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, stepListSARSA, label = 'SARSA')
    qPlot2, = plt.plot(episodeListQ2, stepListQ2, label= 'Q Learning 2 ')
    sarsaPlot2, = plt.plot(episodeListSARSA2, stepListSARSA2, label = 'SARSA 2')
    plt.xlabel('Episode')
    # plt.ylabel('Avg Reward')
    plt.ylabel('Avg Steps Per Episode')
    # plt.title('Q Learning - Avg Reward vs Episode')
    plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot, qPlot2, sarsaPlot2], ['Q Learning', 'SARSA','Q Learning (No Noise)', 'SARSA (No Noise)'], loc='upper right')
    # plt.legend([qPlot, sarsaPlot], ['Q Learning', 'SARSA'], loc='upper right')

import timeit
TRIALS = 50
EPISODES = 500
currentTrial = 1
print(timeit.timeit(testQLearning, number = TRIALS))
currentTrial = 1
print(timeit.timeit(testSARSA, number = TRIALS))
qResultsList_Noise = qResultsList.copy()
sarsaResultsList_Noise = sarsaResultsList.copy()
graphResults(qResultsList, sarsaResultsList, TRIALS)

# qResultsList = []
# sarsaResultsList = []
# mdp = GridTest2()
# currentTrial = 1
# print(timeit.timeit(testQLearning, number = TRIALS))
# currentTrial = 1
# print(timeit.timeit(testSARSA, number = TRIALS))

# graphResults_Expanded(qResultsList_Noise, qResultsList, sarsaResultsList_Noise, sarsaResultsList, TRIALS)

# graphResults(qResultsList, sarsaResultsList, TRIALS)
# testQLearning()
# policies, qValues = getResults(agent)
# printPolicies_Grid(mdp, policies, qValues)
# # print(masterResultsList)

# testSARSA()
# policies, qValues = getResults(agent)
# printPolicies_Grid(mdp, policies, qValues, isQLearning=False)

plt.show()