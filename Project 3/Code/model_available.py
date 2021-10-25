# Written by Jacob Regan
# For CS 5313/7313: Advanced Artificial Intelligence
# Solving Markov Decision Processes

from mdp.gridworld import * #DiscreteGridWorldMDP
from mdp.wumpus import *
import numpy as np
import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import timeit
import math


def valueIteration(mdp, error_est):
    u1 = {state: 0 for state in mdp.states}
    i = 0
    while True:
        u = u1.copy()
        i += 1
        dUtil = 0
        for state in mdp.states:
            u1[state] = mdp.r(state) + mdp.gamma * max([
                    sum([u[x]*p for (x, p) in mdp.p(state, a)])
                    for a in mdp.actions_at(state)
                ])
            dUtil = max(dUtil, abs(u1[state] - u[state]))
        if dUtil < error_est * (1-mdp.gamma)/mdp.gamma:
            return u, i

def bestPolicy(mdp, U):
    pi = {}
    for state in mdp.states:
        pi[state] = argMax(mdp, U, state)
    return pi

def expectedUtility(mdp, U, state, action):
    return sum(U[s]*p for (s, p) in mdp.p(state, action))

def argMax(mdp, U, state):
    return max(mdp.actions_at(state), key=lambda a: expectedUtility(mdp, U, state, a))

def policyEvaluation(pi, U, mdp, k=20):
    for i in range(k):
        for state in mdp.states:
            U[state] = mdp.r(state) + mdp.gamma * sum([U[s] * p for s, p in mdp.p(state, pi[state])])
    return U

def policyIteration(mdp):
    U = {state: 0 for state in mdp.states}
    pi = {state: random.choice(mdp.actions_at(state)) for state in mdp.states}
    i = 0
    while True:
        U = policyEvaluation(pi, U, mdp)
        i += 1
        unchanged = True
        for state in mdp.states:
            action = argMax(mdp, U, state)
            if action != pi[state]:
                pi[state] = action
                unchanged = False
        if unchanged:
            return pi, i


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

def getVisitsDistribution(mdp, policies):
    visitCounts = {state.coords: 0 for state in mdp.states}
    x = mdp.initial_state
    visitCounts[x.coords] += 1
    while not mdp.is_terminal(x):
        # print('state', x)
        # print(policies[x])
        a = policies[x]
        # print(a)
        xPoss, _ = zip(*mdp.p(x, a))
        x = xPoss[0]
        #NOTE: Use below segment to add noise to exploration
        # x, _ = mdp.act(x, a)
        visitCounts[x.coords] += 1
    # print(visitCounts)
    return visitCounts

def printUtility(mdp, utilities, iCount, policies=None):
    policies = bestPolicy(mdp, utilities)
    # for key,value in policies.items():
    #     print(key, value)

    data = [[np.nan] * mdp.width for _ in range(mdp.height)]
    for key, value in utilities.items():
        key = key.coords
        data[key[1]][key[0]] = round(value,4)
    data.reverse()
    dataArray = np.array(data)
    

    fig, ax = plt.subplots()
    im = ax.imshow(dataArray, cmap="cool")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Utility', rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(range(mdp.width), 1)
    ax.set_yticks(range(mdp.height))
    # ... and label them with the respective list entries
    ax.set_xticklabels(range(mdp.width))
    ax.set_yticklabels(range(mdp.height-1, -1, -1))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")


    tPolicies = getTransform(mdp, policies)
    # Loop over data dimensions and create text annotations.
    for i in range(len(dataArray)):
        for j in range(len(dataArray[0])):
            text = ax.text(j, i, dataArray[i, j],
                    ha="center", va="center", color="w")
            # print('T', i, j, tPolicies[i, j])
            if tPolicies[i, j] != None:
                # print('T', i, j, tPolicies[i, j])
                xOffset, yOffset = tPolicies[i, j]
                # print(xOffset, yOffset)
                plt.arrow(j - xOffset, i+0.2 + yOffset, xOffset, -yOffset, head_width = 0.1, head_length = 0.1)

    ax.set_title("Utility of Moves")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    fig.text(0.05,0.2, 'Gamma: ' + str(mdp.gamma))
    # fig.text(0.55,0.2, 'Maximum Error: ' + str(mdp.gamma))
    fig.text(0.05, 0.15, 'Number of iterations: ' + str(iCount))
    # plt.show()

def printPolicies_Grid(mdp, policies, utilities=None, iCount=None):

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
        cbar.ax.set_ylabel('Utility', rotation=-90, va="bottom")

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
                    ha="center", va="center", color="w", fontsize = 'medium')
            roomInfo = l_s+l_p+l_gl
            text = ax.text(j+ 0.5, i+ 0.9, roomInfo, #abs(i - len(dataArray) + 1)
                        ha="center", va="top", color="b", fontweight = 'bold', fontsize = 'medium')
            if tPolicies[i, j] != None:
                xOffset, yOffset = tPolicies[i, j]
                plt.arrow(j+0.5 - xOffset, i+0.3 - yOffset, xOffset, yOffset, head_width = 0.1, head_length = 0.1, width = 0.02)

    ax.set_title("GridWorld Policies")
    ax.grid(which='major', color='k', linewidth=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    fig.text(0.05,0.2, 'Gamma: ' + str(mdp.gamma))
    fig.text(0.05, 0.15, 'Number of iterations: ' + str(iCount))
    # plt.show()

def printPolicies_Wumpus(mdp, policies, utilities = None, iCount=None):

    tPolicies = [[[]] * mdp.width for _ in range(mdp.height)]
    appendUtility = lambda state: ''
    if utilities != None:
        appendUtility = lambda state: '\n      Utility: ' + str(round(utilities[state],3)) + '    '
    for state, value in policies.items():
        key = state.coords
        temp = tPolicies[key[1]][key[0]].copy()
        if value == None:
            temp.append(['Gold: T' if state.has_gold else 'Gold: F',
                'Immunity: T' if state.has_immunity else 'Immunity: F', appendUtility(state) + 'Action: None'
                ])
            tPolicies[key[1]][key[0]] = temp
            continue
        value = value.value
        if value == Actions.UP.value:
            temp.append(['Gold: T' if state.has_gold else 'Gold: F',
                'Immunity: T' if state.has_immunity else 'Immunity: F', appendUtility(state) + 'Action: ^^'
                ])
        elif value == Actions.DOWN.value:
            temp.append(['Gold: T' if state.has_gold else 'Gold: F',
                'Immunity: T' if state.has_immunity else 'Immunity: F', appendUtility(state) + 'Action: vv'
                ])
        elif value == Actions.LEFT.value:
            temp.append(['Gold: T' if state.has_gold else 'Gold: F',
                'Immunity: T' if state.has_immunity else 'Immunity: F', appendUtility(state) + 'Action: <<'
                ])
        elif value == Actions.RIGHT.value:
            temp.append(['Gold: T' if state.has_gold else 'Gold: F',
                'Immunity: T' if state.has_immunity else 'Immunity: F', appendUtility(state) + 'Action: >>' 
                ])
        elif value == Actions.PICK_UP.value:
            pickUpItem = '[G]' if  mdp.obj_at('gold', state.pos) else '[I]'
            temp.append(['Gold: T' if state.has_gold else 'Gold: F',
                'Immunity: T' if state.has_immunity else 'Immunity: F', appendUtility(state) + 'Action: ' + pickUpItem
                ])
        else: 
            continue
        tPolicies[key[1]][key[0]] = temp
        
    policyArray = np.array(tPolicies)
    data = [[np.nan] * mdp.width for _ in range(mdp.height)]
    visitDistr = getVisitsDistribution(mdp, policies)
    for key, value in visitDistr.items():
        data[key[1]][key[0]] = value
    dataArray = np.array(data)
    
    fig, ax = plt.subplots()
    im = ax.imshow(dataArray, origin = 'lower', extent = (0, dataArray.shape[1], 0, dataArray.shape[0]), cmap="GnBu")
    if utilities != None:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Actions per Space', rotation=-90, va="bottom")
    # cbar = ax.figure.colorbar(im, shrink = 0.5, ax=ax, ticks=np.arange(np.min(dataArray), np.max(dataArray)+1), orientation='horizontal')
    # cbar.ax.set_xlabel('Actions per Space')
    # We want to show all ticks...
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

    for i in range(len(dataArray)):
        for j in range(len(dataArray[i])):
            obs_lab = lambda p, lab, kind: lab + ': ' + str(mdp._obs[kind][p]) + '\n' if mdp.obs_at(kind, p) else ''
            obj_lab = lambda p, lab, kind: lab if mdp.obj_at(kind, p) else ''
            p = (j, i)
            l_s = 'Start\n' if p == (0, 0) else ''
            l_w = obs_lab(p, 'Wumpus', 'wumpus')
            l_p = obs_lab(p, 'Pit', 'pit')
            l_gl = obs_lab(p, 'Goal', 'goal')
            l_gd = obj_lab(p, 'Gold: +' + str(mdp.gold_reward) + ' at Goal\n', 'gold')
            l_i = obj_lab(p, 'Immunity\n', 'immune')
            roomInfo = l_s+l_w+l_p+l_gl+l_gd+l_i
            if l_w == '' and l_p == '' and l_gl == '':
                roomInfo += 'Step Cost: ' + str(mdp.getMoveCost(np.array([j, i])))
            text = ax.text(j+ 0.5, i+ 0.9, roomInfo, #abs(i - len(dataArray) + 1)
                        ha="center", va="top", color="k")
            for s in range(len(policyArray[i, j])):
                text = ax.text(j + .05, i+ 0.7 - 0.15 * s, "  ".join(policyArray[i, j, s]),
                        ha="left", va="top", color="m", wrap=True, fontsize=10, fontweight='semibold')

    ax.set_title("Wumpus World Policy")
    ax.grid(which='major', color='k', linewidth=2)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(bottom=0.15)

    fig.text(0.3,0.08, 'Gamma: ' + str(mdp.gamma))
    fig.text(0.3, 0.04, 'Number of iterations: ' + str(iCount))

    # fig.text(0.05,0.2, 'Gamma: ' + str(mdp.gamma))
    # fig.text(0.55,0.2, 'Maximum Error: ' + str(mdp.gamma))
    # fig.text(0.05, 0.15, 'Number of iterations: ' + str(iCount))
    # plt.show()

def GridTest():
    mdp = DiscreteGridWorldMDP(4, 3, move_cost=-0.04, gamma=0.9)
    mdp.setWalls((1,1))
    mdp.add_obstacle('goal', [3, 2])
    # mdp.add_obstacle('goal', [2, 0])
    mdp.add_obstacle('pit', [3, 1])
    return mdp

def WumpusTest(width = 4, height = 4):
    mdp = WumpusMDP(width, height, move_cost = -.1, gold_reward = 10, noise = 0.2)
    mdp.add_obstacle('goal', [3, 3])
    mdp.add_object('gold', [0, 2])
    # mdp.add_object('gold', [1, 2])
    mdp.add_object('immune', [2, 1])
    mdp.add_obstacle('wumpus', [1, 2], reward=-10)
    # mdp.add_obstacle('wumpus', [1, 2])
    mdp.add_obstacle('pit', [0, 1], reward = -10)
    mdp.add_obstacle('pit', [2, 3], reward = -2)
    mdp.add_obstacle('pit', [3, 1], reward = -0.5)
    # mdp.add_obstacle('pit', [2, 1], reward = -1)
    
    return mdp


# mdp = WumpusTest()
# mdp = WumpusTest()
# mdp.display()

# Value Iteration Test
# utilities, iCount = valueIteration(mdp, 0.001)
# # policies = policyIteration(mdp)
# def testUtilities_W():
#     utilities, iCount = valueIteration(mdp, 0.001)

# def testPolicies_W():
#     policies, iCount = policyIteration(mdp)
# # timeit.timeit
# policies = bestPolicy(mdp, utilities)
# # for key,value in policies.items():
# #     print(key, value)

# printPolicies_Wumpus(mdp, policies, utilities, iCount)
# # policies, iCount = policyIteration(mdp)
# # printPolicies_Wumpus(mdp, policies, iCount=iCount)
# plt.show()

# for key,value in utilities.items():
#     print(key, value)

# for state in mdp.all_poss_states:
#     print(state)

# policies = bestPolicy(mdp, utilities)

# printPolicies_Wumpus(mdp, policies)
# def testValueIteration(mdp):
#     utilities, iCount = valueIteration(mdp, 0.001)

# def testPolicyIteration(mdp):
#     policy = policyIteration(mdp)

def getRandomGridMDP(width, height):
    mdp = DiscreteGridWorldMDP(width, height, move_cost=-0.1, gamma=0.9, noise = 0.2)
    # mdp.setWalls((1,1))
    goal_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    while goal_pos == [0, 0]:
        goal_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    pit_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    while pit_pos == [0, 0] or pit_pos == goal_pos:
        pit_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    mdp.add_obstacle('goal', goal_pos)
    # mdp.add_obstacle('goal', [2, 0])
    mdp.add_obstacle('pit', pit_pos)
    return mdp

def getRandomWumpusMDP(width, height):
    mdp = DiscreteGridWorldMDP(width, height, move_cost=-0.1, gamma=0.9, noise = 0.2)
    mdp = WumpusMDP(width, height, move_cost = -0.1, gold_reward = 10, gamma = 0.9)
    # mdp.setWalls((1,1))
    goal_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    while goal_pos == [0, 0]:
        goal_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    pit_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    while pit_pos == [0, 0] or pit_pos == goal_pos:
        pit_pos = [random.randint(0, width-1), random.randint(0, height-1)]
    mdp.add_obstacle('goal', goal_pos)
    # mdp.add_obstacle('goal', [2, 0])
    mdp.add_obstacle('pit', pit_pos)
    return mdp


def compareIterations():
    sample_sizes = range(4, 8, 1)
    sample_sizes = [x* x for x in sample_sizes]
    TRIALS = 10
    results = []
    viResults = {}
    piResults = {}
    def testValueIteration():
        # mdp = WumpusTest(width, height)
        # mdp = WumpusTest()
        # mdp = getRandomGridMDP(width, height)
        utilities, iCount = valueIteration(mdp, 0.001)
        viResults[c] += iCount
        # viResults[int((c-4)/2)] += iCount

    def testPolicyIteration():
        # mdp = WumpusTest(width, height)
        # mdp = getRandomGridMDP(width, height)
        policy, iCount = policyIteration(mdp)
        piResults[c] += iCount
        # piResults[int((c-4)/2)] += iCount

    # for c in range(4, MAX_BOXES, 4):
    for c in sample_sizes:
        print(c)
        piResults[c] = 0
        viResults[c] = 0
        width = int(math.sqrt(c))
        height = width
        mdp = GridTest()
        # mdp = getRandomGridMDP(width, height)
        vi = timeit.timeit(testValueIteration, number=TRIALS)
        pi = timeit.timeit(testPolicyIteration, number = TRIALS)
        results.append([c, vi, pi])
    # fileList = []
    iterResults = []
    for i in range(len(results)):
        totalVi = viResults[results[i][0]] / TRIALS
        totalPi = piResults[results[i][0]] / TRIALS
        iterResults.append([results[i][0], totalVi, totalPi])
        results[i][1] /= TRIALS
        results[i][2] /= TRIALS
    x, y, z = zip(*results)
    plt.plot(x, y, label='Value Iteration')
    plt.plot(x, z, label='Policy Iteration')
    plt.ylabel('Avg Time (s)')
    plt.xlabel('GridSize')
    plt.grid()
    plt.legend()
    plt.show()
    import csv

    with open('results_wumpus.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        for row in results:
            wr.writerow(row)

    with open('iterationResults_wumpus.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        for row in iterResults:
            wr.writerow(row)

# compareIterations()
results, iterResults = compareIterations()

# print(timeit.timeit(testQLearning, number = 1))
