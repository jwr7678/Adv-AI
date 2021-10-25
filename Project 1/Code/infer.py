#Jacob Regan

import json
import random
import math
import numpy as np
from genBN import gen_bn, draw_net
#Global Values
T, F = 1,0
def openFile(file_name='BurglaryExample.json'):
    with open(file_name) as f:
        data = json.load(f)
    return data


class BayesianNetwork:
    def __init__(self, data: dict):
        self.network = {}
        temp = {}
        for key in data.keys():
            #NOTE Use this one for dag
            # self.network[str(key)] = Node(str(key), data[key]['parents'], data[key]['prob'])

            temp[str(key)] = Node(str(key), data[key]['parents'], data[key]['prob'])
            
        self.network = {k: node for k, node in sorted(temp.items(), key=lambda x: (len(x[1].parents), int(x[0])), reverse=True)}
        self.setChildren()

    def getNode(self, var):
        return self.network[var]
    
    def getNodeList(self, var_list: list):
        rlist = []
        for var in var_list:
            rlist.append(self.getNode(var))
        return rlist

    def getVars(self):
        return list(self.network.values())
    
    def defineEvent(self, e: dict):
        events = {}
        for key, value in e.items():
            events[self.getNode(key)] = value
        return events
    
    def getNodes(self):
        return self.network.values()

    def setChildren(self):
        for key, node in self.network.items():
            children = [var.var for var in self.network.values() if key in var.parents]
            node.children = children

    #Returns x, event representing the query variable and the set of observations randomly generated
    def genRandVariables(self, count):
        # print("TEST")
        obs = {}
        nodeNames = [node.var for node in self.getNodes() if node.isConnected]
        events_x = np.random.choice(nodeNames, min(len(nodeNames), count + 1), replace = False)
        x = np.random.choice(events_x, 1)[0]
        events = np.delete(events_x, np.where(events_x == x))
        # print(events_x)
        # print('X', type(x))
        # print(events)
        for e in events:
            val = random.choice([True, False])
            obs[e] = val
        return x, obs
        

class Node:
    def __init__(self, x: str, parents: list, probabilities: list):
        #Prob list will take form []
        self.var = x
        self.parents = [str(par) for par in parents]
        # self.parents.reverse()
        # print(self.parents)
        self.probDist = self.createProbDistDict(probabilities)
        self.children = []

    def getValues(self):
        return [T, F]

    def createProbDistDict(self, prob_list: list):
        dist = {}
        for item in prob_list:
            dist[tuple(item[0]) or T] = item[1]
        return dist


    def getProb(self, value, event):
        if len(self.parents) == 0:
            prob = self.probDist[(T)]
            return prob if value == T else 1.00 - prob

        else:
            # print('Node', self.var, self.parents)
            # print('Event', event)
            try:
                parentValues = [event[parent] for parent in self.parents]
            except:
                print("ERROR - Probably change polytree snippet in BN")
            prob = self.probDist[tuple(parentValues)]
            return prob if value == T else 1.00 - prob

    def sample(self, event):
        return self.getProb(True, event) > random.uniform(0.0, 1.0)

    def addChild(self, node):
        self.children.append(node)

    @property
    def isConnected(self):
        if self.parents or self.children:
            return True
        return False

       

#Takes X: Query Variable, e: evidence values array, and bn:json formatted bayesian network as input
def enumAsk(x, e: dict, bn):
    qx = x if isinstance(x, Node) else bn.getNode(x)
    #event = bn.defineEvent(e)
    event = e
    q_vector = []
    for i in qx.getValues():
        event[qx.var] = i
        q_vector.append(enumAll(bn.getVars(), event))
        del event[qx.var]
    return normalize(q_vector)
        

def enumAll(varibles, e: dict):
    value = 1.0
    if len(varibles) == 0:
        return value
    y = varibles.pop() #Changed from 0 to last element
    if y.var in e:
        value = y.getProb(e[y.var], e) * enumAll(varibles, e)
        varibles.append(y)
        return value
    else:
        temp = 0.0
        for val in y.getValues():
            e[y.var] = val
            temp += y.getProb(val, e) * enumAll(varibles, e)
            del e[y.var]
        value = temp
        varibles.append(y)
        return value

def printEvents(myDict: dict):
    print("-----")
    for key, value in myDict.items():
        print(key, value)
    print("-----")

def normalize(mylist):
    total = sum(mylist)
    norm_list = []
    for ele in mylist:
        norm_list.append(ele/total)
    return norm_list

#Likelihood-Weighting
def weightedSample(bn:BayesianNetwork, evid:dict):
    w = 1
    x = dict(evid)
    bl = list(bn.getNodes())
    bl.reverse()
    #print(bl)
    for node in bl:
        #print(node.var)
        var = node.var
        if var in evid:
            #print(evid, x)
            w *= node.getProb(evid[var], x)
            # print(w)
        else:
            x[var] = node.sample(x)
    #print(x, w)
    return x, w


#x - query variable, event, bayesian network, n - number of samples to generate
def likelihoodWeighting(x, evid, bnet, n):
    #weighted_vector = {x: 0 for x in }
    weighted_vector = {True: 0, False: 0}
    for i in range(n):
        e, w = weightedSample(bnet, evid)
        weighted_vector[e[x]] += w
    #print(weighted_vector)
    return normalize(list(weighted_vector.values()))

def gibbsAsk(x, evid, bn, n):
    count_vector = {True: 0, False: 0}
    z = [var for var in bn.getNodes() if var.var not in evid.keys()]
    z.reverse()
    states = dict(evid)
    # print(evid)
    # print([node.var for node in z])
    for node in z:
        states[node.var] = random.choice(node.getValues())
    for i in range(n):
        for node in z:
            states[node.var] = markovBlanketSample(node, bn, states)
            count_vector[states[x]] += 1
    return normalize(list(count_vector.values()))

def markovBlanketSample(x, bn, event):
    result_vector = {True: 0, False: 0}
    eventX = dict(event)
    for value in x.getValues():
        eventX[x.var] = value
        # print(eventX)
        children = [bn.getNode(child).getProb(eventX[child], eventX) for child in x.children]
        # print(x.var)
        # print(children)
        result_vector[value] = x.getProb(value, event) * math.prod(children)
    q = list(result_vector.values())
    # print(result_vector)
    q = normalize(q)
    return q[0] > random.uniform(0.0, 1.0)


def runTimeTrials():
    # global bn, data
    from timer import Timer
    MIN_NODE_COUNT = 5
    MAX_NODE_COUNT = 50
    TRIAL_COUNT = 5
    resultsList = []
    enumTimer = Timer()
    lkhdTimer = Timer()
    gibbsTimer = Timer()
    for size in range(MIN_NODE_COUNT, MAX_NODE_COUNT, 1):
        print('Size:', size)
        for i in range(0, TRIAL_COUNT):
            data = gen_bn('polytree', size, prob=1)
            # print(data)
            bn = BayesianNetwork(data) 
            # event = pass #Random selection from bn {'1': True, '2': True}
            # x = pass #Random variable from bn (str)
            x, event = bn.genRandVariables(1+ int(.2 * size))

            enumTimer.start()
            enumAsk(x, event, bn)
            enumTimer.stop()
            lkhdTimer.start()
            likelihoodWeighting(x, event, bn, 2000)
            lkhdTimer.stop()
            gibbsTimer.start()
            gibbsAsk(x, event, bn, 2000)
            gibbsTimer.stop()


        enum_avg_time = enumTimer.reset() / TRIAL_COUNT
        lkhd_avg_time = lkhdTimer.reset() / TRIAL_COUNT
        gibbs_avg_time = gibbsTimer.reset() / TRIAL_COUNT

        resultsList.append([size, enum_avg_time, lkhd_avg_time, gibbs_avg_time])

    return zip(*resultsList)


# sizeList, enumTimeList, lkhdTimeList, gibbsTimeList = runTimeTrials()

import matplotlib.pyplot as plt

def calcKLDivergence(exact, approx):
    result = 0.0
    for index, x in enumerate(exact):
        if approx[index] != 0:
            result += x * np.log(x/ approx[index])
    return result

def plotTimes(sizeList, enumList, lkhdList, gibbsList):
    plt.figure('Avg Time vs Graph Size')
    #plt.plot(sizeList, enumList, c='r', label='Enumerate')
    plt.plot(sizeList, lkhdList, c='b', label='Likelihood')
    plt.plot(sizeList, gibbsList, c='g', label='Gibbs')
    plt.xlabel('Graph Size')
    plt.ylabel('Avg Time')
    plt.title('Avg Time vs Graph Size')
    # plt.legend(['Enumerate Ask', 'Likelihood Weighting', 'Gibbs Sampling'])
    plt.legend(['Likelihood Weighting', 'Gibbs Sampling'])


from collections import defaultdict
def plotAccuracy(lkhdList, gibbsList):
    plt.figure('KL Divergence vs Iterations')
    x, y = zip(*lkhdList)
    lkTemp = defaultdict(list)
    for key, value in lkhdList:
        lkTemp[key].append(value)
    lkPlot = []
    for key in lkTemp:
        lkPlot.append((key, np.mean(lkTemp[key])))
    
    plt.scatter(x, y, c='b', label='Likelihood')
    x, y = zip(*gibbsList)
    gibbsTemp = defaultdict(list)
    for key, value in gibbsList:
        gibbsTemp[key].append(value)
    gibbsPlot = []
    for key in gibbsTemp:
        gibbsPlot.append((key, np.mean(gibbsTemp[key])))
    plt.scatter(x, y, c='g', label='Gibbs')
    plt.plot(*zip(*lkPlot), c='b')
    plt.plot(*zip(*gibbsPlot), c='g')
    
    
    plt.xlabel('Iterations')
    plt.ylabel('KL')
    plt.title('KL Divergence vs Iterations')
    # plt.legend(['Enumerate Ask', 'Likelihood Weighting', 'Gibbs Sampling'])
    plt.legend(['Likelihood Weighting', 'Gibbs Sampling'])
        

#Burglary case from book
def trial_1():
    MIN_ITERATIONS = 10
    MAX_ITERATIONS = 1000
    STEP = 100
    TRIALS = 20
    size = 20
    # data = openFile('BurglaryExample.json')
    # bn = BayesianNetwork(data) 
    # x = 'Burglary'
    # event = dict(JohnCalls=True, MaryCalls=True)
    
    data = gen_bn('polytree', size, prob=0.5)
    bn = BayesianNetwork(data) 
    x, event = bn.genRandVariables(1+ int(.2 * size))
    # x, event = bn.genRandVariables()
    print(x, event)
    lkList = []
    gibbsList = []
    # for i in range(MIN_ITERATIONS, MAX_ITERATIONS+1, STEP):
    # i = MIN_ITERATIONS
    enumValues = enumAsk(x, event, bn)
    i = 8
    while i <= MAX_ITERATIONS:
        print('Iteration', i)

        # enumList.append(i, enumValues)
        for _ in range(0, TRIALS):
            lkValues = likelihoodWeighting(x, event, bn, i)
            gibbsValues = gibbsAsk(x, event, bn, i)
            # print(enumValues, lkValues, gibbsValues)
            klLk =calcKLDivergence(enumValues, lkValues)
            if klLk >= 0:
                lkList.append((i, klLk))
            klGibbs = calcKLDivergence(enumValues, gibbsValues)
            if klGibbs >= 0:
                gibbsList.append((i, klGibbs))
        i *= 2
    plotAccuracy(lkList, gibbsList)
        


trial_1()
plt.show()