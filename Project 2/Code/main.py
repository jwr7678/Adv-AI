"""
A professor wants to know if students are getting enough sleep. Each day, the professor observes whether students sleep in class, 
and whether they have red eyes. The professor has the following domain theory:
-The prior probability of getting enough sleep, with no observations, is 0.7.
-The probability of getting enough sleep on night t is 0.8 given that the student got enough sleep the previous night, and 0.3 if not.
-The probability of having red eyes is 0.2 if the student got enough sleep, and 0.7 if not.
-The probability of sleeping in class is 0.1 if the student got enough sleep, and 0.3 if not.

Formulate this information as a dynamic Bayesian network that the professor could use to filter or predict from a sequence of observations. 
Then reformulate it as a hidden Markov model that has only a single observation variable. Give the probability tables for the model.
Develop a HMM model.
"""

#Only use pomegranate for Structures, not algorithms
# from pomegranate import *
#from pgmpy.models import BayesianModel

#states Enought Sleep
# states = ('Enough Sleep', 'Not Enough Sleep')
# observtions = ('red eyes', 'sleeping in class')
# prior_prob = {'Enough Sleep': 0.7, 'Not Enough Sleep': 0.3}
# #Given E_t-1: E_t
# # transition_model = {
# #     'Enough Sleep': {'Enough Sleep': 0.8, 'Not Enough Sleep': 0.3},
# #     'Not Enough Sleep':  {'Enough Sleep': 0.2, 'Not Enough Sleep': 0.7}
# # }
# #{A: {B:1}} = P(B|A) = 1
# transition_model = {
#     'Enough Sleep': {'Enough Sleep': 0.8, 'Not Enough Sleep': 0.2},
#     'Not Enough Sleep': {'Enough Sleep': 0.3, 'Not Enough Sleep': 0.7}
# }

# observation_model = {
#     'Enough Sleep': {'red eyes': 0.2, 'sleeping in class': 0.1},
#     'Not Enough Sleep': {'red eyes': 0.7, 'sleeping in class': 0.3}
# }
import numpy as np
from decimal import Decimal

def _getProb(dist, dv, iv):
    if dv:
        if iv:
            return dist[0][0]
        else:
            return dist[0][1]
    else:
        if iv:
            return dist[1][0]
        else:
            return dist[1][1]

def _normalize(dist):
    total = 0
    normDist = []
    for i in dist:
        total += i
    for i in dist:
        normDist.append(i / total)
    return normDist


class HiddenMarkovModel:
    def __init__(self,state, observations, transition_model, observation_model, prior_prob):
        self.state = state
        self.observations = observations
        self.transition_model = transition_model
        self.observation_model = observation_model
        if isinstance(prior_prob, list):
            self.prior_prob = prior_prob
        else:
            self.prior_prob = [prior_prob, 1 - prior_prob]

    def formulateObservationModel(self, observation_model):
        model = []
        # for value in observation_model.values():
    #Event will be a dictionary
    #event = {'red eyes': True, 'sleeping in class': True}
    def getObservationDist(self, event):
        dist = []
        for i in [True, False]:
            total = 1.0
            for key, value in event.items():
                p = _getProb(self.observation_model[key], value, i)
                total = total * p
            dist.append(total)
        return dist
        


    def draw_net(self, iterations = 3):
        import networkx as nx
        import matplotlib.pyplot as plt

       # g = nx.DiGraph()
        g = nx.MultiDiGraph()

        ## add nodes
        g.add_node(self.state)
        for i in self.observations:
            g.add_node(i)

        ## add edges
        for i in self.observations:
            #i.add_edge(i, j)
            g.add_edge(self.state, i)


        nx.draw(g, with_labels=True)
        plt.show()

    def getTransitionalDist(self, prior):
        #dist = []
        a = _getProb(self.transition_model, True, prior)
        b = _getProb(self.transition_model, False, prior)
        return [a,b]

    #Original f_vector should be prior_prob
    def forward(self, f_vector, evidence):
        # print("FORWARD")
        # print('f_vector', f_vector)
        # print('T True', self.getTransitionalDist(True))
        # print('T False', self.getTransitionalDist(False))
        a = np.multiply(f_vector[0], self.getTransitionalDist(True))
        b = np.multiply(f_vector[1], self.getTransitionalDist(False))
        prediction = np.add(a, b)
        observedDist = self.getObservationDist(evidence)
        prediction = np.multiply(prediction, observedDist)
        # print(prediction)
        return _normalize(prediction)
        #prediction = np.add()
        #predition = np.add(np.multiply(f_vector[0],self.transition_model ))

    def backward(self, b_vector, evidence):
        observedDist = self.getObservationDist(evidence)
        prediction = np.multiply(observedDist, b_vector)
        print(prediction)
        a = np.multiply(prediction[0],self.getTransitionalDist(True))
        b = np.multiply(prediction[1], self.getTransitionalDist(False))
        #print(_normalize(np.add(a, b)))
        return _normalize(np.add(a,b))

    def forwardBackward(self, evidence, prior = None):
        if prior == None:
            prior = self.prior_prob
        fv = []
        b = [1,1]
        sv = []
        fv.append(prior)
        t = len(evidence)
        for i in range(0, t):
            f = self.forward(fv[i], evidence[i])
            fv.append(f)
        for i in range(t-1, -1, -1):
            sv.insert(0, _normalize(np.multiply(fv[i], b)))
            b = self.backward(b, evidence[i])
        return sv



state = 'Enough Sleep'
observations = ('red eyes', 'sleeping in class')
prior_prob = [0.7, 0.3]
#For 
transition_model = [
    [0.8, 0.3], #Prob of getting enough sleep, given previous night = T, F
    [0.2, 0.7]  #Prob of not getting enough sleep
]
observation_model = {
    'red eyes': [
        [0.2, 0.7], #Prof of red eyes given enough sleep = T, F
        [0.8, 0.3]
    ],
    'sleeping in class': [
        [0.1, 0.3],
        [0.9, 0.7]
    ]
}

observed_events = [
    # {'red eyes': False, 'sleeping in class': False},
    # {'red eyes': False, 'sleeping in class': False},
    # {'red eyes': False, 'sleeping in class': False},
    {'red eyes': False, 'sleeping in class': False},
    {'red eyes': True, 'sleeping in class': False},
    {'red eyes': True, 'sleeping in class': True}
]


# state = 'Rain'
# observations = ('u')
# transition_model = [
#     [0.7, 0.3],
#     [0.3, 0.7],
# ]
# observation_model = {
#     'u': [
#         [0.9, 0.2],
#         [0.1, 0.8]
#     ]
# }
# prior_prob = [0.5, 0.5]

# observed_events = [{'u': True}]

hmm = HiddenMarkovModel(state, observations, transition_model, observation_model, prior_prob)
#f = hmm.forward([0.7, 0.3], [])
# print(f)
# fwrd = [0.7, 0.3]
fwrd = prior_prob
b = [1, 1]
hmm.backward(b, observed_events[0])
for t in range(0,1):
    # fwrd = hmm.forward(fwrd, observed_events[t])
    print(observed_events[t])
    fwrd = hmm.forwardBackward([observed_events[t]], fwrd)
    print(fwrd)
    for i in range(len(fwrd)):
        pass
        # fwrd[i] = round(fwrd[i], 4)
    print(t, fwrd)
    
print(hmm.forwardBackward(observed_events))
#print(hmm.getObservationDist({'red eyes': False, 'sleeping in class': False}))
#print(Decimal(0.8) * Decimal(0.9))
#hmm.draw_net()
    

"""
You will also implement the following algorithms:
• exact HMM smoothing algorithm using constant space, the forward-backward
Country-Dance algorithm, and
• the online fixed-lag smoothing algorithm in Figure 15.6,
• the most likely sequence of states, using the Viterbi algorithm.
You will use these algorithms to compute the state estimation, smoothing, and fixedlag smoothing (report results from lag values over the range [2,5]) probabilities for the
scenarios in Exercise 15.14 but for all t ∈ {1 . . . 25}.
"""