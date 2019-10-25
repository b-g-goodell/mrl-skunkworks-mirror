from math import *
from random import *
from copy import deepcopy
# from itertools import groupby
from graphtheory import *
from simulator import *
from graphtheory import *

# TODO: TXN AMOUNTS

class Player(object):
    def __init__(self, par=None):
        # Need 
        self.data = par
        
    def respond(self, g, my_knowledge):
        # print("Deleting known spurious ring members")
        to_del = [fid for fid in g.red_edges if any([eid[1] == fid[1] and fid != eid for eid in my_knowledge])]
        g.del_edge(to_del)
        
        # print("Weighting graph")
        ct = 0
        dct = 0
        for sig_node in g.right_nodes:
            ct += 1
            if ct/len(g.right_nodes) > (dct+1)*0.099999:
                # print("We are " + str(round(100.0*float(ct/len(g.right_nodes)))) + "% done weighting.")
                dct += 1
            ring = [eid for eid in g.red_edges if eid[1] == sig_node]
            ast = {} # alleged spendtimes
            for eid in ring:
                ast.update({eid: eid[2] - eid[0][1]})
                
            base_likelihood = 1.0
            for eid in ast:
                base_likelihood = base_likelihood*self.data['wallet'](ast[eid])
            for eid in ring:
                g.red_edges[eid] = base_likelihood*self.data['null'](ast[eid])/self.data['wallet'](ast[eid])
                
        # print("Done weighting. Beginning optimization.")
        return g.optimize(1)
                
                
class Challenger(object):
    def __init__(self, par=None):
        self.data = par
        
    def generate(self, par=None):
        # In the stochastic matrix, represent Alice as all but the last two
        # indices.
        sally = Simulator(self.data['simulator'])
        sally.run()
        # print(len(sally.g.right_nodes))
        return sally
        
def tracing_game(par):
    ''' tracing_game executes the actual tracing game. '''
    # print("Beginning tracing game by initializing Chuck and Eve.")
    eve = Player(par['eve'])
    chuck = Challenger(par['chuck'])
    u = len(par['chuck']['simulator']['stochastic matrix'])
    
    # print("Running simulation.")
    sally = chuck.generate()
    
    # print("Getting Eve's response.")
    # Ownership of an edge is the same as ownership of it's signature node.
    # Ownership of a signature node is a pair (k, x) where k is an 
    # owner index in the stochastic matrix, x is the left_node 
    # being spent
    # TODO: This is not correct; Eve receives all ownership info of both 
    # endpoints of all of her edges, not merely ownership information for 
    # the edge itself. 
    eve_edges = [eid for eid in sally.g.red_edges if u-1 in [sally.ownership[eid[0]], sally.ownership[eid[1]]]]
    eve_ownership = dict()
    for eid in eve_edges:
        eve_ownership[eid] = sally.ownership[eid]
        eve_ownership[eid[0]] = sally.ownership[eid[0]]
        eve_ownership[eid[1]] = sally.ownership[eid[1]]
    resp = eve.respond(sally.g, eve_ownership) # response to the simulator (ownership dict)
            
    # print("Compiling confusion matrix.")
    positives = {}
    negatives = {}
    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}
    
    for eid in sally.g.red_edges:
        assert eid in sally.ownership
        ow = sally.ownership[eid]
        # print(ow)
        if ow[0] != -1 % u and ow[0] != -2 % u:
            positives[eid] = eid
        else:
            negatives[eid] = eid
            
    for eid in positives:
        if eid in resp:
            true_positives[eid] = eid
        else:
            false_negatives[eid] = eid
            
    for eid in negatives:
        if eid in resp:
            false_positives[eid] = eid
        else:
            true_negatives[eid] = eid
            
    P = len(positives)
    N = len(negatives)
    TP = len(true_positives)
    TN = len(true_negatives)
    FP = len(false_positives)
    FN = len(false_negatives)
    
    if P != 0:
        TPR = TP/P
        FNR = FN/P
    else:
        TPR = None
        FNR = None
    if N != 0:
        TNR = TN/N
        FPR = FP/N
    else:
        TNR = None
        FPR = None
    if TP + FP != 0:
        PPV = TP/(TP + FP)
        FDR = FP/(TP + FP)
    else:
        PPV = None
        FDR = None
    if TN + FN != 0:
        NPV = TN/(TN + FN)
        FOR = FN/(TN + FN)
    else:
        NPV = None
        FOR = None
    if TP + FN + FP != 0:
        TS = TP/(TP + FN + FP)
    else:
        TS = None
    if P + N != 0:
        ACC = (TP + TN)/(P + N)
    else:
        ACC = None
    if 2*TP + FP + FN != 0:
        F1 = 2*TP/(2*TP + FP + FN)
    else:
        F1 = None
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0:
        MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    else:
        MCC = None
    if TPR is not None and TNR is not None:
        BM = TPR + TNR - 1
    else:
        BM = None
    if PPV is not None and NPV is not None:
        MK = PPV + NPV - 1
    else:
        MK = None
    
    to_write = str()
    to_write += "\nRESULTS ======\n\n"
    to_write += "Positives = " + str(P) + "\n"
    to_write += "Negatives = " + str(N) + "\n"
    to_write += "True Positives = " + str(TP) + "\n"
    to_write += "True Negatives = " + str(TN) + "\n"
    to_write += "False Positives = " + str(FP) + "\n"
    to_write += "False Negatives = " + str(FN) + "\n"
    to_write += "True Positive Rate = " + str(TPR) + "\n"
    to_write += "True Negative Rate = " + str(TNR) + "\n"
    to_write += "Positive Predictive Value = " + str(PPV) + "\n"
    to_write += "Negative Predictive Value = " + str(NPV) + "\n"
    to_write += "False Negative Rate = " + str(FNR) + "\n"
    to_write += "False Positive Rate = " + str(FPR) + "\n"
    to_write += "False Discovery Rate = " + str(FDR) + "\n"
    to_write += "False Omission Rate = " + str(FOR) + "\n"
    to_write += "Threat Score = " + str(TS) + "\n"
    to_write += "Accuracy = " + str(ACC) + "\n"
    to_write += "F1 Score = " + str(F1) + "\n"
    to_write += "Matthews Correlation Coefficient = " + str(MCC) + "\n"
    to_write += "Informedness = " + str(BM) + "\n"
    to_write += "Markedness = " + str(MK) + "\n"
    to_write += "        ======\n"
    with open(par['filename'], "w+") as wf:
        wf.write(str(to_write))
    print(to_write)

par = {}

par['filename'] = "../data/confusion.txt"

par['chuck'] = {}
par['chuck']['simulator'] = {}
par['chuck']['simulator']['min spendtime'] = 2
par['chuck']['simulator']['runtime'] = 8
par['chuck']['simulator']['filename'] = "../data/output.txt"
par['chuck']['simulator']['stochastic matrix'] = [[0.0, 0.9, 0.1], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]
par['chuck']['simulator']['hashrate'] = [0.8075, 0.125, 0.0625]
par['chuck']['simulator']['spendtimes'] = []

# Alice will always be the first few indices.
# Alice's spendtime has expectation 20 blocks and support
# on min_spendtime, min_spendtime + 1, ...
par['chuck']['simulator']['spendtimes'] += [lambda 
    x: 0.01*((1.0-0.01)**(x-par['chuck']['simulator']['min spendtime']))]
# Eve will always be second-to-last-index.
# Eve's spenditme has expectation 100 blocks and support
# on min_spendtime, min_spendtime + 1, ...
par['chuck']['simulator']['spendtimes'] += [lambda 
    x: 0.025*((1.0-0.025)**(x-par['chuck']['simulator']['min spendtime']))]
# Bob will be last index.
# Bob's (background) spendtime has expectation 40 blocks and support
# on min_spendtime, min_spendtime + 1, ...
par['chuck']['simulator']['spendtimes'] += [lambda 
    x: 0.0125*((1.0-0.0125)**(x-par['chuck']['simulator']['min spendtime']))]
    
par['chuck']['simulator']['ring size'] = 11
par['chuck']['simulator']['reporting modulus'] = 1

# Eve's hypotheses about Bob's behavior (wallet) and Alice's behavior (null)
# match the distributions above.
par['eve'] = {}
par['eve']['min spendtime'] = deepcopy(par['chuck']['simulator']['min spendtime'])
par['eve']['null'] = par['chuck']['simulator']['spendtimes'][0]
par['eve']['wallet'] = par['chuck']['simulator']['spendtimes'][-1]

ss = 16

for k in range(ss): 
    par['chuck']['simulator']['runtime'] += 1
    tracing_game(par)
