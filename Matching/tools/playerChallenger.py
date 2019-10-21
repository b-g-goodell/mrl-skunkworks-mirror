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
        
    def respond(self, g, my_edges):
        print("Deleting known spurious ring members")
        for eid in my_edges:
            to_del = []
            for fid in g.red_edges:
                if fid[1] == eid[1] and fid != eid:
                    to_del += [fid]
            for fid in to_del:
                del g.red_edges[fid]
        print("Weighting graph")
        ct = 0
        dct = 0
        for sig_node in g.right_nodes:
            ct += 1
            if ct/len(g.right_nodes) > (dct+1)*0.099999:
                print("We are " + str(round(100.0*float(ct/len(g.right_nodes)))) + "% done weighting.")
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
                
        print("Done weighting. Beginning optimization.")
        return g.optimize(1)
                
                
class Challenger(object):
    def __init__(self, par=None):
        self.data = par
        
    def generate(self, par=None):
        # In the stochastic matrix, represent Alice as all but the last two
        # indices.
        sally = Simulator(self.data['simulator'])
        sally.run()
        #print(len(sally.g.right_nodes))
        return sally
        
def tracing_game(par):
    print("Beginning tracing game by initializing Chuck and Eve.")
    eve = Player(par['eve'])
    chuck = Challenger(par['chuck'])
    u = len(par['chuck']['simulator']['stochastic matrix'])
    
    print("Running simulation.")
    sally = chuck.generate() # get the simulator
    
    print("Getting Eve's response.")
    # Ownership of an edge is the same as ownership of it's signature node.
    # Ownership of a signature node is a pair (k, x) where k is an 
    # owner index in the stochastic matrix, x is the left_node 
    # being spent
    resp = eve.respond(sally.g, [eid for eid in sally.g.red_edges if \
        sally.ownership[eid][0] == u-1]) # response to the simulator (ownership dict)
            
    print("Compiling confusion matrix.")
    positives = {}
    negatives = {}
    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}
    
    for eid in sally.g.red_edges:
        assert eid in sally.ownership
        if eid in resp:
            # This indicates Eve thinks this eid belongs to Alice.
            positives[eid] = eid
        else:
            # This indicates Eve thinks/knows this eid does not.
            negatives[eid] = eid
            
    for eid in positives:
        if sally.ownership[eid][0] != -1 % u and sally.ownership[eid][0] != -2 % u:
            true_positives[eid] = eid
        else:
            false_positives[eid] = eid
            
    for eid in negatives:
        if sally.ownership[eid][0] != -1 % u and sally.ownership[eid][0] != -2 % u:
            false_negatives[eid] = eid
        else:
            true_negatives[eid] = eid
            
    P = len(positives)
    N = len(negatives)
    TP = len(true_positives)
    TN = len(true_negatives)
    FP = len(false_positives)
    FN = len(false_negatives)
    
    TPR = TP/P
    TNR = TN/N
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FNR = FN/P
    FPR = FP/N
    FDR = FP/(FP + TP)
    FOR = FN/(FN + TN)
    TS = TP/(TP+FN+FP)

    ACC = (TP+TN)/(P+N)
    F1 = 2*TP/(2*TP + FP + FN)
    MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    BM = TPR + TNR - 1
    MK = PPV + NPV - 1
        
    with open(par['filename'], "w+") as wf:
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
        print(to_write)
        wf.write(str(to_write))
        
par = {}

par['filename'] = "../data/confusion.txt"

par['chuck'] = {}
par['chuck']['simulator'] = {}
par['chuck']['simulator']['runtime'] = 15
par['chuck']['simulator']['filename'] = "../data/output.txt"
par['chuck']['simulator']['stochastic matrix'] = [[0.0, 0.9, 0.1], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]
par['chuck']['simulator']['hashrate'] = [0.8075, 0.125, 0.0625]
par['chuck']['simulator']['min spendtime'] = 10
par['chuck']['simulator']['spendtimes'] = []
# Alice's spendtime has expectation 20 blocks
par['chuck']['simulator']['spendtimes'] += [lambda 
    x: 0.05*((1.0-0.05)**(x-par['chuck']['simulator']['min spendtime']))]
# Eve's spenditme has expectation 100 blocks
par['chuck']['simulator']['spendtimes'] += [lambda 
    x: 0.01*((1.0-0.01)**(x-par['chuck']['simulator']['min spendtime']))]
# Bob's (background) spendtime has expectation 40 blocks.
par['chuck']['simulator']['spendtimes'] += [lambda 
    x: 0.025*((1.0-0.025)**(x-par['chuck']['simulator']['min spendtime']))]
par['chuck']['simulator']['ring size'] = 11

par['eve'] = {}
par['eve']['min spendtime'] = deepcopy(par['chuck']['simulator']['min spendtime'])
par['eve']['null'] = lambda x: 0.05*((1.0-0.05)**(x-par['chuck']['simulator']['min spendtime']))
par['eve']['wallet'] = lambda x: 0.025*((1.0-0.025)**(x-par['chuck']['simulator']['min spendtime']))

tracing_game(par)
