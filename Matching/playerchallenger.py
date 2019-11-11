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
    # TODO: Modify to ensure Eve receives all info she is due eg amounts
    eve = Player(par['eve'])
    chuck = Challenger(par['chuck'])
    u = len(par['chuck']['simulator']['stochastic matrix'])
    sally = chuck.generate()
    eve_edges = [eid for eid in sally.g.red_edges if u-1 in [sally.ownership[eid[0]], sally.ownership[eid[1]]]]
    eve_ownership = dict()
    for eid in eve_edges:
        eve_ownership[eid] = sally.ownership[eid]
        eve_ownership[eid[0]] = sally.ownership[eid[0]]
        eve_ownership[eid[1]] = sally.ownership[eid[1]]
    resp = eve.respond(sally.g, eve_ownership)  # response to the simulator (ownership dict)
    return sally, resp

def interpret(par, sally, resp):
    """ interpret prints a confusion matrix to the screen and to file
    """
    positives = {}
    negatives = {}
    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}

    u = len(sally.stoch_matrix)
    
    for eid in sally.g.red_edges:
        assert eid in sally.ownership
        ow = sally.ownership[eid]
        # print(ow)
        if ow[0] == u-1 or ow[0] == u-2:
            negatives[eid] = eid
        else:
            positives[eid] = eid
            
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

    assert P + N == len(sally.g.red_edges)
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

def go():
    par = {}

    par['filename'] = "../data/confusion.txt"

    par['chuck'] = {}
    par['chuck']['simulator'] = {}
    par['chuck']['simulator']['min spendtime'] = 1
    par['chuck']['simulator']['runtime'] = 8
    par['chuck']['simulator']['filename'] = "../data/output.txt"
    # Index order: Alice is @ 0, 1, 2, ..., then Eve @ -2, then Bob @ -1
    par['chuck']['simulator']['stochastic matrix'] = [[0.0, 0.9, 0.1], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]

    alice_hashrate = 0.33
    eve_hashrate = 0.33
    bob_hashrate = 1.0 - alice_hashrate - eve_hashrate
    par['chuck']['simulator']['hashrate'] = [alice_hashrate, eve_hashrate, bob_hashrate]
    par['chuck']['simulator']['spendtimes'] = []

    u = len(par['chuck']['simulator']['stochastic matrix'])
    # print(u)

    par['chuck']['simulator']['ring size'] = 11
    par['chuck']['simulator']['reporting modulus'] = 1

    # These must be >= 1
    exp_alice_spendtime = 5*par['chuck']['simulator']['runtime']
    exp_bob_spendtime = 3*par['chuck']['simulator']['runtime']
    exp_eve_spendtime = 4*par['chuck']['simulator']['runtime']	

    # Which forces these to be 0<= * <= 1.0
    pa = 1/exp_alice_spendtime
    pb = 1/exp_bob_spendtime
    pe = 1/exp_eve_spendtime

    # Alice will always be the first few indices.
    # Alice's spendtime has expectation 20 blocks and support
    # on min_spendtime, min_spendtime + 1, ...
    par['chuck']['simulator']['spendtimes'] += [lambda 
        x: pa*((1.0-pa)**(x-par['chuck']['simulator']['min spendtime']))]

    # Eve will always be second-to-last-index.
    # Eve's spenditme has expectation 100 blocks and support
    # on min_spendtime, min_spendtime + 1, ...
    par['chuck']['simulator']['spendtimes'] += [lambda 
        x: pe*((1.0-pe)**(x-par['chuck']['simulator']['min spendtime']))]

    # Bob will be last index.
    # Bob's (background) spendtime has expectation 40 blocks and support
    # on min_spendtime, min_spendtime + 1, ...
    par['chuck']['simulator']['spendtimes'] += [lambda 
        x: pb*((1.0-pb)**(x-par['chuck']['simulator']['min spendtime']))]
        
    # Eve's hypotheses about Bob's behavior (wallet) and Alice's behavior (null)
    # match the distributions above.
    par['eve'] = {}
    par['eve']['min spendtime'] = deepcopy(par['chuck']['simulator']['min spendtime'])
    par['eve']['null'] = par['chuck']['simulator']['spendtimes'][0]
    par['eve']['wallet'] = par['chuck']['simulator']['spendtimes'][-1]

    ss = 16

    ct = 0
    mcc_none = []

    for k in range(ss): 
        par['chuck']['simulator']['ring size'] += 3
        ct = 0

        print("Calling tracing_game. Looking for a ledger")
        sally, resp = tracing_game(par)
        # Recall: ownership[new_eid] = ownership[new_eid[1]]
        alice_edges = [x for x in sally.g.red_edges if sally.ownership[x[1]] != u-2 and sally.ownership[x[1]] != u-1]
        al = len(alice_edges)

        eve_edges = [x for x in sally.g.red_edges if sally.ownership[x[1]] == u - 2]
        el = len(eve_edges)

        bob_edges = [x for x in sally.g.red_edges if sally.ownership[x[1]] == u - 1]
        bl = len(bob_edges)

        while al == 0 or el == 0 or bl == 0:
            # print("Degenerate ledger. Calling again.")

            with open("temp.txt", "a") as wf:
                s = "\n\nAnother degenerate ledger found. Here are the deets.\n"
                s += "\nOwnership dict====\n"
                for x in sally.ownership:
                    s += str(x) + ": " + str(sally.ownership[x]) + "\n"
                s += "\nLeft nodes====\n"
                for x in sally.g.left_nodes:
                    s += str(x) + "\n"
                s += "\n\nRight nodes====\n"
                for x in sally.g.right_nodes:
                    s += str(x) + "\n"
                s += "\n\nRed edges====\n"
                for x in sally.g.red_edges:
                    s += str(x) + "\n"
                s += "\n\nBlue edges====\n"
                for x in sally.g.blue_edges:
                    s += str(x) + "\n"
                s += "\n\nAlice edges====\n"
                
                for x in alice_edges:
                    s += str(x) + "\n"
                s += "\n\nEve edges====\n"
                for x in eve_edges:
                    s += str(x) + "\n"
                s += "\n\nBob edges====\n"
                for x in bob_edges:
                    s += str(x) + "\n"
                wf.write(s)
            ct += 1

            sally, resp = tracing_game(par)
            al = len([x for x in sally.g.red_edges if sally.ownership[x][0] in range(u-2)])
            el = len([x for x in sally.g.red_edges if sally.ownership[x][0] == u-2])
            bl = len([x for x in sally.g.red_edges if sally.ownership[x][0] == u-1])

        mcc = interpret(par, sally, resp)
        mcc_none += [mcc is None]

    # print("Repeats vector = " + str(repeats))
    print("mcc_none = " + str(mcc_none))

go()
