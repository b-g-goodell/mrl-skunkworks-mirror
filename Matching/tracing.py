from copy import deepcopy
from random import random, choice, sample, randrange
from math import pi, log
from graphtheory import *
from simulator import *
from graphtheory import *

#if not os.path.exists('data'):
#    os.makedirs('data')


def player_one(player_aux):
    # return inp
    pass


def challenger_one(inp):
    sally = make_simulator(inp)
    sally.run()
    g = deepcopy(sally.g)
    eve_left_nodes = [_ for _ in g.left_nodes if sally.ownership[_] == 1]
    eve_right_nodes = [_ for _ in g.right_nodes if sally.ownership[_] == 1]
    for _ in g.red_edges:
        assert sally.ownership[_[1]] == sally.ownership[_[0]]
    eve_true_reds = [(_, g.real_red_edges[_]) for _ in g.red_edges if sally.ownership[_[1]] == 1)
    return g, eve_left_nodes, eve_right_nodes, eve_true_reds, sally
    
def remove_spurious(g, reddies, player_aux):
    h = deepcopy(g)
    edges_to_del = [_ for _ in g.red_edges if _ not in reddies]
    h.del_edge(edges_to_del)
    return h
    
def apply_weights(g, player_aux="spendtime"):
    if player_aux['weight type'] == "spendtime":
        ct = 0
        dct = 0
        for nid in g.right_nodes:
            ct += 1
            if ct / len(g.right_nodes) > (dct + 1) * 0.099999:
                print("We are " + str(round(100.0*float(ct/len(h.right_nodes)))) + "% done weighting.")
                dct += 1
            ring = [eid for eid in g.red_edges if eid[1] == nid]
            ast = {}  # alleged spendtimes
            for eid in ring:
                # eid = ((left_node_id, left_node_tag), (right_node_id, right_node_tag), edge_tag)
                # tags are ages
                age_of_ring_member = eid[2] - eid[0][1] + 1
                assert age_of_ring_member >= sally.minspendtime
                assert age_of_ring_member >= MIN_SPENDTIME
                assert player_aux['spendtimes'][-1](age_of_ring_member) > 0.0  # Bob's spendtime
                ast.update({eid: age_of_ring_member})

            likelihood = 0.0
            for eid in ast:
                base_likelihood += log(inp_sim_par['spendtimes'][-1](ast[eid]))
            for eid in ring:
                h.red_edges[eid] = base_likelihood + inp_sim_par['spendtimes'][0](ast[eid]) - inp_sim_par['spendtimes'][-1](ast[eid])


def player_two(g, lefties, righties, reddies, player_aux):
    h = deepcopy(g)
    h = remove_spurious(h, reddies, player_aux)
    h = apply_weights(h, player_aux)
    mle = h.optimize(1)
    return mle
    
def challenger_two(sally, mle):
    """ Report confusion table  """
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
        if ow == u - 1 or ow == u - 2:
            negatives[eid] = eid
        else:
            positives[eid] = eid
        assert eid in negatives or eid in positives

    # print("Number of red edges = " + str(len(sally.g.red_edges)))
    # print("Number of negatives = " + str(len(negatives)))
    # print("Number of positives = " + str(len(positives)))

    assert len(sally.g.red_edges) == len(negatives) + len(positives)

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
        TPR = TP / P
        FNR = FN / P
    else:
        TPR = None
        FNR = None
    if N != 0:
        TNR = TN / N
        FPR = FP / N
    else:
        TNR = None
        FPR = None
    if TP + FP != 0:
        PPV = TP / (TP + FP)
        FDR = FP / (TP + FP)
    else:
        PPV = None
        FDR = None
    if TN + FN != 0:
        NPV = TN / (TN + FN)
        FOR = FN / (TN + FN)
    else:
        NPV = None
        FOR = None
    if TP + FN + FP != 0:
        TS = TP / (TP + FN + FP)
    else:
        TS = None
    if P + N != 0:
        ACC = (TP + TN) / (P + N)
    else:
        ACC = None
    if 2 * TP + FP + FN != 0:
        F1 = 2 * TP / (2 * TP + FP + FN)
    else:
        F1 = None
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
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

    return [P, N, TP, TN, FP, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, ACC, F1, MCC, BM, MK]


def play_game(player_aux):
    inp = player_one(player_aux)
    g, lefties, righties, reddies, sally = challenger_one(inp)
    mle = player_two(g, lefties, righties, reddies, player_aux)
    results = challenger_two(sally, mle)
    

def expand_stoch_mat(true_stochastic_matrix, k):
    """ Helper function: take churn number, return a stochastic matrix """
    size = k + len(true_stochastic_matrix)
    m = [[0.0 for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            if i < k:
                if j == i + 1:
                    # Such that party with index 0 (Alice with churn number k) always sends to party with index 1
                    # (Alice with churn number k - 1) with probability 1.0. Same with index i sending to index j
                    # for each i < k.
                    m[i][j] = 1.0
            elif j == 0:
                # And such that all parties send to Alice with index 0 with the same probability as sending
                # to Alice in the ground truth matrix
                m[i][j] = true_stochastic_matrix[i - k][0]
            elif j > k:
                # And such that all parties send to Eve or Bob with the same probability as in the ground truth
                m[i][j] = true_stochastic_matrix[i - k][j - k]
    return m


def is_sally_suitable(sally):
    """ Check that all possible owners own at least one edge in sally.g """
    owners = []
    for eid in sally.g.red_edges:
        if sally.ownership[eid] not in owners:
            owners += [sally.ownership[eid]]
        if len(owners) == 3:
            break
    return len(owners) == 3


def run_experiment(inp_sim_par, sm, label, verbosity):
    """ Actually run the experiment:
        sim_par = dictionary with input parameters for Simulator object
        r = ring_size
        l = churn length
        a = alice spendtime (lambda function)
        e = eve spendtime (lambda function, not used by eve but she knows it)
        b = bob spendtime (lambda function)
        hr = hashrate list
        sm = stochastic matrix
        label = label for this experiment.
        FIRST, challenger generates a graph and sends it to player.
        SECOND, player attempts to find a true ledger history and sends it back to the challenger.
        THIRD, the challenger judges success.
    """
    # FIRST: Player gets a chance to pick the stochastic matrix and other stuff
    inp = player_one()
    # TODO: player_one not yet written (isthmus?) use the following until then
    inp = 
    
    # FIRST: challenger generates a graph and sends it to player.

    # Extract number of transacting parties.
    u = len(sm)

    # Initialize a simulator
    sally = Simulator(inp_sim_par, verbosity)
    ct = 0

    print("Constructing ledger.")
    # Simulate a ledger; do so until a ledger is found in which all parties own at least one edge.
    while sally.halting_run():
        ct += 1
        if ct % 100 == 0:
            print(".", end='')
        pass
    while not is_sally_suitable(sally):
        print()
        sally = Simulator(inp_sim_par)
        while sally.halting_run():
            ct += 1
            if ct % 100 == 0:
                print(".", end='')
            pass

    h = deepcopy(sally.g)
    # Send h to player (mock)

    print("Setting ownership")

    # SECOND: Eve gains her own knowledge from running a KYC exchange and tries to find an estimate of ledger history.
    # Edges known by Eve and associated ownership info
    eve_edges = [eid for eid in h.red_edges if sally.ownership[eid[0]] == u - 2]
    eve_ownership = dict()
    for eid in eve_edges:
        eve_ownership[eid] = sally.ownership[eid]
        eve_ownership[eid[0]] = sally.ownership[eid[0]]
        eve_ownership[eid[1]] = sally.ownership[eid[1]]

    # Eve deletes known spurious ring members
    print("Deleting known spurious ring members")
    to_del = [fid for fid in h.red_edges if any([eid[1] == fid[1] and fid != eid for eid in eve_ownership])]
    h.del_edge(to_del)

    # Eve weights graph
    print("Eve starts weighting the graph.")
    ct = 0
    dct = 0
    for sig_node in h.right_nodes:
        ct += 1
        if ct / len(h.right_nodes) > (dct + 1) * 0.099999:
            print("We are " + str(round(100.0*float(ct/len(h.right_nodes)))) + "% done weighting.")
            dct += 1
        ring = [eid for eid in h.red_edges if eid[1] == sig_node]
        ast = {}  # alleged spendtimes
        for eid in ring:
            # eid = ((left_node_id, left_node_tag), (right_node_id, right_node_tag), edge_tag)
            # tags are ages
            age_of_ring_member = eid[2] - eid[0][1] + 1
            assert age_of_ring_member >= sally.minspendtime
            assert age_of_ring_member >= MIN_SPENDTIME
            assert inp_sim_par['spendtimes'][-1](age_of_ring_member) > 0.0  # Bob's spendtime
            ast.update({eid: age_of_ring_member})

        if any([eid in eve_ownership for eid in ring]):
            for eid in ring:
                if eid in eve_ownership:
                    h.red_edges[eid] = 1.0
                    for fid in ring:
                        if fid != eid:
                            h.red_edges[fid] = 0.0

        base_likelihood = 1.0
        for eid in ast:
            base_likelihood = base_likelihood * inp_sim_par['spendtimes'][-1](ast[eid])
        for eid in ring:
            h.red_edges[eid] = base_likelihood * inp_sim_par['spendtimes'][0](ast[eid]) / inp_sim_par['spendtimes'][-1](ast[eid])

    # Eve finds optimal matching
    print("Eve finds optimal match.")
    x = h.optimize(1)
    print(x, len(h.left_nodes))
    fn = GRAPH_FILENAME[:-4] + str(label) + GRAPH_FILENAME[-4:]
    with open(fn, "a") as wf:
        line = ""
        for edge_ident in x:
            line += "Output "
            line += str(edge_ident[0])
            line += " owned by "
            line += sally.owner_names[sally.ownership[edge_ident[0]]]
            line += " is thought by Eve to have created ring signature "
            line += str(edge_ident[1])
            line += ". In reality, this ring signature is owned by  "
            line += sally.owner_names[sally.ownership[edge_ident[1]]]
            line += ".\n"
        wf.write(line)


    # Send h to challenger (mock)

    # THIRD: Challenger judges success.
    print("Interpreting results and constructing confusion matrix.")
    results = interpret(sally, x)
    # [P, N, TP, TN, FP, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, ACC, F1, MCC, BM, MK]
    line = ""
    for thing in results:
        line += "," + str(thing)
    line += "\n"
    return line


# PARAMETER SPACE EXPLORATION:
line = "RUN"
line += ",SAMPLE"
line += ",SIM LABEL"
line += ",RING_SIZE"
line += ",CHURN_LENGTH"
line += ",ALICE_EXP_SPENDTIME"
line += ",EVE_EXP_SPENDTIME"
line += ",BOB_EXP_SPENDTIME"
line += ",ALICE_HASHRATE"
line += ",EVE_HASHRATE"
line += ",BOB_HASHRATE"
line += ",pAA,pAE,pAB,pEA,pEE,pEB,pBA,pBE,pBB"
line += ",P,N,TP,TN,FP,FN,TPR,TNR,PPV,NPV,FNR,FPR,FDR,FOR,TS,ACC,F1,MCC,INF,MRK\n"
with open(FILENAME, "w+") as wf:
    wf.write(line)

verbosity = True

tot = len(RING_SIZES)*len(CHURN_LENGTHS)*(len(SPENDTIME_LAMBDAS)**3)*len(HASHRATES)*SAMPLE_SIZE
ct = 0
for ring_size in RING_SIZES:
    for churn_length in CHURN_LENGTHS:
        stoch_mat = get_stoch_mat(churn_length)
        for ia in range(len(SPENDTIME_LAMBDAS)):
            alice_spendtime = SPENDTIME_LAMBDAS[ia]
            alice_exp_spendtime = EXP_SPENDTIMES[ia]
            for ie in range(len(SPENDTIME_LAMBDAS)):
                eve_spendtime = SPENDTIME_LAMBDAS[ie]
                eve_exp_spendtime = EXP_SPENDTIMES[ie]
                for ib in range(len(SPENDTIME_LAMBDAS)):
                    # If alice and bob have the same dist, they are indistinguishable.
                    # Cases when ia = ib can be used as Alice's best-case anonymity.
                    bob_spendtime = SPENDTIME_LAMBDAS[ib]
                    bob_exp_spendtime = EXP_SPENDTIMES[ib]
                    sim_spendtimes = [deepcopy(alice_spendtime) for i in range(churn_length+1)]
                    sim_spendtimes += [deepcopy(eve_spendtime), deepcopy(bob_spendtime)]
                    for hashrate in HASHRATES:
                        # print(hashrate)
                        sim_hashrate = [hashrate[0]]
                        sim_hashrate += [0.0 for i in range(churn_length)]
                        sim_hashrate += [hashrate[i-churn_length] for i in range(churn_length + 1, churn_length + len(hashrate))]
                        try:
                            assert sum(sim_hashrate)==1.0
                        except AssertionError:
                            print("Woops, tried to use a hashrate vector " + str(sim_hashrate) + " that doesn't sum to 1.0!")
                            assert False
                        for c in range(SAMPLE_SIZE):
                            line = "RUN"
                            line += ",SAMPLE"
                            line += ",SIM LABEL"
                            line += ",RING_SIZE"
                            line += ",CHURN_LENGTH"
                            line += ",ALICE_EXP_SPENDTIME"
                            line += ",EVE_EXP_SPENDTIME"
                            line += ",BOB_EXP_SPENDTIME"
                            line += ",ALICE_HASHRATE"
                            line += ",EVE_HASHRATE"
                            line += ",BOB_HASHRATE"
                            line += ",pAA,pAE,pAB,pEA,pEE,pEB,pBA,pBE,pBB"
                            line += ",P,N,TP,TN,FP,FN,TPR,TNR,PPV,NPV,FNR,FPR,FDR,FOR,TS,ACC,F1,MCC,INF,MRK\n"

                            label_msg = (ct, c, ring_size, churn_length, alice_exp_spendtime, eve_exp_spendtime, bob_exp_spendtime, hashrate[0], hashrate[1], hashrate[2], TRUE_STOCHASTIC_MATRIX[0][0],  TRUE_STOCHASTIC_MATRIX[0][1],  TRUE_STOCHASTIC_MATRIX[0][2],  TRUE_STOCHASTIC_MATRIX[1][0],  TRUE_STOCHASTIC_MATRIX[1][1],  TRUE_STOCHASTIC_MATRIX[1][2],  TRUE_STOCHASTIC_MATRIX[2][0],  TRUE_STOCHASTIC_MATRIX[2][1],  TRUE_STOCHASTIC_MATRIX[2][2])
                            label = str(hash(label_msg))
                            label = label[-8:]

                            line = str(ct) + "," + str(c) + "," + str(label) + "," + str(ring_size) + "," + str(churn_length) + "," + str(alice_exp_spendtime) + "," + str(eve_exp_spendtime) + "," + str(bob_exp_spendtime) + "," + str(hashrate[0]) + "," + str(hashrate[1]) + "," + str(hashrate[2]) + "," + str(TRUE_STOCHASTIC_MATRIX[0][0]) + "," + str(TRUE_STOCHASTIC_MATRIX[0][1]) + "," + str(TRUE_STOCHASTIC_MATRIX[0][2]) + "," + str(TRUE_STOCHASTIC_MATRIX[1][0]) + "," + str(TRUE_STOCHASTIC_MATRIX[1][1]) + "," + str(TRUE_STOCHASTIC_MATRIX[1][2]) + "," + str(TRUE_STOCHASTIC_MATRIX[2][0]) + "," + str(TRUE_STOCHASTIC_MATRIX[2][1])+ "," + str(TRUE_STOCHASTIC_MATRIX[2][2])   # No comma at the end

                            sim_par = dict()
                            fn = SIM_FILENAME[:-4] + str(label) + SIM_FILENAME[-4:]
                            sim_par.update({'runtime': RUNTIME, 'filename': fn, 'stochastic matrix': stoch_mat, 'hashrate': sim_hashrate, 'min spendtime': MIN_SPENDTIME, 'spendtimes': sim_spendtimes, 'ring size': ring_size, 'reporting modulus': 1})

                            print("Beginning run_experiment.")
                            line += run_experiment(sim_par, stoch_mat, label, verbosity)
                            with open(FILENAME, "a+") as wf:
                                wf.write(line)

                            ct += 1
                            print("\n" + str(floor(float(ct)/float(tot)*100.0)) + "% of parameter space explored.")

