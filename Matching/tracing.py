from simulator import Simulator
from math import sqrt, floor
from copy import deepcopy

# Presume we have the following:

# Alice sends to herself about 10% of her own transactions, moving money for various reasons: p[a][a] = 0.1
# Alice sends about 10% of her transactions to Eve representing her exchange activity: p[a][e] = 0.1
# Alice sends the remainder of her activity to background players: p[a][b] = 1.0 - 0.1 - 0.1 (ensure sums to 1.0)

# Eve is an exchange who transacts with her customers. Alice represents about 1/25 of Eve's activity reflecting the
# hashrate ratio between Alice and the rest of the economy. So p[e][a] = 0.04
# Eve never sends to herself on-chain so p[e][e] = 0.0
# Eve sends the rest to bob, so p[e][b] = 1.0 - 0.04

# Bob is the sum total of the rest of the economy.
# Bob sends around 10% of his transactions to Eve, reflecting Eve's share of about 10% of miner activity. So p[b][e] = 0.1
# Bob sends a small proportion of the remainder of his transactions to Alice, reflecting the likelihood a random Bob
# knows this specific Alice. Say p[b][a] = 2**(-7).
# This leaves Bob-to-Bob transactions at p[b][b] = 1.0 - 0.1 - 2**(-7)

TRUE_STOCHASTIC_MATRIX = [[0.1, 0.1, 1.0 - 0.1 - 0.1], [0.04, 0.0, 1.0 - 0.04], [2**(-7), 0.1, 1.0 - 0.1 - 2**(-7)]]

MIN_SPENDTIME = 3
RUNTIME = 10
# RING_SIZES = [2**i for i in range(2, 4)]
RING_SIZES = [4]
# CHURN_LENGTHS = [i for i in range(1, 4)]
CHURN_LENGTHS = [0]
# EXP_SPENDTIMES = [10*i for i in range(1, 5)] # Expected number of blocks after min_spendtime each player spends
EXP_SPENDTIMES = [10]
SPENDTIME_LAMBDAS = [lambda x: (1.0/ex)*((1.0 - (1.0/ex)**(x - MIN_SPENDTIME))) for ex in EXP_SPENDTIMES]
# With N increments indexed 0, ..., N-1, then i/(N-1) partitions [0,1] uniformly
# If N = 2, everything is trivial.
# If N = 3, then everyone can have proportional hashrate 0.0, 0.5, or 1.0. This contradicts requiring x, y < 1/2
# If N = 4, then Alice can have hashrate 0.33 and so can Bob and so can Eve. So this is the first case in which
# the HASHRATES vector is non-empty. However, Alice and Eve each have two possible hashrates: 0 and 1/3, so you can
# think of this as a very extreme case where Eve is trying to track someone mining 1/3 of all Monero.
# For N >= 5, we start getting some gradient in hashrates with less extreme cases.
HASHRATE_INCREMENT = 4  # N >= 4 required, N >> 4 for wider exploration.
HASHRATES = [[float(x)/(HASHRATE_INCREMENT-1), float(y)/(HASHRATE_INCREMENT-1), float(1.0 - x/(HASHRATE_INCREMENT-1) - y/(HASHRATE_INCREMENT-1))] for x in range(HASHRATE_INCREMENT) for y in range(HASHRATE_INCREMENT) if float(x)/(HASHRATE_INCREMENT-1) < 1/2 and float(y)/(HASHRATE_INCREMENT-1) < 1/2]
FILENAME = "output.csv"
SIM_FILENAME = "simulator-output.csv"
SAMPLE_SIZE = 1

input_parameters = {}


def get_stoch_mat(k):
    """ Helper function: take churn number, return a stochastic matrix """
    size = k + len(TRUE_STOCHASTIC_MATRIX)
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
                m[i][j] = TRUE_STOCHASTIC_MATRIX[i - k][0]
            elif j > k:
                # And such that all parties send to Eve or Bob with the same probability as in the ground truth
                m[i][j] = TRUE_STOCHASTIC_MATRIX[i - k][j - k]
    return m


def interpret(sally, resp):
    """ Helper fcn: take a simulator (ground truth ledger, sally) and an estimate (resp), extract confusion table  """

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
        if ow[0] == u - 1 or ow[0] == u - 2:
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


def is_sally_suitable(sally):
    """ Check that all possible owners own at least one edge in sally.g """
    owners = []
    for eid in sally.g.red_edges:
        if sally.ownership[eid] not in owners:
            owners += [sally.ownership[eid]]
        if len(owners) == 3:
            break
    return len(owners) == 3


def run_experiment(sim_par, r, l, a, e, b, hr, sm, label):
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
    # FIRST: challenger generates a graph and sends it to player.

    # Extract number of transacting parties.
    u = len(sm)

    # Initialize a simulator
    sally = Simulator(sim_par)

    # Simulate a ledger; do so until a ledger is found in which all parties own at least one edge.
    while sally.halting_run():
        pass
    while not is_sally_suitable(sally):
        sally = Simulator(sim_par)
        while sally.halting_run():
            pass
    h = deepcopy(sally.g)
    # Send h to player (mock)

    # SECOND: Eve gains her own knowledge from running a KYC exchange and tries to find an estimate of ledger history.
    # Edges known by Eve and associated ownership info
    eve_edges = [eid for eid in h.red_edges if sally.ownership[eid[0]] == u - 2]
    eve_ownership = dict()
    for eid in eve_edges:
        eve_ownership[eid] = sally.ownership[eid]
        eve_ownership[eid[0]] = sally.ownership[eid[0]]
        eve_ownership[eid[1]] = sally.ownership[eid[1]]

    # Eve deletes known spurious ring members
    to_del = [fid for fid in h.red_edges if any([eid[1] == fid[1] and fid != eid for eid in eve_ownership])]
    h.del_edge(to_del)

    # Eve weights graph
    ct = 0
    dct = 0
    for sig_node in h.right_nodes:
        ct += 1
        if ct / len(h.right_nodes) > (dct + 1) * 0.099999:
            print("We are " + str(round(100.0*float(ct/len(g.right_nodes)))) + "% done weighting.")
            dct += 1
        ring = [eid for eid in h.red_edges if eid[1] == sig_node]
        ast = {}  # alleged spendtimes
        for eid in ring:
            # eid = ((left_node_id, left_node_tag), (right_node_id, right_node_tag), edge_tag)
            # tags are ages
            age_of_ring_member = eid[2] - eid[0][1] + 1
            assert age_of_ring_member >= sally.minspendtime
            assert age_of_ring_member >= MIN_SPENDTIME
            assert b(age_of_ring_member) > 0.0
            ast.update({eid: age_of_ring_member})

        base_likelihood = 1.0
        for eid in ast:
            base_likelihood = base_likelihood * b(ast[eid])
        for eid in ring:
            h.red_edges[eid] = base_likelihood * a(ast[eid]) / b(ast[eid])

    # Eve finds optimal matching
    x = h.optimize(1)

    # Send h to challenger (mock)

    # THIRD: Challenger judges success.
    results = interpret(sally, x)
    # [P, N, TP, TN, FP, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, ACC, F1, MCC, BM, MK]
    line = ""
    for thing in results:
        line += "," + str(thing)
    line += "\n"
    return line

# PARAMETER SPACE EXPLORATION:
line = "RING_SIZE,CHURN_LENGTH,ALICE_EXP_SPENDTIME,EVE_EXP_SPENDTIME,BOB_EXP_SPENDTIME,ALICE_HASHRATE,EVE_HASHRATE,BOB_HASHRATE,pAA,pAE,pAB,pEA,pEE,pEB,pBA,pBE,pBB,SAMPLE,RUN,P,N,TP,TN,FP,FN,TPR,TNR,PPV,NPV,FNR,FPR,FDR,FOR,TS,ACC,F1,MCC,INF,MRK\n"
with open(FILENAME, "w+") as wf:
    wf.write(line)

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
                            line = str(ring_size) + "," + str(churn_length) + "," + str(
                                alice_exp_spendtime) + "," + str(eve_exp_spendtime) + "," + str(
                                bob_exp_spendtime) + "," + str(hashrate[0]) + "," + str(hashrate[1]) + "," + str(
                                hashrate[2]) + "," + str(TRUE_STOCHASTIC_MATRIX[0][0]) + "," + str(
                                TRUE_STOCHASTIC_MATRIX[0][1]) + "," + str(TRUE_STOCHASTIC_MATRIX[0][2]) + "," + str(TRUE_STOCHASTIC_MATRIX[1][0]) + "," + str(
                                TRUE_STOCHASTIC_MATRIX[1][1]) + "," + str(TRUE_STOCHASTIC_MATRIX[1][2]) + "," + str(
                                TRUE_STOCHASTIC_MATRIX[2][0]) + "," + str(TRUE_STOCHASTIC_MATRIX[2][1])+ "," + str(TRUE_STOCHASTIC_MATRIX[2][2]) + "," + str(
                                c) + "," + str(ct)  # No comma at the end

                            label = str(hash(line))
                            label = label[-8:]

                            sim_par = dict()
                            sim_par.update({'runtime': RUNTIME, 'filename': SIM_FILENAME,
                                                'stochastic matrix': stoch_mat, 'hashrate': sim_hashrate,
                                                'min spendtime': MIN_SPENDTIME, 'spendtimes': sim_spendtimes,
                                                'ring size': ring_size, 'reporting modulus': 1})

                            line += run_experiment(sim_par, ring_size, churn_length, alice_spendtime, eve_spendtime, bob_spendtime, hashrate, stoch_mat, label)
                            with open(FILENAME, "a+") as wf:
                                wf.write(line)

                            ct += 1
                            print("\n" + str(floor(float(ct)/float(tot)*100.0)) + "% of parameter space explored.")

