from math import *
from graphtheory import *
from collections import deque
from random import *
from copy import *


def generate_age(foo):
    u = random()
    assert 0.0 <= u < 1.0
    x = 0
    u = u - foo(x)
    # print(foo)
    # print(foo(0))
    while u > 0.0:
        x += 1
        u = u - foo(x)
    #print(x)
    return x


def generate_graph_for_first_experiment(par=None):
    # In this experiment we generate a tree of txns in the following way.
    # The root node has age zero.
    # Each node with depth < par["depth"] has par["ring size"] children, one of whom has an age randomly selected
    # from PMF par["alice spend time"] and the rest of which have an age randomly from PMF par["bob spend time"].
    #
    # There exists a unique path of nodes from root to leaf whose ages are all chosen by par["alice spend time"].
    #
    # This experiment encodes the above tree as a bipartite graph by adding each non-leaf node to both the left- and
    # right-node list of G (assuming a 1:1 output/input ratio for all transactions) and assigning red edges by ring
    # member.
    #
    # Eve applies the matching algorithm using edges determined by log likelihood functions and her success is
    # compared to the ground truth.
    #
    # This experiment is bad at approximating the Monero blockchain with too great a depth (ie depth > 3)

    if "depth" not in par:
        par.update({"depth": 3})
    if "ring size" not in par:
        par.update({"ring size": 3})
    if "lock time" not in par:
        par.update({"lock time": 15})

    if "alice spend param" not in par:
        par.update({"alice spend param": 0.1})
    if "alice offset" not in par:
        par.update({"alice offset": par["locktime"]})
    par.update({"alice spend time": lambda x: par["alice spend param"]*((1.0 - par["alice spend param"])**(x-1))})

    if "bob spend param" not in par:
        par.update({"bob spend param": 0.25})
    if "bob offset" not in par:
        par.update({"bob offset": par["locktime"]})
    par.update({"bob spend time": lambda x: par["bob spend param"]*((1.0 - par["bob spend param"])**(x-1))})

    g = BipartiteGraph()
    g.data = {"ages": {}, "depths": {}, "true edges": [], "alice spend time": par["alice spend time"]}
    g.data.update({"alice offset": par["alice offset"], "bob spend time": par["bob spend time"]})
    g.data.update({"bob offset": par["bob offset"], "lock time": par["lock time"]})

    q = deque()

    nid = g.add_node(0)
    q.append((0, 0, nid))  # output/key node from final signature in the tree, has depth 0 and age 0.
    g.data["ages"].update({nid: 0})
    g.data["depths"].update({nid: 0})

    while len(q) > 0:
        (d, a, nid) = q.popleft()  # take the latest depth and output/key node added to the queue.
        if d <= par["depth"]:
            sid = g.add_node(1)  # if it's not too deep, add the signature node and create ring members
            g.add_edge(0, (nid, sid), 0.0)  # add blue edge

            sig_idx = randint(0, par["ring size"]-1)  # pick the true signing index
            for mem_idx in range(par["ring size"]):
                mid = g.add_node(0)  # add the output/key node of the true signer
                g.data["depths"].update({mid: d+1})
                eid = (mid, sid)
                g.add_edge(1, eid, 0.0)  # add red edge from temp to idx
                if mem_idx == sig_idx:
                    age = generate_age(par["alice spend time"])+par["alice offset"]  # pick its age
                    g.data["true edges"] += [eid]
                else:
                    age = generate_age(par["bob spend time"])+par["bob offset"]  # pick its age
                g.data["ages"].update({mid: age})
                q.append((d+1, age, mid))
    return g


def assign_weights_first_experiment(g=BipartiteGraph()):
    h = deepcopy(g)
    for sid in g.right_nodes:
        temp = [eid for eid in g.red_edges if eid[1]==sid]
        tot = sum([log(g.data["bob spend time"](g.data["ages"][eid[0]])) for eid in temp])
        for eid in temp:
            w = deepcopy(tot)
            w += log(g.data["alice spend time"](g.data["ages"][eid[0]]-g.data["lock time"]))
            w -= log(g.data["bob spend time"](g.data["ages"][eid[0]] - g.data["lock time"]))
            h.red_edges[eid] = deepcopy(w)
    return h


def run_first_experiment(d, r):
    par = {"depth": d, "ring size": r, "lock time": 10, "alice spend time": lambda x: exp(-1.0*x), "alice offset": 10,
           "bob spend time": lambda x: 2.0*exp(-2.0*x), "bob offset": 10}
    g = None
    print("Generating graph")
    g = generate_graph_for_first_experiment(par)
    print("Assigning weights")
    g = assign_weights_first_experiment(g)
    s = "============\ndepth = " + str(d) + "\n"
    s += "ring size = " + str(r) + "\n"

    b = 1
    ct = 0
    input_match = []
    result = g.extend_match(b, input_match)
    s += "beginning match extension\n"
    while len(result) > len(input_match):
        ct += 1
        input_match = deepcopy(result)
        result = g.extend_match(b, input_match)
        # print(result)
        # print(2*ct)
    s += "maximal match found, proceeding with boosts\n"
    old_result = deepcopy(result)
    next_result = g.boost_match(b, result)
    s += "first boosted match found\n"
    ct = 0
    print("Finding boosted matches...")
    while next_result != old_result:
        ct += 1
        print(ct)
        old_result = deepcopy(next_result)
        next_result = g.boost_match(b, old_result)
        s += "old result without any of next result:\n"
        s += str([x for x in old_result if x not in next_result])
        s += "\nnew result without any of old result:\n"
        s += str([x for x in next_result if x not in old_result])
    print("Getting stats...")
    true_edges = g.data["true edges"]
    true_pos = [eid for eid in g.red_edges if eid in true_edges and eid in next_result]
    true_neg = [eid for eid in g.red_edges if eid not in true_edges and eid not in next_result]
    false_pos = [eid for eid in g.red_edges if eid not in true_edges and eid in next_result]
    false_neg = [eid for eid in g.red_edges if eid in true_edges and eid not in next_result]
    s += "\n======\n======\nCONFUSION TABLE\n"
    s += "            \t             \t Ground Truth \n"
    s += "            \t             \t True Edge    \t Mix-in Edge\n"
    s += "Estimated   \t True Edge   \t " + str(len(true_pos)) + "\t\t" + str(len(false_pos)) + "\n"
    s += "            \t Mix-in Edge \t " + str(len(false_neg)) + "\t\t" + str(len(true_neg)) + "\n"

    total_population = len(true_pos) + len(false_pos) + len(true_neg) + len(false_neg)
    assert total_population != 0
    prevalence = (len(true_pos) + len(false_pos))/total_population
    accuracy = (len(true_pos) + len(true_neg))/total_population

    if len(true_pos) + len(false_pos) != 0:
        true_positive_rate = len(true_pos) / (len(true_pos) + len(false_pos))
        false_negative_rate = len(false_neg)/(len(true_pos) + len(false_pos))
        pos_pred_val = len(true_pos) / (len(true_pos) + len(false_pos))
        false_disco_rate = len(false_pos) / (len(true_pos) + len(false_pos))
    else:
        true_positive_rate = None
        false_negative_rate = None
        pos_pred_val = None
        false_disco_rate = None

    if len(true_neg)+len(false_neg) != 0:
        false_positive_rate = len(false_pos)/(len(true_neg) + len(false_neg))
        true_negative_rate = len(true_neg)/(len(true_neg) + len(false_neg))
        false_omission_rate = len(false_neg) / (len(true_neg) + len(false_neg))
        neg_pred_val = len(true_neg) / (len(true_neg) + len(false_neg))
    else:
        false_positive_rate = None
        true_negative_rate = None
        false_omission_rate = None
        neg_pred_val = None

    pos_likelihood_ratio = None
    neg_likelihood_ratio = None
    if false_positive_rate is not None and false_positive_rate != 0:
        pos_likelihood_ratio = true_positive_rate/false_positive_rate
    if true_negative_rate is not None and true_negative_rate != 0:
        neg_likelihood_ratio = false_negative_rate/true_negative_rate

    diagnostic_odds_ratio = None
    if neg_likelihood_ratio != 0 and neg_likelihood_ratio is not None:
        diagnostic_odds_ratio = pos_likelihood_ratio/neg_likelihood_ratio
    f_one_score = None
    if pos_pred_val is not None and true_positive_rate is not None and pos_pred_val + true_positive_rate != 0:
        f_one_score = 2.0*pos_pred_val*true_positive_rate/(pos_pred_val+true_positive_rate)
    mcc_score = None
    if len(true_pos)+len(false_pos) != 0 and len(true_pos)+len(false_neg) != 0:
        if len(true_neg)+len(false_pos) != 0 and len(true_neg)+len(false_neg) != 0:
            mcc_score = len(true_pos)*len(true_neg)
            mcc_score = mcc_score - len(false_pos)*len(false_neg)
            mcc_score = mcc_score/sqrt(len(true_pos)+len(false_pos))
            mcc_score = mcc_score/sqrt(len(true_pos)+len(false_neg))
            mcc_score = mcc_score/sqrt(len(true_neg)+len(false_pos))
            mcc_score = mcc_score/sqrt(len(true_neg)+len(false_neg))

    s += "======\n======\nCONFUSION STATS\n"
    s += "Prevalence: " + str(prevalence) + "\n"
    s += "Accuracy: " + str(accuracy) + "\n"
    s += "True Positive Rate (Recall): " + str(true_positive_rate) + "\n"
    s += "False Positive Rate (Fallout): " + str(false_positive_rate) + "\n"
    s += "False Negative Rate (Miss): " + str(false_negative_rate) + "\n"
    s += "True Negative Rate (Specificity): " + str(true_negative_rate) + "\n"
    s += "Positive predictive value (Precision): " + str(pos_pred_val) + "\n"
    s += "False discovery rate: " + str(false_disco_rate) + "\n"
    s += "False omission rate: " + str(false_omission_rate) + "\n"
    s += "Negative predictive value: " + str(neg_pred_val) + "\n"
    s += "Positive likelihood ratio: " + str(pos_likelihood_ratio) + "\n"
    s += "Diagnostic odds ratio: " + str(diagnostic_odds_ratio) + "\n"
    s += "F1 score: " + str(f_one_score) + "\n"
    s += "mcc score: " + str(mcc_score) + "\n"
    s += "======\n======\n"
    s += "True edges " +  str(g.data["true edges"]) + "\n"
    s += "Max likelihood estimate of true edges " +  str(next_result) + "\n"

    with open("output.txt", "a") as wf:
        wf.write(s)
    return f_one_score

with open("output.txt", "w") as wf:
    wf.write("")
results = {}
sample_size = 25
for i in range(3, 6):
    for j in range(3, 7):
        results.update({(i,j):[]})
        print("Running experiment with depth " + str(i) + " and ring size " + str(j))
        for k in range(sample_size):
            if k%10 == 0:
                print("Entering simulation " + str(k))
            results[(i,j)] += [run_first_experiment(i, j)]

with open("output.txt", "a") as wf:
    wf.write(str(results))
