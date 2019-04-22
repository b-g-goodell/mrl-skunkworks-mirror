from graphtheory import *
from simulator import *

class Experimenter(object):
    def __init__(self, params=None):
        self.params = params

    def runExperiment(self):
        Sally = Simulator()
        rejected = Sally.runSimulation()
        assert not rejected
        G = Sally.G
        trueEdges = Sally.trueEdges # list of edge identities
        G = self.weightGraph(G)
        results = G.opt_matching()

        # False positives:
        fp = [eid for eid in results if eid not in trueEdges]

        # False negatives:
        fn = [eid for eid in trueEdges if eid not in results]

        # True positives:
        tp = [eid for eid in results if eid in trueEdges]

        # True negatives:
        tn = [eid for eid in G.in_edges if eid not in trueEdges]

        print("Total number of edges = ", str(len(fp)+len(fn)+len(tp)+len(tn)))
        print("Number of false positives = ", str(len(fp)))
        print("Number of false negatives = ", str(len(fn)))
        print("Number of true positives = ", str(len(tp)))
        print("Number of true negatives = ", str(len(tn)))
        pwr = float(len(tn))/float(len(tn)+len(fp))
        print("Approximate power = ", str(pwr))
        

    def weightGraph(self, G):
        return G

    def report(self):
        pass

Ed = Experimenter()
Ed.runExperiment()
