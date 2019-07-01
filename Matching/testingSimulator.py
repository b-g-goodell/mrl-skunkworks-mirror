import unittest
import random
from graphtheory import *
from simulator import *

class Test_simulator(unittest.TestCase):
    def test_one(self):
        Sally = Simulator()
        rejected = Sally.runSimulation()
        self.assertFalse(rejected)
        self.assertTrue(Sally.currentHeight == Sally.T)
        self.assertTrue(len(Sally.G.left_nodes) > 1)
        #print(Sally._report())

    def test_spend(self):
        Sally = Simulator()
        Tau = Sally.T//2
        Sally.runSimulation(Tau)
        numRight = len(Sally.G.right_nodes)
        numInEdges = len(Sally.G.red_edge_weights)
        numToBeSpent = len(Sally.timesUp[Sally.currentHeight])
        Sally._spend()
        self.assertEqual(len(Sally.G.right_nodes) - numRight, numToBeSpent)
        self.assertEqual(len(Sally.G.red_edge_weights) - numInEdges, numToBeSpent*Sally.R)

    def get_untouched_input(self, Sally):
        # Find a provably unspent output
        need_another = True
        touched_nodes = []
        result = None
        count = 0
        while(need_another and len(touched_nodes) < len(Sally.G.left_nodes)):
            count += 1
            # print(count)
            to_be_spent = random.choice(list(Sally.G.left_nodes.keys()))
            while to_be_spent in touched_nodes:
                to_be_spent = random.choice(list(Sally.G.left_nodes.keys()))
            touched_nodes.append(to_be_spent)
            red_edge_found = False
            for eid in Sally.G.red_edge_weights:
                if eid[0]==to_be_spent:
                    red_edge_found = True
                    break
            need_another = red_edge_found
        if not need_another:
            result = to_be_spent
        return result

    def test_sign_on_untouched_output(self):
        ''' Testing sign on untouched input '''
        Sally = Simulator()
        Tau = Sally.T//2
        print("Sally initiated, timespan set")
        to_be_spent = self.get_untouched_input(Sally)
        print("index found!")
        numOuts = 3
        print("number of outs selected")
        numRight = len(Sally.G.right_nodes)
        print("number of right nodes = ", numRight)
        numLeft = len(Sally.G.left_nodes)
        print("number of left nodes = ", numLeft)
        numRedEdges = len(Sally.G.red_edge_weights)
        print("number of red edges = ", numRedEdges)
        numBlueEdges = len(Sally.G.blue_edge_weights)
        print("number of blue edges = ", numBlueEdges)
        print("Calling sign")
        Sally._sign(numOuts, [to_be_spent])
        print("Making assertions")
        self.assertEqual(len(Sally.G.right_nodes) - numRight, 1)
        self.assertEqual(len(Sally.G.left_nodes) - numLeft, 3)
        self.assertEqual(len(Sally.G.red_edge_weights) - numInEdges, Sally.R)
        self.assertEqual(len(Sally.G.blue_edge_weights) - numOutEdges, 3)

    def test_get_ring_on_untouched_input(self):
        ct = 0
        while(ct < 1000):
            print("Initializing Sally ct = ", ct)
            Sally = Simulator()
            Tau = Sally.T//2
            print("Running Sally")
            Sally.runSimulation(Tau)
            print("Picking a random output from Sally")
            to_be_spent = self.get_untouched_input(Sally)
            print("Selecting a ring")
            # Get a ring
            next_ring = Sally.getRing(to_be_spent)
            print("Sally.G.left_nodes = ", Sally.G.left_nodes)
            print("next_ring = ", next_ring)
            self.assertTrue(len(next_ring) >=1 and len(next_ring) <= Sally.R and to_be_spent in next_ring)
            # Verify correct size and each ring member is in Sally.G.left_nodes
            print("Verifying necessary key is inside.")
            self.assertTrue(to_be_spent in next_ring)
            print("Verifying size is appropriate")
            self.assertTrue(len(next_ring) <= Sally.R)
            print("Verifying all ring members are in Sally.G.left_nodes")
            for ring_member in next_ring:
                self.assertTrue(ring_member in Sally.G.left_nodes)
            print("Incrementing counter before starting again with a new sally.")
            ct += 1
        print("We are all out of sally.")
        

tests = [Test_simulator]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
