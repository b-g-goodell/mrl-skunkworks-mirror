import unittest
import random
from graphtheory import *
from simulator import *

class Test_simulator(unittest.TestCase):
    def test_one(self):
        Sally = Simulator()
        self.assertTrue(Sally.runSimulation())
        self.assertTrue(Sally.currentHeight == Sally.T)
        self.assertTrue(len(Sally.G.left) > 1)
        print(Sally._report())

    def test_step(self):
        Sally = Simulator()
        Tau = Sally.T//2
        Sally.runSimulation(Tau)
        h = Sally.currentHeight
        numLeft = len(Sally.G.left)
        numRight = len(Sally.G.right)
        numInEdges = len(Sally.G.in_edges)
        numOutEdges = len(Sally.G.out_edges)
        numToBeSpent = len(Sally.timesUp[h])
        Sally._step()
        self.assertEqual(Sally.currentHeight - h, 1)
        self.assertEqual(len(Sally.G.right) - numRight, numToBeSpent)
        self.assertEqual(len(Sally.G.in_edges) - numInEdges, numToBeSpent*Sally.R)

    def test_spend(self):
        Sally = Simulator()
        Tau = Sally.T//2
        Sally.runSimulation(Tau)
        numRight = len(Sally.G.right)
        numInEdges = len(Sally.G.in_edges)
        numToBeSpent = len(Sally.timesUp[Sally.currentHeight])
        Sally._spend()
        self.assertEqual(len(Sally.G.right) - numRight, numToBeSpent)
        self.assertEqual(len(Sally.G.in_edges) - numInEdges, numToBeSpent*Sally.R)

    def test_sign_on_untouched_output(self):
        Sally = Simulator()
        Tau = Sally.T//2
        Sally.runSimulation(Tau)
        to_be_spent = random.choice(list(Sally.G.left.keys()))
        k = Sally.G.left[to_be_spent]
        while len(k.in_edges) > 0:
            # Find an untoiuched output
            to_be_spent = random.choice(list(Sally.G.left.keys()))
            k = Sally.G.left[to_be_spent]
        numOuts = 3
        numRight = len(Sally.G.right)
        numLeft = len(Sally.G.left)
        numInEdges = len(Sally.G.in_edges)
        numOutEdges = len(Sally.G.out_edges)
        Sally._sign(numOuts, [to_be_spent])
        self.assertEqual(len(Sally.G.right) - numRight, 1)
        self.assertEqual(len(Sally.G.left) - numLeft, 3)
        self.assertEqual(len(Sally.G.in_edges) - numInEdges, Sally.R)
        self.assertEqual(len(Sally.G.out_edges) - numOutEdges, 3)

    def test_get_ring_on_untouched_input(self):
        Sally = Simulator()
        Tau = Sally.T//2
        Sally.runSimulation(Tau)
        to_be_spent = random.choice(list(Sally.G.left.keys()))
        k = Sally.G.left[to_be_spent]
        while len(k.in_edges) > 0:
            # Find an untoiuched output
            to_be_spent = random.choice(list(Sally.G.left.keys()))
            k = Sally.G.left[to_be_spent]
        next_ring = Sally.getRing(to_be_spent)
        self.assertTrue(len(next_ring)==Sally.R)
        for ring_member in next_ring:
            self.assertTrue(ring_member in Sally.G.left)
        

tests = [Test_simulator]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
