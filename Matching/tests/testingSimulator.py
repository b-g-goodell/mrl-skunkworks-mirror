import unittest as ut
import random
from graphtheory import *
from simulator import *
from copy import deepcopy

FILENAME = "../data/output.txt"
STOCHMAT = [[0.0, 0.9, 0.1], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]
HASHRATE_ALICE = 0.33
HASHRATE_EVE = 0.33
HASHRATE_BOB = 1.0 - HASHRATE_ALICE - HASHRATE_EVE
HASHRATE = [HASHRATE_ALICE, HASHRATE_EVE, HASHRATE_BOB]
MINSPENDTIME = 10
# Expected spend-times are 20.0, 100.0, and 50.0 blocks, respectively.
SPENDTIMES = [lambda x:0.05*((1.0-0.05)**(x-MINSPENDTIME)), \
    lambda x: 0.01*((1.0-0.01)**(x-MINSPENDTIME)), \
    lambda x: 0.025*((1.0-0.025)**(x-MINSPENDTIME))]
RINGSIZE = 11
RUNTIME = 100
REPORTING_MODULUS = 7  # lol whatever

def make_sally():
    par = dict()
    par['runtime'] = RUNTIME
    par['filename'] = FILENAME
    par['stochastic matrix'] = STOCHMAT
    par['hashrate'] = HASHRATE
    par['min spendtime'] = MINSPENDTIME
    par['spendtimes'] = SPENDTIMES
    par['ring size'] = RINGSIZE
    par['reporting modulus'] = REPORTING_MODULUS
    sally = Simulator(par)
    return sally

class TestSimulation(ut.TestCase):
    # @ut.skip("Skipping test_simulation")
    def test_simulation(self):
        sally = make_sally()
        sally.run()

class TestSimulator(ut.TestCase):
    """ TestSimulator tests our simulator """
    # @ut.skip("Skipping test_init")
    def test_init(self):
        sally = make_sally()
        self.assertEqual(sally.runtime, RUNTIME)
        self.assertEqual(sally.fn, FILENAME)
        self.assertEqual(sally.stoch_matrix, STOCHMAT)
        self.assertEqual(sally.hashrate, HASHRATE)
        x = choice(range(1,9))
        self.assertEqual(sally.spendtimes[0](x), SPENDTIMES[0](x))
        self.assertEqual(sally.spendtimes[1](x), SPENDTIMES[1](x))
        self.assertEqual(sally.spendtimes[2](x), SPENDTIMES[2](x))
        self.assertEqual(sally.ringsize, RINGSIZE)
        self.assertEqual(sally.mode, "uniform")
        self.assertEqual(sally.buffer, [[]]*sally.runtime)
        self.assertEqual(sally.ownership, dict())
        self.assertEqual(sally.amounts, dict())
        self.assertTrue(isinstance(sally.g, BipartiteGraph))
        self.assertEqual(len(sally.g.left_nodes), 0)
        self.assertEqual(len(sally.g.right_nodes), 0)
        self.assertEqual(len(sally.g.red_edges), 0)
        self.assertEqual(len(sally.g.blue_edges), 0)
        self.assertEqual(sally.t, 0)
        self.assertEqual(sally.dummy_monero_mining_flat, False)
        
    # @ut.skip("Skipping test_run")
    def test_run(self):
        sally = make_sally()  
        sally.run()
        self.assertTrue(len(sally.g.left_nodes) >= sally.runtime)  
        self.assertTrue(len(sally.g.right_nodes) >= 0)
        self.assertLessEqual(len(sally.g.red_edges), sally.ringsize*len(sally.g.right_nodes))
        self.assertEqual(len(sally.g.blue_edges), 2*len(sally.g.right_nodes))

    # @ut.skip("Skipping test_make_coinbase")
    def test_make_coinbase(self):
        sally = make_sally()
        self.assertEqual(len(sally.g.left_nodes), 0)
        self.assertEqual(len(sally.g.right_nodes), 0)
        self.assertEqual(len(sally.g.red_edges), 0)
        self.assertEqual(len(sally.g.blue_edges), 0)
        x, dt = sally.make_coinbase()
        self.assertEqual(len(sally.g.left_nodes), 1)
        self.assertEqual(len(sally.g.right_nodes), 0)
        self.assertEqual(len(sally.g.red_edges), 0)
        self.assertEqual(len(sally.g.blue_edges), 0)
        self.assertTrue(x in sally.g.left_nodes)
        self.assertTrue(0 < dt)
        self.assertEqual(sally.amounts[x], sally.pick_coinbase_amt())
        # print("\n\nTESTMAKECOINBASE\n\n", x, sally.ownership[x])
        self.assertTrue(sally.ownership[x] in range(len(sally.stoch_matrix)))
        if dt < sally.runtime:
            self.assertTrue(len(sally.buffer[dt]) > 0) 
        # Test no repeats make it into the buffer.
        for entry in sally.buffer:
            self.assertEqual(len(list(set(entry))), len(entry))

    # @ut.skip("Skipping test_spend_from_buffer")
    def test_spend_from_buffer(self):
        sally = make_sally()
        # mimic run() for a few iterations without spending from the buffer (forgetting any txns incidentally placed in buffer this early, nbd for the test)
        assert sally.runtime > 3*sally.minspendtime
        while sally.t < 2*sally.minspendtime:
            num_left_nodes = len(sally.g.left_nodes)
            num_right_nodes = len(sally.g.right_nodes)
            num_red_edges = len(sally.g.red_edges)
            num_blue_edges = len(sally.g.blue_edges)
            new_coinbase_node, spend_time = sally.make_coinbase()
            assert sally.amounts[new_coinbase_node] > 0.0
            self.assertEqual(len(sally.g.left_nodes), num_left_nodes + 1)
            self.assertEqual(len(sally.g.right_nodes), num_right_nodes)
            self.assertEqual(len(sally.g.red_edges), num_red_edges)
            self.assertEqual(len(sally.g.blue_edges), num_blue_edges)
            if sally.t % sally.reporting_modulus == 0:
                sally.report()
            sally.t += 1

        # Test a simple buffer element:
        # Pick a random left node to spend, look up it's ownership, use that to generate a new recipient.
        # Ignore anything else placed in the buffer here so far.
        node_to_spend = choice(list(sally.g.left_nodes.keys()))
        owner = sally.ownership[node_to_spend]
        recip = sally.pick_next_recip(owner)
        sally.buffer[sally.t] = [(owner, recip, node_to_spend)] 

        # Count nodes and edges
        num_left_nodes = len(sally.g.left_nodes)
        num_right_nodes = len(sally.g.right_nodes)
        num_red_edges = len(sally.g.red_edges)
        num_blue_edges = len(sally.g.blue_edges)

        # Should add exactly one new right_node, two new left nodes, two new blue edges, and a variable number of red edges.
        # We can only expect max(len(sally.g.left_nodes), sally.ringsize) red edges, since a ring in our simulations has no repeated members.
        eff_rs = min(len(sally.g.left_nodes), sally.ringsize)

        sally.spend_from_buffer() 

        # Test no repeats make it into the buffer.
        for entry in sally.buffer:
            set_entry = set(entry)
            len_entry = len(entry)
            len_set_entry = len(set_entry)
            self.assertEqual(len_entry, len_set_entry)

        self.assertEqual(len(sally.g.left_nodes), num_left_nodes + 2)
        self.assertEqual(len(sally.g.right_nodes), num_right_nodes + 1)
        self.assertEqual(len(sally.g.red_edges), num_red_edges + eff_rs)
        self.assertEqual(len(sally.g.blue_edges), num_blue_edges + 2)
        
    # @ut.skip("Skipping test_pick_coinbase_owner_correctness")
    def test_pick_coinbase_owner_correctness(self):
        ''' test_pick_coinbase_owner_correctness : Check that pick_coinbase_owner produces an index suitable for indexing into the stochastic matrix. This test does not check that pick_coinbase_owner is drawing from the stochastic matrix appropriately (see test_pick_coinbase_owner_dist).
        '''
        ss = 100
        sally = make_sally()
        for i in range(ss):
            x = sally.pick_coinbase_owner()
            self.assertTrue(x in range(len(sally.stoch_matrix)))

    # @ut.skip("Skipping test_pick_coinbase_owner_dist")
    def test_pick_coinbase_owner_dist(self):
        ''' test_pick_coinbase_owner_dist : Check that pick_coinbase_owner is drawing ring members from the distribution given in the hashrate list. Requires statistical testing. TODO: TEST INCOMPLETE.
        '''
        pass

    # @ut.skip("Skipping test_pick_coinbase_amt")
    def test_pick_coinbase_amt(self):
        ''' test pick_coinbase_amt : Check that pick_coinbase_amt produces coins on schedule. See comments in simulator.py - the way we've coded this, setting sally.t = 17 and skipping sally.2 = 3, 4, 5, ..., 16 will produce the *wrong* coinbase reward... but skipping blocks like this is the only way this formula goes wrong, and that requires having blocks with no block reward, which isn't acceptable, so it shouldn't be a big deal. '''
        sally = make_sally()
        self.assertEqual(sally.pick_coinbase_amt(), DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS) 
        sally.t += 1
        self.assertEqual(sally.pick_coinbase_amt(), DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS*(1.0-DECAY_RATIO))
        sally.t += 1
        self.assertEqual(sally.pick_coinbase_amt(), DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS*(1.0-DECAY_RATIO)**2)
    
    # @ut.skip("Skipping test_r_pick_spend_time_correctness")
    def test_r_pick_spend_time_correctness(self):
        ''' test_r_pick_spend_time_correctness : Check that pick_spend_time produces a positive integer time. This test does not check that pick_spend_time is drawing from the stochastic matrix appropriately (see test_pick_next_recip_dist).
        '''
        ss = 100
        sally = make_sally()
        sally.run()
        for owner in range(len(sally.stoch_matrix)):
            selects = [sally.pick_spend_time(owner) for i in range(ss)]
            for s in selects:
                self.assertTrue(isinstance(s, int))
                self.assertTrue(s >= MINSPENDTIME)

    # @ut.skip("Skipping test_pick_spend_time_dist")
    def test_pick_spend_time_dist(self):
        ''' test_pick_spend_time_dist : Check that pick_spend_time is drawing ages from the correct spend time distributions. TODO: TEST INCOMPLETE.
        '''
        pass

    # @ut.skip("Skipping test_r_pick_next_recip_correctness")
    def test_r_pick_next_recip_correctness(self):
        ''' test_r_pick_next_recip_correctness : Check that pick_next_recip produces an index suitable for indexing into the stochastic matrix. This test does not check that pick_next_recip is drawing from the stochastic matrix appropriately (see test_pick_next_recip_dist).
        '''
        ss = 100
        sally = make_sally()
        sally.run()
        for owner in range(len(sally.stoch_matrix)):
            selects = [sally.pick_next_recip(owner) for i in range(ss)]
            for s in selects:
                # print(s)
                self.assertTrue(s in range(len(sally.stoch_matrix)))

    # @ut.skip("Skipping test_pick_next_recip_dist")
    def test_pick_next_recip_dist(self):
        ''' test_pick_next_recip_dist : Check that pick_next_recip is drawing ring members from the distribution given in the stochastic matrix. Requires statistical testing. TODO: TEST INCOMPLETE.
        '''
        pass

    # @ut.skip("Skipping test_get_ring_correctness")
    def test_get_ring_correctness(self):
        ''' test_get_ring_correctness : Check that get_ring produces a list of correct length with left_node entries, and that input left_node is a ring member. This test does not check that ring selection is drawing from the distribution appropriately (see test_get_ring_dist)
        '''
        sally = make_sally()
        sally.run()
        spender = choice(list(sally.g.left_nodes.keys()))
        self.assertTrue(spender in sally.g.left_nodes)
        ring = sally.get_ring(spender)
        self.assertEqual(len(ring), min(sally.ringsize, len(sally.g.left_nodes)))
        self.assertTrue(spender in ring)
        for elt in ring:
            self.assertTrue(elt in sally.g.left_nodes)

    # @ut.skip("Skipping test_get_ring_dist")
    def test_get_ring_dist(self):
        ''' test_get_ring_dist : Check that ring selection is drawing ring members from the appropriate distribution. Requires statistical testing. TODO: TEST INCOMPLETE.'''
        pass

tests = [TestSimulator, TestSimulation]
for test in tests:
    ut.TextTestRunner(verbosity=2, failfast=True).run(ut.TestLoader().loadTestsFromTestCase(test))
