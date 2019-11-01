import unittest as ut
import itertools
from graphtheory import *
from simulator import *
from copy import deepcopy

FILENAME = "../data/output.txt"
STOCHASTIC_MATRIX = [[0.0, 0.9, 0.1], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]
HASH_RATE_ALICE = 0.33
HASH_RATE_EVE = 0.33
HASH_RATE_BOB = 1.0 - HASH_RATE_ALICE - HASH_RATE_EVE
HASH_RATE = [HASH_RATE_ALICE, HASH_RATE_EVE, HASH_RATE_BOB]
MIN_SPEND_TIME = 10
# Expected spend-times are 20.0, 100.0, and 50.0 blocks, respectively.
SPEND_TIMES = [lambda x: 0.05 * ((1.0 - 0.05) ** (x - MIN_SPEND_TIME)),
               lambda x: 0.01 * ((1.0 - 0.01) ** (x - MIN_SPEND_TIME)),
               lambda x: 0.025 * ((1.0 - 0.025) ** (x - MIN_SPEND_TIME))]
RING_SIZE = 11
RUNTIME = 100
REPORTING_MODULUS = 7  # lol whatever


def make_sally():
    par = dict()
    par['runtime'] = RUNTIME
    par['filename'] = FILENAME
    par['stochastic matrix'] = STOCHASTIC_MATRIX
    par['hashrate'] = HASH_RATE
    par['min spendtime'] = MIN_SPEND_TIME
    par['spendtimes'] = SPEND_TIMES
    par['ring size'] = RING_SIZE
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
        self.assertEqual(sally.stoch_matrix, STOCHASTIC_MATRIX)
        self.assertEqual(sally.hashrate, HASH_RATE)
        x = choice(range(1,9))
        self.assertEqual(sally.spendtimes[0](x), SPEND_TIMES[0](x))
        self.assertEqual(sally.spendtimes[1](x), SPEND_TIMES[1](x))
        self.assertEqual(sally.spendtimes[2](x), SPEND_TIMES[2](x))
        self.assertEqual(sally.ringsize, RING_SIZE)
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

    @staticmethod
    def gather_stats(sally):
        t = sally.t
        num_left_nodes = len(sally.g.left_nodes)
        num_right_nodes = len(sally.g.right_nodes)
        num_red_edges = len(sally.g.red_edges)
        num_blue_edges = len(sally.g.blue_edges)
        temp_to_spend = sorted(sally.buffer[t + 1],
                               key=lambda x: (x[0], x[1]))
        buffer_len = len(temp_to_spend)
        txn_bundles = groupby(temp_to_spend, key=lambda x: (x[0], x[1]))

        return [t, num_left_nodes, num_right_nodes, num_red_edges,
                num_blue_edges, buffer_len, txn_bundles]

    def new_vs_old(self, new_t, old_t, dt, new_num_left_nodes, old_num_left_nodes,
                   left_nodes_to_be_added, new_num_right_nodes,
                   old_num_right_nodes, right_nodes_to_be_added,
                   new_num_red_edges, old_num_red_edges, red_edges_to_be_added,
                   new_num_blue_edges, old_num_blue_edges,
                   blue_edges_to_be_added):
        print("CALLING NEW_VS_OLD:\n")
        print("INPUT = " + str((new_t, old_t, dt, new_num_left_nodes, old_num_left_nodes,
                   left_nodes_to_be_added, new_num_right_nodes,
                   old_num_right_nodes, right_nodes_to_be_added,
                   new_num_red_edges, old_num_red_edges, red_edges_to_be_added,
                   new_num_blue_edges, old_num_blue_edges,
                   blue_edges_to_be_added)))
        self.assertEqual(new_t, old_t + dt)
        self.assertEqual(new_num_left_nodes, old_num_left_nodes +
                         left_nodes_to_be_added)
        self.assertEqual(new_num_right_nodes, old_num_right_nodes +
                         right_nodes_to_be_added)
        self.assertEqual(new_num_red_edges, old_num_red_edges +
                         red_edges_to_be_added)
        self.assertEqual(new_num_blue_edges, old_num_blue_edges +
                         blue_edges_to_be_added)

    def next_timestep(self, dt, old_stats, predictions, sally):
        """ next_timestep(self.gather_stats(sally), predictions, sally)
        """
        # Process old stats
        [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges,
         old_num_blue_edges, old_buffer_len, old_txn_bundles] = old_stats

        # Process prediction
        [dt, left_nodes_to_be_added, right_nodes_to_be_added,
         red_edges_to_be_added, blue_edges_to_be_added] = predictions

        # Simulate the next timestemp
        sally.halting_run()

        # Gather some "new" stats
        result = self.gather_stats(sally)
        [new_t, new_num_left_nodes, new_num_right_nodes,
         new_num_red_edges, new_num_blue_edges, new_buffer_len,
         new_txn_bundles] = result

        # Test new stats against predictions based on old stats
        self.new_vs_old(new_t, old_t, dt, new_num_left_nodes,
                        old_num_left_nodes, left_nodes_to_be_added,
                        new_num_right_nodes, old_num_right_nodes,
                        right_nodes_to_be_added, new_num_red_edges,
                        old_num_red_edges, red_edges_to_be_added,
                        new_num_blue_edges, old_num_blue_edges,
                        blue_edges_to_be_added)
        return result

    def stepwise_predict_and_verify(self, dt, sally, n, old_stats = None):
        if old_stats is None:
            old_stats = self.gather_stats(sally)
        while n > 0:
            n -= 1
            # Make predictions
            right_nodes_to_be_added = old_stats[5]
            num_txn_bundles = sum([1 for k, grp in deepcopy(old_stats[6])])
            num_true_spenders = sum(
                [1 for k, grp in deepcopy(old_stats[6]) for
                 entry in deepcopy(grp)])
            self.assertEqual(num_true_spenders, right_nodes_to_be_added)
            blue_edges_to_be_added = 2 * num_txn_bundles
            left_nodes_to_be_added = 2 * num_txn_bundles + 1
            red_edges_per_sig = max(1, min(old_stats[1], sally.ringsize))
            red_edges_to_be_added = red_edges_per_sig * num_true_spenders
            predictions = [dt, left_nodes_to_be_added,
                           right_nodes_to_be_added,
                           red_edges_to_be_added, blue_edges_to_be_added]

            # Verify predictions and get next stats
            old_stats = self.next_timestep(dt, old_stats, predictions, sally)
        return sally, old_stats

    def test_halting_run(self):
        """ test_halting_run tests step-by-step halting run
        function by manually manipulating the buffer to ensure expected beh-
        avior. TODO: an automated/randomized test that isn't manually tweaked,
        see test_halting_run."""
        # Initialize simulator
        dt = 1
        sally = make_sally()
        old_stats = self.gather_stats(sally)

        # Make predictions
        left_nodes_to_be_added = 1
        right_nodes_to_be_added = 0
        red_edges_to_be_added = 0
        blue_edges_to_be_added = 0
        predictions = [dt, left_nodes_to_be_added, right_nodes_to_be_added,
                       red_edges_to_be_added, blue_edges_to_be_added]

        # Verify predictions and get next stats.
        old_stats = self.next_timestep(dt, old_stats, predictions, sally)

        # Check sally.buffer is all empty except for <= 1 entry (genesis block)
        genesis_block = list(sally.g.left_nodes.keys())[0]
        for entry in sally.buffer:
            self.assertTrue(len(entry) == 0 or (
                        len(entry) == 1 and genesis_block == entry[0][2]))

        sally, old_stats = self.stepwise_predict_and_verify(dt, sally, 12, old_stats)

        # Manually mess with the buffer. Swap next
        # block of planned spends with the first non-empty one we come across.
        offset = 1
        while sally.t + offset < len(sally.buffer) and len(sally.buffer[sally.t]) == 0:
            sally.buffer[sally.t] = sally.buffer[sally.t + offset]
            sally.buffer[sally.t + offset] = []
            offset += 1
        self.assertTrue(sally.t + offset < len(sally.buffer))  # True w high prob
        self.assertTrue(len(sally.buffer[sally.t]) > 0)  # True w high prob

        sally, old_stats = self.stepwise_predict_and_verify(dt, sally, 12)
        
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
        """ test_make_coinbase tests making a coinbase output. TODO: should include a test that the buffer contains no repeats."""
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
            print("x, buffer = " + str((x, sally.buffer[dt])))
            self.assertTrue(any([y[2] == x for y in sally.buffer[dt]]))
            
        # Test no repeats make it into the buffer at any timestep.
        # flat_buffer = set()
        # running_len = 0
        # for entry in sally.buffer:
        #     flat_buffer = flat_buffer.union(set(entry))
        #     running_len += len(entry)
        #  self.assertEqual(running_len, len(list(flat_buffer))) 

    # @ut.skip("Skipping test_spend_from_buffer_one")
    def test_spend_from_buffer_one(self):
        """ test_spend_from_buffer_one tests spending directly from the buffer. """
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

        sally.spnd_from_buffer() 

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

    # @ut.skip("Skipping test_spend_from_buffer_three")
    def test_spend_from_buffer_three(self):
        """ test_spend_from_buffer_three does a similar test to
        test_spend_from_buffer_one but with simulation-generated buffer. """
        sally = make_sally()
        assert sally.runtime > 3*sally.minspendtime
        while(sally.t + 1 < sally.runtime and \
            len(sally.buffer[sally.t + 1]) == 0):
            sally.halting_run()
                        
        if sally.t < sally.runtime:
            self.assertTrue(len(sally.buffer[sally.t+1]) > 0)
            print("Next buffer = " + str(sally.buffer[sally.t+1]))

            # Gather some "old" stats
            [old_t, old_num_left_nodes, old_num_right_nodes,
             old_num_red_edges, old_num_blue_edges, old_buffer_len,
             old_txn_bundles] = self.gather_stats(sally)
            
            # Make some predictions
            right_nodes_to_be_added = old_buffer_len
            num_txn_bundles = sum([1 for k, grp in deepcopy(old_txn_bundles)]) 
            num_true_spenders = sum([1 for k, grp in deepcopy(old_txn_bundles) \
                for entry in deepcopy(grp)]) 

            self.assertEqual(num_true_spenders, right_nodes_to_be_added)

            blue_edges_to_be_added = 2*num_txn_bundles  # Each bundle produces 2 outputs.
            left_nodes_to_be_added = 2*num_txn_bundles + 1 # don't forget the coinbase
            red_edges_per_sig = max(1, min(old_num_left_nodes, sally.ringsize))
            red_edges_to_be_added = red_edges_per_sig*num_true_spenders # Each true spender picks max(0, min(num_left_nodes, ringsize-1)) mix-ins

            # Spend from dat buffer tho
            sally.spnd_from_buffer()

            # Gather some "new" stats
            [new_t, new_num_left_nodes, new_num_right_nodes,
             new_num_red_edges, new_num_blue_edges, new_buffer_len,
             new_txn_bundles] = self.gather_stats(sally)

            # Test new stats against predictions based on old stats
            self.new_vs_old(new_t-1, old_t, new_num_left_nodes,
                            old_num_left_nodes,
                            left_nodes_to_be_added, new_num_right_nodes,
                            old_num_right_nodes, right_nodes_to_be_added,
                            new_num_red_edges, old_num_red_edges,
                            red_edges_to_be_added, new_num_blue_edges,
                            old_num_blue_edges, blue_edges_to_be_added)

    # @ut.skip("Skipping test_spend_from_buffer_two")
    def test_spend_from_buffer_two(self):
        """ test_spend_from_buffer_two does similar to test_spend_from_buffer_one but with simulation-generated buffer."""
        sally = make_sally()
        assert sally.runtime > 3*sally.minspendtime
        while len(sally.buffer[sally.t + 1]) != 0 and sally.t < sally.runtime:
            sally.halting_run()

        if sally.t < sally.runtime:
            l = len(sally.buffer[sally.t+1])
            num_left_nodes = len(sally.g.left_nodes)
            num_right_nodes = len(sally.g.right_nodes)
            num_red_edges = len(sally.g.red_edges)
            num_blue_edges = len(sally.g.blue_edges)
            num_to_be_spent = len(sally.buffer[sally.t+1])
            num_in_tail_buffer = sum([len(x) for x in sally.buffer[sally.t + 2:]])

            sally.spnd_from_buffer()

            self.assertEqual(len(sally.g.left_nodes), num_left_nodes + 2*num_to_be_spent)
            self.assertEqual(len(sally.g.right_nodes), num_right_nodes + num_to_be_spent)
            self.assertEqual(len(sally.g.red_edges), num_red_edges + sally.ringsize*num_to_be_spent)
            self.assertEqual(len(sally.g.blue_edges), num_blue_edges + 2*num_to_be_spent)
            self.assertEqual(sum([len(x) for x in sally.buffer[sally.t+1:]]), num_in_tail_buffer + 1 + 2*num_to_be_spent)
        
    # @ut.skip("Skipping test_pick_coinbase_owner_correctness")
    def test_pick_coinbase_owner_correctness(self):
        """ test_pick_coinbase_owner_correctness : Check that pick_coinbase_owner produces an index suitable for indexing into the stochastic matrix. This test does not check that pick_coinbase_owner is drawing from the stochastic matrix appropriately (see test_pick_coinbase_owner_dist).
        """
        ss = 100
        sally = make_sally()
        for i in range(ss):
            x = sally.pick_coinbase_owner()
            self.assertTrue(x in range(len(sally.stoch_matrix)))

    # @ut.skip("Skipping test_pick_coinbase_owner_dist")
    def test_pick_coinbase_owner_dist(self):
        """ test_pick_coinbase_owner_dist : Check that pick_coinbase_owner is drawing ring members from the distribution given in the hashrate list. Requires statistical testing. TODO: TEST INCOMPLETE.
        """
        pass

    # @ut.skip("Skipping test_pick_coinbase_amt")
    def test_pick_coinbase_amt(self):
        """ test pick_coinbase_amt : Check that pick_coinbase_amt produces coins on schedule. See comments in simulator.py - the way we've coded this, setting sally.t = 17 and skipping sally.2 = 3, 4, 5, ..., 16 will produce the *wrong* coinbase reward... but skipping blocks like this is the only way this formula goes wrong, and that requires having blocks with no block reward, which isn't acceptable, so it shouldn't be a big deal. """
        sally = make_sally()
        self.assertEqual(sally.pick_coinbase_amt(), DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS) 
        sally.t += 1
        self.assertEqual(sally.pick_coinbase_amt(), DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS*(1.0-DECAY_RATIO))
        sally.t += 1
        self.assertEqual(sally.pick_coinbase_amt(), DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS*(1.0-DECAY_RATIO)**2)
    
    # @ut.skip("Skipping test_r_pick_spend_time_correctness")
    def test_r_pick_spend_time_correctness(self):
        """ test_r_pick_spend_time_correctness : Check that pick_spend_time produces a positive integer time. This test does not check that pick_spend_time is drawing from the stochastic matrix appropriately (see test_pick_next_recip_dist).
        """
        ss = 100
        sally = make_sally()
        sally.run()
        for owner in range(len(sally.stoch_matrix)):
            selects = [sally.pick_spend_time(owner) for i in range(ss)]
            for s in selects:
                self.assertTrue(isinstance(s, int))
                self.assertTrue(s >= MIN_SPEND_TIME)

    # @ut.skip("Skipping test_pick_spend_time_dist")
    def test_pick_spend_time_dist(self):
        """ test_pick_spend_time_dist : Check that pick_spend_time is drawing ages from the correct spend time distributions. TODO: TEST INCOMPLETE.
        """
        pass

    # @ut.skip("Skipping test_r_pick_next_recip_correctness")
    def test_r_pick_next_recip_correctness(self):
        """ test_r_pick_next_recip_correctness : Check that pick_next_recip produces an index suitable for indexing into the stochastic matrix. This test does not check that pick_next_recip is drawing from the stochastic matrix appropriately (see test_pick_next_recip_dist).
        """
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
        """ test_pick_next_recip_dist : Check that pick_next_recip is drawing ring members from the distribution given in the stochastic matrix. Requires statistical testing. TODO: TEST INCOMPLETE.
        """
        pass

    # @ut.skip("Skipping test_get_ring_correctness")
    def test_get_ring_correctness(self):
        """ test_get_ring_correctness : Check that get_ring produces a list of correct length with left_node entries, and that input left_node is a ring member. This test does not check that ring selection is drawing from the distribution appropriately (see test_get_ring_dist)
        """
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
        """ test_get_ring_dist : Check that ring selection is drawing ring members from the appropriate distribution. Requires statistical testing. TODO: TEST INCOMPLETE."""
        pass

tests = [TestSimulator, TestSimulation]
for test in tests:
    ut.TextTestRunner(verbosity=2, failfast=True).run(ut.TestLoader().loadTestsFromTestCase(test))
