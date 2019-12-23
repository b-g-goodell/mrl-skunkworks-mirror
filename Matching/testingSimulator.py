import unittest as ut
from itertools import groupby
from graphtheory import *
from simulator import *
from copy import deepcopy
from random import choice, sample

FILENAME = "data/output.txt"
STOCHASTIC_MATRIX = [[0.0, 0.9, 1.0 - 0.9], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]
HASH_RATE_ALICE = 0.33
HASH_RATE_EVE = 0.33
HASH_RATE_BOB = 1.0 - HASH_RATE_ALICE - HASH_RATE_EVE
HASH_RATE = [HASH_RATE_ALICE, HASH_RATE_EVE, HASH_RATE_BOB]
MIN_SPEND_TIME = 2
# Expected spend-times are 20.0, 100.0, and 50.0 blocks, respectively.
SPEND_TIMES = [lambda x: 0.05 * ((1.0 - 0.05) ** (x - MIN_SPEND_TIME)),
               lambda x: 0.01 * ((1.0 - 0.01) ** (x - MIN_SPEND_TIME)),
               lambda x: 0.025 * ((1.0 - 0.025) ** (x - MIN_SPEND_TIME))]
RING_SIZE = 11
RUNTIME = 25
REPORTING_MODULUS = 7  # lol whatever


def make_sally():
    """ Create a standard Sally the Simulator. """
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


class TestSimulator(ut.TestCase):
    """ TestSimulator tests our simulator """
    # @ut.skip("Skipping test_init")
    def test_init(self):
        """ test_init verifies that a standard simulator has all the corresponding correct data """
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
        """ Helper function that extracts some helpful statistics from the input simulator for use in the next
        block. Note the index shift when computing ring member choices... """
        bndl = []
        l = 0
        if sally.t + 1 < len(sally.buffer):
            bndl = groupby(sally.buffer[sally.t + 1], key = lambda x: (x[0], x[1]))
            l = sum([1 for _ in deepcopy(bndl)])
        rmc = [x for x in sally.g.left_nodes if x[1] + sally.minspendtime <= sally.t]
        return [sally.t, len(sally.g.left_nodes), len(sally.g.right_nodes), len(sally.g.red_edges),
                len(sally.g.blue_edges), l, bndl, rmc]
          

    def new_vs_old(self, new_t, old_t, new_num_left_nodes, old_num_left_nodes,
                   left_nodes_to_be_added, new_num_right_nodes,
                   old_num_right_nodes, right_nodes_to_be_added,
                   new_num_red_edges, old_num_red_edges, red_edges_to_be_added,
                   new_num_blue_edges, old_num_blue_edges,
                   blue_edges_to_be_added):
        """ Takes some new data, some old data, and some predictions, and verifies consistency. """
        
        self.assertEqual(new_t, old_t + 1)
        self.assertEqual(new_num_left_nodes, old_num_left_nodes +
                         left_nodes_to_be_added)
        self.assertEqual(new_num_right_nodes, old_num_right_nodes +
                         right_nodes_to_be_added)
        self.assertEqual(new_num_red_edges, old_num_red_edges +
                         red_edges_to_be_added)
        self.assertEqual(new_num_blue_edges, old_num_blue_edges +
                         blue_edges_to_be_added)
        return True

    def next_timestep(self, predictions, sally):      
        # print("\nNT: Timestep\n")

        # Process old stats
        old_bndl = groupby(sally.buffer[sally.t], key=lambda x: (x[0], x[1]))
        old_r = len(sally.buffer[sally.t + 1])
        old_l = sum([1 for _ in deepcopy(old_bndl)])
        old_rmc = [x for x in sally.g.left_nodes if x[1] + sally.minspendtime <= sally.t]
        old_stats = self.gather_stats(sally)
        [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges,
         old_num_blue_edges, old_buffer_len, old_txn_bundles, old_ring_member_choices] = old_stats

        # print("NT: old num left nodes = " + str(old_stats[1]))
        # print("NT: old num right nodes = " + str(old_stats[2]))
        # print("NT: old num red edges = " + str(old_stats[3]))
        # print("NT: old num blue edges = " + str(old_stats[4]))

        # Generate prediction

        predictions = [1, 0, 0, 0]
        ring_member_choices = [x for x in sally.g.left_nodes if x[1] + sally.minspendtime <= sally.t + 1]
        num_rmc = len(ring_member_choices)
        red_edges_per_sig = min(num_rmc, sally.ringsize)

        if red_edges_per_sig == sally.ringsize:
            right_nodes_remaining = [x for x in sally.buffer[sally.t + 1] if x[2] in ring_member_choices]
            if len(right_nodes_remaining) > 0:
                bndl = groupby(right_nodes_remaining, key=lambda x: (x[0], x[1]))
                num_txns = sum([1 for k, grp in bndl])
                num_new_rights = len(right_nodes_remaining)
                predictions = [1 + 2 * num_txns, num_new_rights, sally.ringsize * num_new_rights, 2 * num_new_rights]

        # print("NT: Predicted num left nodes = " + str(old_stats[1] + predictions[0]))
        # print("NT: Predicted num right nodes = " + str(old_stats[2] + predictions[1]))
        # print("NT: Predicted num red edges = " + str(old_stats[3] + predictions[2]))
        # print("NT: Predicted num blue edges = " + str(old_stats[4] + predictions[3]))

        # Simulate the next timestep; this includes making coinbases.
        sally.halting_run()

        # Gather some "new" stats
        new_stats = self.gather_stats(sally)
        [new_t, new_num_left_nodes, new_num_right_nodes,
         new_num_red_edges, new_num_blue_edges, new_buffer_len,
         new_txn_bundles, new_ring_member_choices] = new_stats

        result = [new_stats[i] - old_stats[i] for i in range(len(new_stats) - 3)]
                  
        # Check predictions
        assert new_stats[0] == old_stats[0] + 1
        
        # print("NT: Predictions = " + str(predictions))
        #
        # print("NT: Observed num left nodes = " + str(new_stats[1]))
        # print("NT: Observed num right nodes = " + str(new_stats[2]))
        # print("NT: Observed num red edges = " + str(new_stats[3]))
        # print("NT: Observed num blue edges = " + str(new_stats[4]))
        #
        # print("new_stats " + str(new_stats))
        # print("old stats " + str(old_stats))
        # print("predictions " + str(predictions))
        # print("delta (new - old) " + str(result))

        assert new_stats[1] == old_stats[1] + predictions[0]
        assert new_stats[2] == old_stats[2] + predictions[1]
        assert new_stats[3] == old_stats[3] + predictions[2]
        assert new_stats[4] == old_stats[4] + predictions[3]

        # Test new stats against predictions based on old stats
        x = self.new_vs_old(new_t, old_t, new_num_left_nodes,
                        old_num_left_nodes, predictions[0],
                        new_num_right_nodes, old_num_right_nodes,
                        predictions[1], new_num_red_edges,
                        old_num_red_edges, predictions[2],
                        new_num_blue_edges, old_num_blue_edges,
                        predictions[3])

        self.assertTrue(x)

        return result

    # @ut.skip("Skipping test_halting_run")
    def test_halting_run(self):
        """ test_halting_run tests step-by-step halting run
        function by manually manipulating the buffer to ensure expected beh-
        avior. TODO: an automated/randomized test that isn't manually tweaked,
        see test_halting_run."""
        
        # Initialize simulator
        sally = make_sally()
        self.assertEqual(sum([len(x) for x in sally.buffer]), 0)

        old_stats = self.gather_stats(sally)
        [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges, old_num_blue_edges, old_buffer_len,
         old_txn_bundles, old_ring_member_choices] = old_stats

        # Make predictions
        predictions = [1, 0, 0, 0]

        # Verify predictions and get next stats.
        result = self.next_timestep(predictions, sally)

        # We've now created the genesis block with the first coinbase output and nothing else.

        new_stats = self.gather_stats(sally)
        [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len,
         new_txn_bundles, new_ring_member_choices] = new_stats

        self.assertEqual(new_num_left_nodes, old_num_left_nodes + predictions[0])
        self.assertEqual(new_num_right_nodes, old_num_right_nodes + predictions[1])
        self.assertEqual(new_num_red_edges, old_num_red_edges + predictions[2])
        self.assertEqual(new_num_blue_edges, old_num_blue_edges + predictions[3])

        old_stats = deepcopy(new_stats)

        # Proceed block by block until the next buffer element is non-empty, the num ring_member_choices sufficient
        # or until runtime elapses
        keep_going = True
        while len(sally.buffer[sally.t + 1]) == 0 and len(new_ring_member_choices) <= sally.ringsize and keep_going:
            # print("  Left nodes before next timestep " + str(list(sally.g.left_nodes.keys())))
            # print("  Right nodes before next timestep " + str(list(sally.g.right_nodes.keys())))
            old_stats = deepcopy(new_stats)
            [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges, old_num_blue_edges, old_buffer_len,
             old_txn_bundles, old_ring_member_choices] = old_stats
             
            keep_going = self.next_timestep(predictions, sally)
            
            # print("  Left nodes after next timestep " + str(list(sally.g.left_nodes.keys())))
            # print("  Right nodes after next timestep " + str(list(sally.g.right_nodes.keys())))
            #
            # print("  Predictions = " + str(predictions))
            new_stats = self.gather_stats(sally)
            [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len, new_txn_bundles, new_ring_member_choices] = new_stats
            self.assertEqual(new_num_left_nodes, old_num_left_nodes + predictions[0])
            self.assertEqual(new_num_right_nodes, old_num_right_nodes + predictions[1])
            self.assertEqual(new_num_red_edges, old_num_red_edges + predictions[2])
            self.assertEqual(new_num_blue_edges, old_num_blue_edges + predictions[3])

        # Repeat all the above until a simulator is found with a non-empty next buffer element before runtime elapses
        while sally.t >= sally.runtime:
            sally = make_sally()
            old_stats = self.gather_stats(sally)
            [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges, old_num_blue_edges, old_buffer_len,
             old_txn_bundles, old_ring_member_choices] = old_stats

            result = self.next_timestep(predictions, sally)

            # We've now created the genesis block with the first coinbase output and nothing else.

            new_stats = self.gather_stats(sally)
            [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len,
             new_txn_bundles, new_ring_member_choices] = new_stats
            self.assertEqual(new_num_left_nodes, old_num_left_nodes + predictions[0])
            self.assertEqual(new_num_right_nodes, old_num_right_nodes + predictions[1])
            self.assertEqual(new_num_red_edges, old_num_red_edges + predictions[2])
            self.assertEqual(new_num_blue_edges, old_num_blue_edges + predictions[3])
            
            old_stats = deepcopy(new_stats)
            
            keep_going = True
            while len(sally.buffer[sally.t + 1]) == 0 and len(new_ring_member_choices) <= sally.ringsize and keep_going:
                old_stats = deepcopy(new_stats)
                keep_going = self.next_timestep(predictions, sally)
                new_stats = self.gather_stats(sally)
                [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len, new_txn_bundles, new_ring_member_choices] = new_stats

            self.assertEqual(new_num_left_nodes, old_num_left_nodes + predictions[0])
            self.assertEqual(new_num_right_nodes, old_num_right_nodes + predictions[1])
            self.assertEqual(new_num_red_edges, old_num_red_edges + predictions[2])
            self.assertEqual(new_num_blue_edges, old_num_blue_edges + predictions[3])                
            
        if sally.t + 1 < len(sally.buffer) and len(sally.buffer[sally.t + 1]) > 0:
            sally.buffer[sally.t + 1] = [sally.buffer[sally.t + 1][0]]
            self.assertEqual(len(sally.buffer[sally.t + 1]), 1)

            ring_member_choices = [x for x in sally.g.left_nodes if x[1] + sally.minspendtime <= sally.t]
            pending_nodes_remaining = [x for x in sally.buffer[sally.t] if x[2] in ring_member_choices]

            left_nodes_to_be_added = 3
            right_nodes_to_be_added = 1
            red_edges_to_be_added = sally.ringsize
            blue_edges_to_be_added = 2
            
            predictions = [left_nodes_to_be_added, right_nodes_to_be_added,
                           red_edges_to_be_added, blue_edges_to_be_added]

            result = self.next_timestep(predictions, sally)
        else:
            result = None

        return result
        
    # @ut.skip("Skipping test_run")
    def test_run(self):
        """ Tests the whole run() function. Not much to test, since the outcomes are random.
        TODO: Improve test_run by checking that the correct number of blocks have been accounted for, etc."""
        sally = make_sally()  
        sally.run()
        self.assertTrue(len(sally.g.left_nodes) >= sally.runtime)  
        self.assertTrue(len(sally.g.right_nodes) >= 0)
        self.assertLessEqual(len(sally.g.red_edges), sally.ringsize*len(sally.g.right_nodes))
        self.assertEqual(len(sally.g.blue_edges), 2*len(sally.g.right_nodes))

    # @ut.skip("Skipping test_make_coinbase")
    def test_make_coinbase(self):
        """ test_make_coinbase tests making a coinbase output.
        TODO: should include a test that the buffer contains no repeats."""
        # print("Beginning test_make_coinbase")
        sally = make_sally()
        self.assertEqual(len(sally.g.left_nodes), 0)
        self.assertEqual(len(sally.g.right_nodes), 0)
        self.assertEqual(len(sally.g.red_edges), 0)
        self.assertEqual(len(sally.g.blue_edges), 0)
        predicted_amt = sally.next_mining_reward
        # print("Making first coinbase")
        x, dt = sally.make_coinbase()
        self.assertEqual(len(sally.g.left_nodes), 1)
        self.assertEqual(len(sally.g.right_nodes), 0)
        self.assertEqual(len(sally.g.red_edges), 0)
        self.assertEqual(len(sally.g.blue_edges), 0)
        self.assertTrue(x in sally.g.left_nodes)
        self.assertTrue(0 < dt)
        self.assertEqual(sally.amounts[x], predicted_amt)
        self.assertEqual(sally.next_mining_reward, (1.0 - EMISSION_RATIO)*predicted_amt)
        # print("\n\nTESTMAKECOINBASE\n\n", x, sally.ownership[x])
        self.assertTrue(sally.ownership[x] in range(len(sally.stoch_matrix)))
        if dt < sally.runtime:
            # print("Result from make_coinbase x = " + str(x))
            # print("State of buffer =" + str(sally.buffer[sally.t]))
            self.assertTrue(any([y[2] == x for y in sally.buffer[dt]]))
            
        # Test no repeats make it into the buffer at any timestep.
        # flat_buffer = set()
        # running_len = 0
        # for entry in sally.buffer:
        #     flat_buffer = flat_buffer.union(set(entry))
        #     running_len += len(entry)
        #  self.assertEqual(running_len, len(list(flat_buffer))) 

    def look_for_dupes(self, sally):
        """ Returns True if duplicates are in the buffer, False otherwise. """
        whole_buffer_list = [x for entry in sally.buffer for x in entry]
        whole_buffer_set = list(set(whole_buffer_list))
        return len(whole_buffer_list) != len(whole_buffer_set)

    # @ut.skip("Skipping test_spend_from_buffer_one")
    def test_spend_from_buffer_one(self):
        """
        test_spend_from_buffer_one

        Tests spending directly from the buffer. Does so in three major steps.
            FIRST:  Roll a fresh simulator and simulate a ledger for >= 2/3 of its runtime.
            SECOND: Wipe the upcoming buffer element, replace with a single randomly selected output.
            THIRD:  Spend from buffer and verify we gain 2 (respectively, 1, eff_rs, 2) new left nodes (respectively, right nodes, red edges,
                    blue edges).

        """
        # FIRST =====
        # Roll a fresh simulator.
        sally = make_sally()
        assert sally.runtime > 3 * sally.minspendtime  # unnecessary

        # Execute halting_run for a few iterations
        while sally.t <= sally.minspendtime:
            old_stats = self.gather_stats(sally)
            [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges, old_num_blue_edges, old_buffer_len,
             old_txn_bundles, old_ring_member_choices] = old_stats

            sally.halting_run()

            new_stats = self.gather_stats(sally)
            [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len,
             new_txn_bundles, new_ring_member_choices] = new_stats

        # Verify buffer has no duplicates
        self.assertFalse(self.look_for_dupes(sally))

        # SECOND =====

        # Pick a random unspent left node to spend, look up ownership, use that to generate a new recipient.
        unspent_nodes = [x for x in sally.g.left_nodes if not any([x == y[2] for buff in sally.buffer for y in buff])]
        
        while len(unspent_nodes) == 0:
            sally = make_sally()
            while sally.t < 2*sally.minspendtime:
                old_stats = self.gather_stats(sally)
                [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges, old_num_blue_edges, old_buffer_len,
                 old_txn_bundles, old_ring_member_choices] = old_stats

                sally.halting_run()

                new_stats = self.gather_stats(sally)
                [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len,
                 new_txn_bundles, new_ring_member_choices] = new_stats
            unspent_nodes = [x for x in sally.g.left_nodes if not any([x == y[2] for buff in sally.buffer for y in buff])]
            
        node_to_spend = choice(unspent_nodes)
        owner = sally.ownership[node_to_spend]
        recip = sally.pick_next_recip(owner)

        # Verify the new entry isn't already in the buffer at any index
        self.assertFalse(any([(owner, recip, node_to_spend) in buff for buff in sally.buffer]))
        self.assertFalse(self.look_for_dupes(sally))

        # Reset the buffer index... don't merely append
        sally.buffer[sally.t] = [(owner, recip, node_to_spend)]

        self.assertEqual(len(sally.buffer[sally.t]), 1)
        self.assertIn((owner, recip, node_to_spend), sally.buffer[sally.t])
        self.assertFalse(self.look_for_dupes(sally))

        # Count nodes and edges
        old_stats = self.gather_stats(sally)
        [old_t, old_num_left_nodes, old_num_right_nodes, old_num_red_edges, old_num_blue_edges, old_buffer_len,
         old_txn_bundles, old_ring_member_choices] = old_stats

        eff_rs = min(len(old_ring_member_choices), sally.ringsize)

        self.assertEqual(len(sally.buffer[sally.t]), 1)

        # Make predictions
        
        predictions = [0, 0, 0, 0]

        ring_member_choices = [x for x in sally.g.left_nodes if x[1] + sally.minspendtime <= sally.t]
        num_rmc = len(ring_member_choices)
        red_edges_per_sig = min(num_rmc, sally.ringsize)

        if red_edges_per_sig == sally.ringsize:
            right_nodes_remaining = [x for x in sally.buffer[sally.t] if x[2] in ring_member_choices]
            if len(right_nodes_remaining) > 0:
                bndl = groupby(right_nodes_remaining, key=lambda x: (x[0], x[1]))
                num_txns = sum([1 for k, grp in bndl])
                num_new_rights = len(right_nodes_remaining)
                predictions = [2 * num_txns, num_new_rights, sally.ringsize * num_new_rights, 2 * num_new_rights]

        # THIRD =====
        # Spend from the buffer - since we reset the buffer index, this should add exactly one new right node,
        # two new left nodes, eff_rs new red edges, and 2 new blue edges.

        next_result = sally.spend_from_buffer()
        self.assertEqual(next_result, predictions)

        # We've now created the genesis block with the first coinbase output and nothing else.

        new_stats = self.gather_stats(sally)
        [new_t, new_num_left_nodes, new_num_right_nodes, new_num_red_edges, new_num_blue_edges, new_buffer_len,
         new_txn_bundles, new_ring_member_choices] = new_stats

        self.assertEqual(new_num_left_nodes, old_num_left_nodes + predictions[0])
        self.assertEqual(new_num_right_nodes, old_num_right_nodes + predictions[1])
        self.assertEqual(new_num_red_edges, old_num_red_edges + predictions[2])
        self.assertEqual(new_num_blue_edges, old_num_blue_edges + predictions[3])

    # @ut.skip("Skipping test_spend_from_buffer_two")
    def test_spend_from_buffer_two(self):
        """ test_spend_from_buffer_two does similar to test_spend_from_buffer_one but with simulation-generated buffer."""
        dt = 1
        sally = None  # Clear sally
        sally = make_sally()
        assert sally.runtime > 3*sally.minspendtime
        while len(sally.buffer[sally.t + 1]) != 0 and sally.t < sally.runtime:
            sally.halting_run()

        # Gather some "old" stats
        old_stats = self.gather_stats(sally)
        [old_t, old_num_left_nodes, old_num_right_nodes,
         old_num_red_edges, old_num_blue_edges, old_buffer_len,
         old_txn_bundles, old_ring_member_choices] = old_stats

        # Make some predictions
        right_nodes_to_be_added = old_buffer_len
        num_txn_bundles = sum([1 for k, grp in deepcopy(old_txn_bundles)])

        blue_edges_to_be_added = 2 * right_nodes_to_be_added
        left_nodes_to_be_added = 2 * num_txn_bundles
        red_edges_per_sig = max(1, min(len(old_ring_member_choices), sally.ringsize))
        red_edges_to_be_added = red_edges_per_sig * right_nodes_to_be_added

        # Spend from dat buffer tho
        sally.t += 1
        new_stuff = sally.spend_from_buffer()

        # Gather some "new" stats
        [new_t, new_num_left_nodes, new_num_right_nodes,
         new_num_red_edges, new_num_blue_edges, new_buffer_len,
         new_txn_bundles, new_ring_member_choices] = self.gather_stats(sally)

        # Test new stats against predictions based on old stats
        self.new_vs_old(new_t, old_t, new_num_left_nodes,
                        old_num_left_nodes, left_nodes_to_be_added,
                        new_num_right_nodes, old_num_right_nodes,
                        right_nodes_to_be_added, new_num_red_edges,
                        old_num_red_edges, red_edges_to_be_added,
                        new_num_blue_edges, old_num_blue_edges,
                        blue_edges_to_be_added)

        if sally.t < sally.runtime:
            [old_t, old_num_left_nodes, old_num_right_nodes,
             old_num_red_edges, old_num_blue_edges, old_buffer_len,
             old_txn_bundles, old_ring_member_choices] = [new_t, new_num_left_nodes, new_num_right_nodes,
                                 new_num_red_edges, new_num_blue_edges,
                                 new_buffer_len,
                                 new_txn_bundles, new_ring_member_choices]

            # Make some predictions
            right_nodes_to_be_added = old_buffer_len
            num_txn_bundles = sum([1 for k, grp in deepcopy(old_txn_bundles)])
            num_true_spenders = sum([1 for k, grp in deepcopy(old_txn_bundles) \
                                     for entry in deepcopy(grp)])
            self.assertEqual(num_true_spenders, right_nodes_to_be_added)
            blue_edges_to_be_added = 2 * num_true_spenders
            left_nodes_to_be_added = 2 * num_txn_bundles
            red_edges_per_sig = max(1, min(len(old_ring_member_choices), sally.ringsize))
            red_edges_to_be_added = red_edges_per_sig * num_true_spenders

            sally.t += 1
            new_stuff = sally.spend_from_buffer()

            # Gather some "new" stats
            [new_t, new_num_left_nodes, new_num_right_nodes,
             new_num_red_edges, new_num_blue_edges, new_buffer_len,
             new_txn_bundles, new_ring_member_choices] = self.gather_stats(sally)

            # Test new stats against predictions based on old stats
            self.new_vs_old(new_t, old_t, new_num_left_nodes,
                            old_num_left_nodes, left_nodes_to_be_added,
                            new_num_right_nodes, old_num_right_nodes,
                            right_nodes_to_be_added, new_num_red_edges,
                            old_num_red_edges, red_edges_to_be_added,
                            new_num_blue_edges, old_num_blue_edges,
                            blue_edges_to_be_added)

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
        x = sally.pick_coinbase_amt()
        self.assertEqual(x, EMISSION_RATIO*MAX_MONERO_ATOMIC_UNITS)
        sally.t += 1

        self.assertEqual(sally.pick_coinbase_amt(), EMISSION_RATIO*MAX_MONERO_ATOMIC_UNITS*(1.0-EMISSION_RATIO))
        sally.t += 1
        self.assertEqual(sally.pick_coinbase_amt(), EMISSION_RATIO*MAX_MONERO_ATOMIC_UNITS*(1.0-EMISSION_RATIO)**2)
    
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

        # Gather stats
        result = self.gather_stats(sally)
        [t, num_left_nodes, num_right_nodes,
         num_red_edges, num_blue_edges, buffer_len,
         txn_bundles, ring_member_choices] = result

        spender = choice(ring_member_choices)
        self.assertTrue(spender in sally.g.left_nodes)
        self.assertTrue(spender in ring_member_choices)
        ring = sally.get_ring(spender, ring_member_choices)
        self.assertEqual(len(ring), min(sally.ringsize, len(ring_member_choices)))
        self.assertTrue(spender in ring)
        for elt in ring:
            self.assertTrue(elt in sally.g.left_nodes)

    # @ut.skip("Skipping test_get_ring_dist")
    def test_get_ring_dist(self):
        """ test_get_ring_dist : Check that ring selection is drawing ring members from the appropriate distribution. Requires statistical testing. TODO: TEST INCOMPLETE."""
        pass

ut.TextTestRunner(verbosity=2, failfast=True).run(ut.TestLoader().loadTestsFromTestCase(TestSimulator))
