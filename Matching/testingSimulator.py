from copy import deepcopy
from itertools import groupby
from random import random, choice, sample, randrange
from math import pi
from Matching.simulator import make_simulator, EMISSION_RATIO, MAX_ATOMIC_UNITS, MIN_MINING_REWARD, \
    make_simulated_simulator, MIN_SPEND_TIME
import unittest as ut

SAMPLE_SIZE = 1


class TestSimulator(ut.TestCase):
    """ tests for simulator.py """
    # TODO: Missing tests include the following.
    #  (i)  distributional tests: statistical tests that spend-times, recipients, and coinbase owners are all being
    #       drawn from appropriate distributions.
    #  (ii) test_make_simulator and test_make_simulated_simulator

    def gather_stats_from_sally(self, sally):
        # population numbers of nodes and edges
        # Order counts, and is reflected in predicted_gains below
        old_stats = [len(sally.g.left_nodes), len(sally.g.right_nodes), len(sally.g.blue_edges), len(sally.g.red_edges)]

        num_new_rights = len(sally.buffer[sally.t+1])

        num_new_blues = 2*num_new_rights

        upcoming_available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time <= sally.t+1]
        upcoming_eff_ring_size = min(sally.ringsize, len(upcoming_available_ring_members))
        num_new_reds = num_new_rights*upcoming_eff_ring_size

        bndl = groupby(sally.buffer[sally.t + 1], key=lambda entry: (entry[0], entry[1]))
        temp_bndl = deepcopy(bndl)
        num_txns = len([(k, grp) for k, grp in temp_bndl])
        num_new_lefts = 2*num_txns + 1

        self.assertEqual(num_new_rights, len(sally.buffer[sally.t+1]))
        self.assertGreaterEqual(num_txns, 0)
        self.assertTrue(num_new_lefts, 2*num_txns + 1)
        self.assertEqual(num_new_blues, 2*num_new_rights)
        self.assertEqual(num_new_reds, len(sally.buffer[sally.t+1])*upcoming_eff_ring_size)

        auxiliary_data = [upcoming_available_ring_members, upcoming_eff_ring_size, num_txns]

        # some expected deltas for the above population numbers based on the above extra data
        pred_num_new_rights = len(sally.buffer[sally.t+1])
        self.assertGreaterEqual(pred_num_new_rights, num_txns)

        # Order counts and the order of old_stats goes: left, right, blue, red. Same here.
        predicted_gains = [num_new_lefts, num_new_rights, num_new_blues, num_new_reds]

        return old_stats, auxiliary_data, predicted_gains

    def compare_stats_with_predictions(self, sally, old_stats, old_aux, old_pred, new_stats, new_aux, new_pred, out, total_dt):
        self.assertEqual(len(out), total_dt)
        self.assertTrue(all(len(_) == 2 for _ in out))
        self.assertTrue(all(isinstance(_[1], list) for _ in out))
        self.assertTrue(all(_[0] in sally.g.left_nodes for _ in out))
        self.assertEqual(len(new_stats), len(old_stats))
        self.assertEqual(len(new_stats), len(old_pred))
        for _ in range(len(new_stats)):
            self.assertEqual(new_stats[_], old_stats[_] + old_pred[_])
        for _ in out:
            for txn in _[1]:
                [new_rights, new_lefts, rings, new_reds, new_blues] = txn
                for new_right in new_rights:
                    self.assertTrue(new_right in sally.g.right_nodes)
                for new_left in new_lefts:
                    self.assertTrue(new_left in sally.g.left_nodes)
                for R, new_right in zip(rings, new_rights):
                    for ring_member in R:
                        self.assertIn((ring_member, new_right, sally.t), sally.g.red_edges)
                        self.assertIn((ring_member, new_right, sally.t), new_reds)
                for new_left in new_lefts:
                    for new_right in new_rights:
                        self.assertIn((new_left, new_right, sally.t), sally.g.blue_edges)
                        self.assertIn((new_left, new_right, sally.t), new_blues)

    # @ut.skip("Skipping test_init")
    def test_init(self):
        pass

    # @ut.skip("Skipping test_record")
    def test_record(self):
        pass

    # SINGLE USE TESTS FROM EMPTY LEDGER ####

    # @ut.skip("Skipping test_gen_time_step_from_empty")
    def test_gen_time_step_from_empty(self):
        """ Test that default simulator returns timesteps of 1. """
        sally = make_simulator()
        dt = sally.gen_time_step()
        self.assertEqual(dt, 1)

    # @ut.skip("Skipping test_gen_coinbase_owner_from_empty")
    def test_gen_coinbase_owner_from_empty(self):
        """ Test that default simulator generates allowable coinbase owners."""
        # Note: This does not test whether the simulator is producing coinbase owners from the distribution
        # specified by the hashrate PMF. This only tests the validity of outputs; testing the distribution
        # requires some statistical testing, possibly with some formal verification of code.
        sally = make_simulator()
        domain = list(range(len(sally.hashrate)))
        x = sally.gen_coinbase_owner()
        self.assertIn(x, domain)

    # @ut.skip("Skipping test_gen_coinbase_owner_distribution_from_empty")
    def test_gen_coinbase_owner_distribution_from_empty(self):
        """ Test that the default simulator is generating observations of coinbase owners from correct PMF"""
        # TODO: Write test (hint: simple chi-squared test of a large number of observations, uni stats 1 or 2)
        pass

    # @ut.skip("Skipping test_gen_coinbase_amt_from_empty")
    def test_gen_coinbase_amt_from_empty(self):
        """ Test that coinbase amounts are being correctly deterministically generated. """
        # TODO: Ensure these match actual Monero code.
        # TODO: Compute first several steps by hand to compare against some specific floats.
        # TODO: Ensure integer arithmetic for coinbase amounts works exactly as in Monero (below has floating pt errs)
        sally = make_simulator()

        pred_v = EMISSION_RATIO * MAX_ATOMIC_UNITS
        self.assertEqual(sally.next_mining_reward, pred_v)

        v = sally.gen_coinbase_amt()
        # print("Generating genesis cb, got " + str(v))
        self.assertEqual(v, pred_v)
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertEqual(sally.next_mining_reward, pred_v)

        sally.t += 1
        v = sally.gen_coinbase_amt()
        # print("Generated next cb, got " + str(v))
        self.assertEqual(v, pred_v)
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertEqual(sally.next_mining_reward, pred_v)

        sally.t += 1
        v = sally.gen_coinbase_amt()
        # print("Generated third cb, got " + str(v))
        self.assertEqual(v, pred_v)
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertEqual(sally.next_mining_reward, pred_v)

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_empty")
    def test_add_left_node_to_buffer_from_empty(self):
        """
        Test that adding a non-existent left node to the buffer fails, and that adding an extant left node to the
        buffer that has otherwise not yet been added to the buffer succeeds.
        """
        # First, try to call add_left_node_to_buffer directly with a new default empty simulator.
        # We should get a LookupError.
        sally = make_simulator()
        x = random()
        owner = choice(range(len(sally.hashrate)))
        try:
            sally.add_left_node_to_buffer(x, owner)
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # Next, for each possible owner, roll a new default empty simulator, add a node x, then call
        # add_left_node_to_buffer with (x, owner). This should succeed in the sense that add_left_node_to_buffer
        # should return True, not throwing either a Lookup or AttributeError.
        for owner in range(len(sally.hashrate)):
            # For each miner, create a default simulator, add a left node, and then try to add that node to the buffer.
            sally = make_simulator()
            x = sally.g.add_node(0, sally.t)
            try:
                dt = sally.add_left_node_to_buffer(x, owner)
            except AttributeError:
                self.assertTrue(False)
            except LookupError:
                self.assertTrue(False)
            else:
                self.assertIsInstance(dt, int)
                self.assertGreaterEqual(dt, sally.min_spend_time)
                self.assertTrue(sally.t + dt >= sally.runtime or
                                any([_[2] == x for block in sally.buffer for _ in block]))

    # @ut.skip("Skipping test_make_lefts_from_empty")
    def test_make_lefts_from_empty(self):
        """ Tests that making new left nodes in an empty graph succeeds. """
        sally = make_simulator()
        num_parties = len(sally.hashrate)
        del sally
        for sender in range(num_parties):
            for recip in range(num_parties):
                sally = make_simulator()
                old_ct = len(sally.g.left_nodes)
                amt = random()
                x = sally.make_lefts(sender, recip, amt)
                new_ct = len(sally.g.left_nodes)
                self.assertEqual(old_ct + 2, new_ct)

                self.assertEqual(len(x), 2)
                [ch_out, recip_out] = x

                self.assertIn(ch_out, sally.g.left_nodes)
                self.assertIn(ch_out, sally.ownership)
                self.assertIn(ch_out, sally.amounts)
                self.assertIn(sally.ownership[ch_out], range(len(sally.hashrate)))
                self.assertIn(sally.ownership[ch_out], sally.owner_names)

                self.assertIn(recip_out, sally.g.left_nodes)
                self.assertIn(recip_out, sally.ownership)
                self.assertIn(recip_out, sally.amounts)
                self.assertIn(sally.ownership[recip_out], range(len(sally.hashrate)))
                self.assertIn(sally.ownership[recip_out], sally.owner_names)

    # @ut.skip("Skipping test_make_rights_from_empty")
    def test_make_rights_from_empty(self):
        """ Tests that making new rights from an empty graph fails (there are no ring members available). """
        sally = make_simulator()
        k = len(sally.hashrate)

        # Calling with a sender outside of range(k) with anything in the second slot should produce an error.
        try:
            sally.make_rights(pi, [])
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # Calling with a sender in the range and an empty list should work fine, producing out = []
        for owner in range(k):
            sally = make_simulator()
            try:
                out = sally.make_rights(owner, [])
            except AttributeError:
                self.assertTrue(False)
            else:
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), 0)

        # We should get an AttributeError if we call make_rights with any nonempty list, or a non-list
        sally = make_simulator()
        try:
            sally.make_rights(0, dict())
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_reds_from_empty")
    def test_make_reds_from_empty(self):
        """ Tests that making new red edges in an empty graph fails (empty means no nodes!) """
        sally = make_simulator()
        try:
            sally.make_reds([], [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_blues_from_empty")
    def test_make_blues_from_empty(self):
        """ Tests that making new blue edges in an empty graph fails (empty means no nodes!) """
        sally = make_simulator()
        try:
            sally.make_blues([], [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_gen_spend_time_from_empty")
    def test_gen_spend_time_from_empty(self):
        """ Tests that one output of gen_spend_time for each owner is an integer >= sally.min_spend_time. """
        # NOTE: This does not test whether the spend-time distributions are faithful.
        sally = make_simulator()
        k = len(sally.hashrate)
        # spend_times = []
        for owner in range(k):
            sally = make_simulator()
            i = sally.gen_spend_time(owner)
            self.assertIsInstance(i, int)
            self.assertTrue(i >= sally.min_spend_time)

    # @ut.skip("Skipping test_gen_spend_time_distribution_from_empty")
    def test_gen_spend_time_distribution_from_empty(self):
        """ Tests that each owner's spend-time is generating spend-times according to the correct PMF. """
        # TODO: Needs to be written, requires statistical testing. Chi-squared can do it.
        pass

    # @ut.skip("Skipping test_gen_recipient_from_empty")
    def test_gen_recipient_from_empty(self):
        """ Tests that generating a recipient from an empty graph yields an allowable owner. """
        # Note: Does not test that recipient is generated from the distribution correctly, only tests that they land
        # in the allowable owner set.
        sally = make_simulator()
        allowed_idxs = list(sally.owner_names.keys())
        for owner in range(len(sally.stochastic_matrix)):
            sally = make_simulator()
            self.assertIn(sally.gen_recipient(owner), allowed_idxs)

    # @ut.skip("Skipping test_gen_recipient_distribution_from_empty")
    def test_gen_recipient_distribution_from_empty(self):
        """ Tests that generating a recipient from an empty graph uses the correct PMF. """
        # TODO: Requires statistical testing.
        pass

    # @ut.skip("Skipping test_make_coinbase_from_empty")
    def test_make_coinbase_from_empty(self):
        """ Tests that making a coinbase from an empty graph yields a coinbase in the left nodes."""
        sally = make_simulator()
        out = sally.make_coinbase()
        self.assertIn(out, sally.g.left_nodes)

    # @ut.skip("Skipping test_gen_rings_from_empty")
    def test_gen_rings_from_empty(self):
        """ Tests that generating rings from an empty simulator throws an AttributeError."""
        sally = make_simulator()
        try:
            sally.gen_rings([])
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_txn_from_empty")
    def test_make_txn_from_empty(self):
        """ Tests that make_txn with empty simulator throws an Attribute Error since there is nothing to construct."""
        sally = make_simulator()
        owner_names = sally.owner_names
        for sender in owner_names:
            for recip in owner_names:
                sally = make_simulator()
                try:
                    sally.make_txn(sender, recip, [])
                except AttributeError:
                    self.assertTrue(True)

    # @ut.skip("Skipping test_make_txns_from_empty")
    def test_make_txns_from_empty(self):
        """ Tests that make_txns with an empty simulator returns an empty list. """
        # Note: Calling make_txns with an empty simulator simply loops through nothing.
        self.assertEqual(make_simulator().make_txns(), [])

    # @ut.skip("Skipping test_update_state_from_empty")
    def test_update_state_from_empty(self):
        """ Tests that update_state with an empty graph makes a coinbase, moves time forward, and nothing else."""
        # Make a simulator
        sally = make_simulator()

        # Collect "before" stats.
        old_stats, old_aux, old_pred = self.gather_stats_from_sally(sally)
        self.assertEqual(old_pred[0], 1)
        self.assertTrue(all(_ == 0 for _ in old_pred[1:]))
        # Call update_state
        out = sally.update_state(sally.dt)
        # Collect "after" stats
        new_stats, new_aux, new_pred = self.gather_stats_from_sally(sally)
        # Compare
        self.compare_stats_with_predictions(sally, old_stats, old_aux, old_pred, new_stats, new_aux, new_pred, out,
                                            sally.dt)

    # @ut.skip("Skipping test_step_from_empty")
    def test_step_from_empty(self):
        """ Sort of stupid test that mimics test_update_state, since step merely calls this."""
        # Make a simulator
        sally = make_simulator()

        # Collect "before" stats.
        old_t = sally.t
        old_num_left_nodes = len(sally.g.left_nodes)
        old_num_right_nodes = len(sally.g.right_nodes)
        old_num_blue_edges = len(sally.g.blue_edges)
        old_num_red_edges = len(sally.g.red_edges)
        old_available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time <= sally.t+1]
        old_eff_ring_size = min(sally.ringsize, len(old_available_ring_members))

        self.assertEqual(old_t, 0)
        self.assertEqual(old_num_left_nodes, 0)
        self.assertEqual(old_num_right_nodes, 0)
        self.assertEqual(old_num_blue_edges, 0)
        self.assertEqual(old_num_red_edges, 0)
        self.assertEqual(len(old_available_ring_members), 0)
        self.assertEqual(old_eff_ring_size, 0)

        # Make some predictions.
        pred_dt = sally.dt
        pred_num_new_rights = len(sally.buffer[sally.t + 1])
        pred_num_new_lefts = 1 + 2 * pred_num_new_rights
        pred_num_red_edges = old_eff_ring_size * pred_num_new_rights
        pred_num_blue_edges = 2 * pred_num_new_rights

        self.assertEqual(pred_num_new_rights, 0)
        self.assertEqual(pred_num_new_lefts, 1)
        self.assertEqual(pred_num_red_edges, 0)
        self.assertEqual(pred_num_blue_edges, 0)

        # Call update_state
        out = sally.step()

        # Check a few predictions
        self.assertEqual(len(out), sally.dt)
        self.assertEqual(len(out[0]), 2)
        self.assertIsInstance(out[0][1], list)
        self.assertIn(out[0][0], sally.g.left_nodes)
        self.assertEqual(len(out[0][1]), 0)

        # Collect "after" stats
        new_t = sally.t
        new_num_left_nodes = len(sally.g.left_nodes)
        new_num_right_nodes = len(sally.g.right_nodes)
        new_num_blue_edges = len(sally.g.blue_edges)
        new_num_red_edges = len(sally.g.red_edges)
        new_available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time < sally.t]
        new_eff_ring_size = min(sally.ringsize, len(new_available_ring_members))

        self.assertEqual(old_t + pred_dt, new_t)
        self.assertEqual(old_num_left_nodes + pred_num_new_lefts, new_num_left_nodes)
        self.assertEqual(old_num_right_nodes + pred_num_new_rights, new_num_right_nodes)
        self.assertEqual(old_num_blue_edges + pred_num_blue_edges, new_num_blue_edges)
        self.assertEqual(old_num_red_edges + pred_num_red_edges, new_num_red_edges)
        self.assertEqual(len(new_available_ring_members), 0)
        self.assertEqual(new_eff_ring_size, 0)

    # @ut.skip("Skipping test_run_from_empty")
    def test_run_from_empty(self):
        sally = make_simulator()
        try:
            sally.run()
        except AttributeError:
            self.assertTrue(False)
        except RuntimeError:
            self.assertTrue(False)
        else:
            # TODO: open stored data from file and check it's of the correct format?
            self.assertTrue(True)

    # REPETITION OF SINGLE-USE TESTS ####

    # @ut.skip("Skipping test_gen_time_step_from_empty_repeated")
    def test_gen_time_step_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_time_step_from_empty()

    # @ut.skip("Skipping test_gen_coinbase_owner_from_empty_repeated")
    def test_gen_coinbase_owner_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_coinbase_owner_distribution_from_empty_repeated")
    def test_gen_coinbase_owner_distribution_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_coinbase_amt_from_empty_repeated")
    def test_gen_coinbase_amt_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_empty_repeated")
    def test_add_left_node_to_buffer_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_lefts_from_empty_repeated")
    def test_make_lefts_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_rights_from_empty_repeated")
    def test_make_rights_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_reds_from_empty_repeated")
    def test_make_reds_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_blues_from_empty_repeated")
    def test_make_blues_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_spend_time_from_empty_repeated")
    def test_gen_spend_time_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_spend_time_distribution_from_empty_repeated")
    def test_gen_spend_time_distribution_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_recipient_from_empty_repeated")
    def test_gen_recipient_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_recipient_distribution_from_empty_repeated")
    def test_gen_recipient_distribution_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_coinbase_from_empty_repeated")
    def test_make_coinbase_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_rings_from_empty_repeated")
    def test_gen_rings_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_txn_from_empty_repeated")
    def test_make_txn_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_make_txns_from_empty_repeated")
    def test_make_txns_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_update_state_from_empty_repeated")
    def test_update_state_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_step_from_empty_repeated")
    def test_step_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_run_from_empty_repeated")
    def test_run_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # SINGLE-USE TESTS FROM A SIMULATED LEDGER. ####

    # @ut.skip("Skipping test_gen_time_step_from_simulated")
    def test_gen_time_step_from_simulated(self):
        """ Test that default simulator returns timesteps of 1. """
        sally = make_simulated_simulator()
        dt = sally.gen_time_step()
        self.assertEqual(dt, 1)

    # @ut.skip("Skipping test_gen_coinbase_owner_from_simulated")
    def test_gen_coinbase_owner_from_simulated(self):
        """ Test that default simulator generates allowable coinbase owners."""
        # Note: This does not test whether the simulator is producing coinbase owners from the distribution
        # specified by the hashrate PMF. This only tests the validity of outputs; testing the distribution
        # requires some statistical testing, possibly with some formal verification of code.
        sally = make_simulated_simulator()
        domain = list(range(len(sally.stochastic_matrix)))
        x = sally.gen_coinbase_owner()
        self.assertIn(x, domain)

    # @ut.skip("Skipping test_gen_coinbase_amt_from_simulated")
    def test_gen_coinbase_amt_from_simulated(self):
        """ Test that coinbase amounts are being correctly deterministically generated. """
        # TODO: Ensure these match actual Monero code.
        # TODO: Compute first several steps by hand to compare against some specific floats.
        # TODO: Ensure integer arithmetic for coinbase amounts works exactly as in Monero (below has floating pt errs)
        # Generate a new simulator, keep re-rolling if it's too sparse.
        sally = make_simulated_simulator()
        while sally.runtime - sally.t < 10:
            sally = make_simulated_simulator()

        # Make prediction of upcoming reward and verify it
        pred_v = EMISSION_RATIO * MAX_ATOMIC_UNITS * (1.0 - EMISSION_RATIO)**sally.t
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)

        # Generate coinbase (updating next), make prediction of upcoming award, verify it, increment time.
        v = sally.gen_coinbase_amt()
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)
        sally.t += 1

        # Generate coinbase (updating next), make prediction of upcoming award, verify it, increment time.
        v = sally.gen_coinbase_amt()
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)
        sally.t += 1

        # Generate coinbase (updating next), make prediction of upcoming award, verify it.
        v = sally.gen_coinbase_amt()
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_simulated")
    def test_add_left_node_to_buffer_from_simulated(self):
        """
        Test that adding a non-existent left node to the buffer fails, and that adding an extant left node to the
        buffer that has otherwise not yet been added to the buffer succeeds.
        """
        # First, try to call add_left_node_to_buffer directly with a new default empty simulator and random crap.
        # We should get a LookupError.
        sally = make_simulated_simulator()

        x = random()
        owner = choice(range(len(sally.hashrate)))
        try:
            sally.add_left_node_to_buffer(x, owner)
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # Next, for each possible owner, roll a new default simulated simulator, add a node x, then call
        # add_left_node_to_buffer with (x, owner). This should succeed in the sense that add_left_node_to_buffer
        # should return True, not throwing either a Lookup or AttributeError.
        for owner in range(len(sally.hashrate)):
            # For each miner, create a default simulator, add a left node, and then try to add that node to the buffer.
            sally = make_simulated_simulator()
            old_ct = len(sally.g.left_nodes)
            x = sally.g.add_node(0, sally.t)
            self.assertTrue(all([pending_key[2] != x for block in sally.buffer for pending_key in block]))
            try:
                dt = sally.add_left_node_to_buffer(x, owner)
            except AttributeError:
                self.assertTrue(False)
            except LookupError:
                self.assertTrue(False)
            else:
                new_ct = len(sally.g.left_nodes)
                self.assertEqual(old_ct + 1, new_ct)
                self.assertIsInstance(dt, int)
                self.assertGreaterEqual(dt, sally.min_spend_time)
                self.assertTrue(sally.t + dt >= sally.runtime or any([_[2] == x for blk in sally.buffer for _ in blk]))

    # @ut.skip("Skipping test_make_lefts_from_simulated")
    def test_make_lefts_from_simulated(self):
        """ Tests that making new left nodes in a simulated simulator succeeds. """
        sally = make_simulated_simulator()
        num_parties = len(sally.hashrate)
        del sally
        for sender in range(num_parties):
            for recip in range(num_parties):
                # For each possible sender-recipient pair, roll a new simulated simulator and generate a random
                # amount. Call make_lefts.
                sally = make_simulated_simulator()
                old_ct = len(sally.g.left_nodes)

                amt = random()
                x = [sally.make_lefts(sender, recip, amt)]

                new_ct = len(sally.g.left_nodes)
                self.assertEqual(old_ct + 2, new_ct)
                self.assertEqual(len(x[-1]), 2)
                [ch_out, recip_out] = x[-1]
                # Verify output.
                self.assertIn(ch_out, sally.g.left_nodes)
                self.assertIn(ch_out, sally.ownership)
                self.assertIn(ch_out, sally.amounts)
                self.assertIn(sally.ownership[ch_out], range(len(sally.hashrate)))
                self.assertIn(sally.ownership[ch_out], sally.owner_names)
                self.assertIn(recip_out, sally.g.left_nodes)
                self.assertIn(recip_out, sally.ownership)
                self.assertIn(recip_out, sally.amounts)
                self.assertIn(sally.ownership[recip_out], range(len(sally.hashrate)))
                self.assertIn(sally.ownership[recip_out], sally.owner_names)

    # @ut.skip("Skipping test_make_rights_from_simulated_fail")
    def test_make_rights_from_simulated_fail(self):
        """ Tests that making new rights from a sim'd graph succeeds. """
        # TODO: check before and after each move that sally.g gains correct number of nodes and edges
        sally = make_simulated_simulator()
        # Calling with a sender outside of range(k) and anything in the second slot should produce an error.
        try:
            sally.make_rights(pi, [])
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_rights_from_simulated_simple")
    def test_make_rights_from_simulated_simple(self):
        """ Test that making_rights """
        # Calling with a sender in the range and an empty list should work fine, producing out = []
        sally = make_simulator()
        k = len(sally.hashrate)
        del sally

        for owner in range(k):
            sally = make_simulated_simulator()
            try:
                out = sally.make_rights(owner, [])
            except AttributeError:
                self.assertTrue(False)
            else:
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), 0)

        # Calling with a sender in the range and a list with a single available output shoudl work fine,
        for owner in range(k):
            # Roll a new simulator with some available left nodes.
            sally = make_simulated_simulator()
            available_left_nodes = [x for x in sally.g.left_nodes if x[1] + sally.min_spend_time <= sally.t]
            while len(available_left_nodes) == 0:
                sally = make_simulated_simulator()
                available_left_nodes = [x for x in sally.g.left_nodes if x[1] + sally.min_spend_time <= sally.t]

            # Pick an available left node.
            true_spender = choice(available_left_nodes)
            try:
                out = sally.make_rights(owner, [true_spender])
            except AttributeError:
                self.assertTrue(False)
            else:
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), 1)
                for entry in out:
                    self.assertIsInstance(entry, tuple)
                    self.assertEqual(len(entry), 2)
                    self.assertIn(entry, sally.g.right_nodes)

    # @ut.skip("Skipping test_make_rights_from_simulated_complex")
    def test_make_rights_from_simulated_complex(self):
        """ Tests that making new rights from a sim'd graph succeeds. """
        # TODO: check before and after each move that sally.g gains correct number of nodes and edges

        # We should get an AttributeError if we call make_rights with any nonempty list, or a non-list
        sally = make_simulated_simulator()
        try:
            sally.make_rights(0, dict())
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # If we call make_rights with a true_spender list containing any immature keys, we should also
        # get an AttributeError
        sally = make_simulated_simulator()
        immature_left_nodes = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time > sally.t]
        while len(immature_left_nodes) == 0:
            sally = make_simulated_simulator()
            immature_left_nodes = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time > sally.t]
        true_spenders = [choice(immature_left_nodes)]
        try:
            sally.make_rights(0, true_spenders)
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # If we call make_rights with a true_spender list containing available ring members, we should succeed.
        sally = make_simulated_simulator()
        available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time < sally.t]
        while len(available_ring_members) == 0:
            sally = make_simulated_simulator()
            available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time < sally.t]
        true_spenders = [choice(available_ring_members)]
        try:
            sally.make_rights(0, true_spenders)
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    # @ut.skip("Skipping test_make_reds_from_simulated_no_input")
    def test_make_reds_from_simulated_no_input(self):
        """ Tests that making new reds from a sim'd graph without input fails correctly."""
        # Making reds with no input nodes should throw a LookupError
        sally = make_simulated_simulator()
        try:
            out = sally.make_reds([], [])
            self.assertEqual(out, [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_reds_from_simulated_simple")
    def test_make_reds_from_simulated_simple(self):
        """ Tests that making a single new red succeeds."""
        # Roll a new simulator with at least one non-red-edge available.
        sally = make_simulated_simulator()
        self.assertTrue(len(sally.g.right_nodes) > 0)
        self.assertTrue(len(sally.g.left_nodes) > 0)
        not_red_edges = [(x, y) for x in sally.g.left_nodes for y in sally.g.right_nodes if
                         all([_[0] != x or _[1] != y for _ in sally.g.red_edges])]
        while len(not_red_edges) == 0:
            sally = make_simulated_simulator()
            self.assertTrue(len(sally.g.right_nodes) > 0)
            self.assertTrue(len(sally.g.left_nodes) > 0)

        # Select a pair of nodes that are not a red edge yet
        eid = choice(not_red_edges)
        # Try to add it; should succeed.
        try:
            out = sally.make_reds([[eid[0]]], [eid[1]])
        except LookupError:
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(False)
        else:
            for new_edge_id in out:
                self.assertIn(new_edge_id, sally.g.red_edges)
                self.assertEqual(new_edge_id[0], eid[0])
                self.assertEqual(new_edge_id[1], eid[1])
                self.assertEqual(new_edge_id[2], sally.t)

    # @ut.skip("Skipping test_make_reds_from_simulated_less_simple")
    def test_make_reds_from_simulated_less_simple(self):
        """ Tests that making more than one edge at a time succeeds."""
        # Roll a new simulator with at least one non-red-edge available.
        sally = make_simulated_simulator()
        num_edges_to_add_at_a_time = 2
        self.assertTrue(len(sally.g.right_nodes) > 0)
        self.assertTrue(len(sally.g.left_nodes) > 0)
        not_red_edges = [(x, y) for x in sally.g.left_nodes for y in sally.g.right_nodes if
                         all([eid[0] != x or eid[1] != y for eid in sally.g.red_edges])]
        while len(not_red_edges) == 0:
            sally = make_simulated_simulator()
            self.assertTrue(len(sally.g.right_nodes) > 0)
            self.assertTrue(len(sally.g.left_nodes) > 0)
            not_red_edges = [(x, y) for x in sally.g.left_nodes for y in sally.g.right_nodes if
                             all([eid[0] != x or eid[1] != y for eid in sally.g.red_edges])]

        # Select a pair of nodes that are not a red edge yet
        eids = sample(not_red_edges, k=num_edges_to_add_at_a_time)
        out = []
        for eid in eids:
            try:
                out += [sally.make_reds([[eid[0]]], [eid[1]])]
            except LookupError:
                self.assertTrue(False)
            except AttributeError:
                self.assertTrue(False)
            else:
                for new_edge_id in out[-1]:
                    self.assertIn(new_edge_id, sally.g.red_edges)
                    self.assertEqual(new_edge_id[0], eid[0])
                    self.assertEqual(new_edge_id[1], eid[1])
                    self.assertEqual(new_edge_id[2], sally.t)

    # @ut.skip("Skipping test_make_reds_from_simulated_complex")
    def test_make_reds_from_simulated_complex(self):
        """ Tests that making more than one edge at a time succeeds. This does not test gen_rings, and so ring
        members in this test are manually constructed. """
        # Let's do it again but moar.
        # Roll a new simulator, stepping forward to ensure that we end up with something with at least min_right
        # right nodes to select from and at least min_left left nodes to select from.
        min_right = 2
        desired_ring_size = 11
        min_left = min_right*desired_ring_size
        sally = make_simulated_simulator()
        while len(sally.g.right_nodes) <= min_right or len(sally.g.left_nodes) <= min_left:
            sally.step()

        # Select min_right right nodes independently and without replacement and, for each of these, select
        # desired_ring_size left nodes independently and without replacement to represent a ring.
        # Redraw until all purported edges are not yet red edges.
        signature_keys = sample(list(sally.g.right_nodes.keys()), k=min_right)
        rings = []
        for _ in signature_keys:
            rings += [sample(list(sally.g.left_nodes.keys()), k=desired_ring_size)]
        while any([eid[0] == x and eid[1] == y for ring in rings for x in ring for y in signature_keys for eid in
                   sally.g.red_edges]):
            signature_keys = sample(list(sally.g.right_nodes.keys()), k=min_right)
            rings = []
            for _ in signature_keys:
                rings += [sample(list(sally.g.left_nodes.keys()), k=desired_ring_size)]

        # Verify selected edges are not currently red edges
        self.assertTrue(not any([eid[0] == x and eid[1] == y for ring in rings for x in ring
                                 for y in signature_keys for eid in sally.g.red_edges]))
        # Verify rings is a list
        self.assertIsInstance(rings, list)
        # Verify signature_keys is a list
        self.assertIsInstance(signature_keys, list)
        # Verify rings and signature keys have the same length
        self.assertEqual(len(rings), len(signature_keys))
        # Verify these lengths match min_right
        self.assertEqual(len(rings), min_right)
        for ring in rings:
            # Verify each ring in rings is a list with desired_ring_size
            self.assertIsInstance(ring, list)
            self.assertEqual(len(ring), desired_ring_size)

        # Make list of edges we can expect.
        expected_red_edges = sorted([(x, y) for ring, y in zip(rings, signature_keys) for x in ring])
        # Attempt to make reds - shouldn't throw any exceptions
        try:
            out = sally.make_reds(rings, signature_keys)
        except LookupError:
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(False)
        else:
            # Check output is a subset of red edges matching our expected_red_edge set
            comparable_out = sorted([(_[0], _[1]) for _ in out])
            self.assertTrue(all([_ in sally.g.red_edges for _ in out]))
            self.assertTrue(all([_ in expected_red_edges for _ in comparable_out]))
            self.assertTrue(all([_ in comparable_out for _ in expected_red_edges]))

    # @ut.skip("Skipping test_make_blues_from_simulated")
    def test_make_blues_from_simulated(self):
        """ Tests that making blues from sim'd graph fails and succeeds appropriately."""
        # First things first: cannot add edges between nodes from empty lists.
        sally = make_simulated_simulator()
        try:
            sally.make_blues([], [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # Next, roll a sim and pick an arbitrary possible edge (x, y) such that (x, y) is not a blue edge.
        sally = make_simulated_simulator()
        self.assertTrue(len(sally.g.right_nodes) > 0)
        self.assertTrue(len(sally.g.left_nodes) > 0)
        y = choice(list(sally.g.right_nodes.keys()))
        x = choice(list(sally.g.left_nodes.keys()))
        while any([edge_id[0] == x and edge_id[1] == y for edge_id in sally.g.blue_edges]):
            y = choice(list(sally.g.right_nodes.keys()))
            x = choice(list(sally.g.left_nodes.keys()))

        old_ct = len(sally.g.blue_edges)
        # Try to make the blue edge (x, y)
        try:
            sally.make_blues([x], [y])
        except AttributeError:
            self.assertTrue(False)
        except LookupError:
            self.assertTrue(False)
        else:
            self.assertIn((x, y, sally.t), sally.g.blue_edges)
            self.assertTrue(old_ct + 1, len(sally.g.blue_edges))

    # @ut.skip("Skipping test_gen_spend_time_from_simulated")
    def test_gen_spend_time_from_simulated(self):
        """ Test whether gen_spend_time from a sim'd graph produces a positive integer bigger than min."""
        sally = make_simulated_simulator()
        for owner in range(len(sally.stochastic_matrix)):
            i = sally.gen_spend_time(owner)
            self.assertIsInstance(i, int)
            self.assertGreaterEqual(i, sally.min_spend_time)

    # @ut.skip("Skipping test_gen_recipient_from_simulated")
    def test_gen_recipient_from_simulated(self):
        """ Test that, for each owner, a newly generated recipient is an owner name."""
        # Note: Does not test the distribution.
        sally = make_simulated_simulator()
        for owner in sally.owner_names:
            self.assertIn(sally.gen_recipient(owner), sally.owner_names)

    # @ut.skip("Skipping test_gen_recipient_distribution_from_simulated")
    def test_gen_recipient_distribution_from_simulated(self):
        """ Test that, for each owner, a newly generated recipient is an owner name."""
        # TODO: Write. Requires statistical testing.
        pass

    # @ut.skip("Skipping test_make_coinbase_from_simulated")
    def test_make_coinbase_from_simulated(self):
        """ Tests that making a coinbase in a simulated simulator succeeds. Of course, we couldn't get a simulated
        simulator without this working! """
        sally = make_simulated_simulator()
        old_ct = len(sally.g.left_nodes)
        out = sally.make_coinbase()
        new_ct = len(sally.g.left_nodes)
        self.assertIn(out, sally.g.left_nodes)
        self.assertEqual(old_ct + 1, new_ct)

    # @ut.skip("Skipping test_gen_rings_from_simulated_fail")
    def test_gen_rings_from_simulated_fail(self):
        """ Test that generating rings for an empty list of true spenders fails. """
        sally = make_simulated_simulator()
        try:
            sally.gen_rings([])
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_gen_rings_from_simulated_simple")
    def test_gen_rings_from_simulated_simple(self):
        """ Tests that gen_rings for a singleton works as expected. """
        sally = make_simulated_simulator()
        available = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time <= sally.t]
        eff_ring_size = min(sally.ringsize, len(available))
        while eff_ring_size == 0:
            sally = make_simulated_simulator()
            available = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time <= sally.t]
            eff_ring_size = min(sally.ringsize, len(available))
        x = choice(available)
        try:
            out = sally.gen_rings([x])
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), eff_ring_size)
            self.assertTrue(all([_ in sally.g.left_nodes for ring in out for _ in ring]))

    # @ut.skip("Skipping test_gen_rings_from_simulated_complex")
    def test_gen_rings_from_simulated_complex(self):
        """ Tests that gen_rings for a list of more than one element works as expected. """
        # Reroll a simulator until we have at least two keys waiting to be spent.
        sally = make_simulated_simulator()
        available = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time <= sally.t]
        eff_ring_size = min(sally.ringsize, len(available))
        signing_keys = [_[2] for _ in sally.buffer[sally.t]]
        while len(signing_keys) <= 1 or eff_ring_size == 0:
            sally = make_simulated_simulator()
            available = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time <= sally.t]
            eff_ring_size = min(sally.ringsize, len(available))
            signing_keys = [_[2] for _ in sally.buffer[sally.t]]
        try:
            out = sally.gen_rings(signing_keys)
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), len(signing_keys))
            self.assertTrue(all([len(_) == eff_ring_size for _ in out]))
            self.assertTrue(all([_ in sally.g.left_nodes for ring in out for _ in ring]))

    # @ut.skip("Skipping test_gen_rings_distribution_from_simulated")
    def test_gen_rings_distribution_from_simulated(self):
        # TODO: Requires statistical testing
        pass

    # @ut.skip("Skipping test_make_txn_from_simulated")
    def test_make_txn_from_simulated(self):
        """ Tests make_txn from a sim'd graph. """
        # TODO: This test should be split into more than one test with various names as per make_reds
        # Check that making a transaction without signing keys throws an AttributeError, regardless of sender and recip
        sally = make_simulated_simulator()
        k = len(sally.hashrate)
        del sally
        for sender in range(k):
            for recip in range(k):
                sally = make_simulated_simulator()
                try:
                    sally.make_txn(sender, recip, [])
                except AttributeError:
                    self.assertTrue(True)
                else:
                    self.assertTrue(False)

        # For each sender-recipient pair, roll a new simulator that has at least one available ring member and make_txn
        for sender in range(k):
            for recip in range(k):
                sally = make_simulated_simulator()
                available = [_ for _ in sally.g.left_nodes if _[1] + MIN_SPEND_TIME < sally.t]
                while len(available) == 0:
                    sally = make_simulated_simulator()
                    available = [_ for _ in sally.g.left_nodes if _[1] + MIN_SPEND_TIME < sally.t]
                x = choice(available)
                try:
                    sally.make_txn(sender, recip, [(sender, recip, x)])
                except AttributeError:
                    self.assertTrue(False)
                else:
                    self.assertTrue(True)

        # Reroll a fresh simulated simulator; recall the buffer of this is non-empty.
        sally = make_simulated_simulator()
        self.assertNotEqual(len(sally.buffer[sally.t + 1])*len(sally.buffer[sally.t]), 0)
        self.assertLess(sally.t + 1, sally.runtime)
        bndl = groupby(sally.buffer[sally.t], key=lambda entry: (entry[0], entry[1]))
        for k, grp in bndl:
            sender = randrange(len(sally.stochastic_matrix))
            recip = randrange(len(sally.stochastic_matrix))
            try:
                sally.make_txn(sender, recip, grp)
            except AttributeError:
                self.assertTrue(False)
            else:
                self.assertTrue(True)

        sally.t += 1

        bndl = groupby(sally.buffer[sally.t], key=lambda entry: (entry[0], entry[1]))
        for k, grp in bndl:
            sender = randrange(len(sally.stochastic_matrix))
            recip = randrange(len(sally.stochastic_matrix))
            try:
                sally.make_txn(sender, recip, grp)
            except AttributeError:
                self.assertTrue(False)
            else:
                self.assertTrue(True)

    # @ut.skip("Skipping test_make_txns_from_simulated")
    def test_make_txns_from_simulated(self):
        """ Testing make_txns from a sim'd graph with some txns loaded in the buffer. """
        # Generate a new simulated graph
        sally = make_simulated_simulator()

        # Measure some old statistics
        old_num_rights = len(sally.g.right_nodes)
        old_num_lefts = len(sally.g.left_nodes)

        # Make predictions
        bndl = groupby(sally.buffer[sally.t], key=lambda entry: (entry[0], entry[1]))
        is_empty = all(False for _ in deepcopy(bndl))
        self.assertFalse(is_empty)
        expected_new_lefts = 2*len([(k, grp) for k, grp in bndl])
        expected_new_rights = len(sally.buffer[sally.t])

        # Call make_txns
        out = sally.make_txns()

        # Check that output is a non-empty list and we gained the expected number of lefts and rights.
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)
        new_num_rights = len(sally.g.right_nodes)
        new_num_lefts = len(sally.g.left_nodes)
        self.assertEqual(old_num_rights + expected_new_rights, new_num_rights)
        self.assertEqual(old_num_lefts + expected_new_lefts, new_num_lefts)

        for txn in out:
            self.assertIsInstance(txn, list)
            self.assertEqual(len(txn), 5)
            [new_rights, new_lefts, rings, new_reds, new_blues] = txn
            self.assertIsInstance(new_rights, list)
            self.assertIsInstance(new_lefts, list)
            self.assertEqual(len(new_lefts), 2)
            self.assertIsInstance(rings, list)
            self.assertEqual(len(new_rights), len(rings))
            self.assertIsInstance(new_reds, list)
            self.assertIsInstance(new_blues, list)
            self.assertEqual(len(new_blues), len(new_rights)*2)
            self.assertTrue(all([_ in sally.g.right_nodes for _ in new_rights]))
            self.assertTrue(all([_ in sally.g.left_nodes for _ in new_lefts]))
            self.assertTrue(all([_ in sally.g.red_edges for _ in new_reds]))
            self.assertTrue(all([_ in sally.g.blue_edges for _ in new_blues]))
            self.assertTrue(all([any([_[0] == x and _[1] == y for _ in new_reds]) for R, y in zip(rings, new_rights)
                                 for x in R]))
            self.assertTrue(all([_[1] in new_rights for _ in new_reds]))
            self.assertTrue(all([any([x == _[0] for R in rings for x in R]) for _ in new_reds]))

    # @ut.skip("Skipping test_update_state_from_simulated")
    def test_update_state_from_simulated_simple(self):
        # Generate simulator
        sally = make_simulated_simulator()
        old_stats, old_aux, old_pred = self.gather_stats_from_sally(sally)
        total_dt = deepcopy(sally.dt)
        self.assertEqual(total_dt, 1)
        old_sally = deepcopy(sally)
        out = sally.update_state(total_dt)  # calls make_coinbase and make_txns exactly once
        self.assertEqual(old_sally.t + total_dt, sally.t)
        new_stats, new_aux, new_pred = self.gather_stats_from_sally(sally)
        self.compare_stats_with_predictions(sally, old_stats, old_aux, old_pred, new_stats, new_aux, new_pred, out,
                                            total_dt)

    # @ut.skip("Skipping test_update_state_from_simulated")
    def test_update_state_from_simulated_simple_iterated(self):
        # Generate simulator
        sally = make_simulated_simulator()

        for _ in range(SAMPLE_SIZE):
            old_stats, old_aux, old_pred = self.gather_stats_from_sally(sally)
            old_sally = deepcopy(sally)
            out = sally.update_state()  # calls make_coinbase and make_txns exactly once
            self.assertEqual(old_sally.t + sally.dt, sally.t)
            new_stats, new_aux, new_pred = self.gather_stats_from_sally(sally)
            self.compare_stats_with_predictions(sally, old_stats, old_aux, old_pred, new_stats, new_aux, new_pred, out,
                                                sally.dt)

    @ut.skip("Skipping test_update_state_from_simulated_complex")
    def test_update_state_from_simulated_complex(self):
        magic_numbers = [17, 10]

        sally = make_simulated_simulator()
        old_stats, old_aux, old_pred = self.gather_stats_from_sally(sally)
        total_dt = magic_numbers[1]
        old_sally = deepcopy(sally)
        try:
            out = sally.update_state(total_dt)
        except AttributeError:
            self.assertTrue(False)
        except RuntimeError:
            self.assertTrue(False)
        else:
            self.assertEqual(old_sally.t + total_dt, sally.t)
            new_stats, new_aux, new_pred = self.gather_stats_from_sally(sally)
            self.compare_stats_with_predictions(sally, old_stats, old_aux, old_pred, new_stats, new_aux, new_pred, out,
                                                total_dt)

    # @ut.skip("Skipping test_make_simulated_simulator")
    def test_make_simulated_simulator(self):
        """ Test make_simulated_simulator produces a nonempty graph with some stuff  about to happen or with a clock
        that has run out."""
        sally = make_simulated_simulator()
        self.assertGreater(len(sally.g.left_nodes), 0)
        self.assertGreater(len(sally.g.right_nodes), 0)
        self.assertGreater(sally.runtime, sally.t + 1)
        self.assertTrue(sally.t >= sally.runtime or len(sally.buffer[sally.t])*len(sally.buffer[sally.t + 1]) > 0)

    # @ut.skip("Skipping test_step_from_simulated")
    def test_step_from_simulated(self):
        """ Test the step() function. """
        sally = make_simulated_simulator()
        old_t = sally.t
        out = sally.step()
        new_t = sally.t
        self.assertEqual(old_t + sally.dt, new_t)
        self.assertEqual(len(out), sally.dt)
        for block in out:
            self.assertEqual(len(block), 2)
            cb, txns = block
            self.assertIn(cb, sally.g.left_nodes)
            for txn in txns:
                [new_rights, new_lefts, rings, new_reds, new_blues] = txn
                for _ in new_rights:
                    self.assertIn(_, sally.g.right_nodes)
                for _ in new_lefts:
                    self.assertIn(_, sally.g.left_nodes)
                for R, y in zip(rings, new_rights):
                    for x in R:
                        self.assertTrue(any([eid[0] == x and eid[1] == y for eid in sally.g.red_edges]))
                        self.assertTrue(any([eid[0] == x and eid[1] == y for eid in new_reds]))
                for new_red in new_reds:
                    self.assertTrue(new_red in sally.g.red_edges)
                for y in new_rights:
                    for z in new_lefts:
                        self.assertTrue(any([eid[0] == z and eid[1] == y for eid in sally.g.blue_edges]))
                        self.assertTrue(any([eid[0] == z and eid[1] == y for eid in new_blues]))
                for new_blue in new_blues:
                    self.assertTrue(new_blue in sally.g.blue_edges)

    # @ut.skip("Skipping test_run_from_simulated")
    def test_run_from_simulated(self):
        sally = make_simulated_simulator()
        try:
            sally.run()
        except AttributeError:
            self.assertTrue(False)
        except RuntimeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    # REPETITION OF SINGLE-USE TESTS FROM A SIMULATED LEDGER ####

    # @ut.skip("Skipping test_gen_time_step_from_simulated_repeated")
    def test_gen_time_step_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_time_step_from_simulated()

    # @ut.skip("Skipping test_gen_coinbase_owner_from_simulated_repeated")
    def test_gen_coinbase_owner_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_simulated()

    # @ut.skip("Skipping test_gen_coinbase_amt_from_simulated_repeated")
    def test_gen_coinbase_amt_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_amt_from_simulated()

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_simulated_repeated")
    def test_add_left_node_to_buffer_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_add_left_node_to_buffer_from_simulated()

    # @ut.skip("Skipping test_make_lefts_from_simulated_repeated")
    def test_make_lefts_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_lefts_from_simulated()

    # @ut.skip("Skipping test_make_rights_from_simulated_fail_repeated")
    def test_make_rights_from_simulated_fail_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_rights_from_simulated_fail()

    # @ut.skip("Skipping test_make_rights_from_simulated_simple_repeated")
    def test_make_rights_from_simulated_simple_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_rights_from_simulated_simple()

    # @ut.skip("Skipping test_make_rights_from_simulated_complex_repeated")
    def test_make_rights_from_simulated_complex_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_rights_from_simulated_complex()

    # @ut.skip("Skipping test_make_reds_from_simulated_no_input_repeated")
    def test_make_reds_from_simulated_no_input_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_reds_from_simulated_no_input()

    # @ut.skip("Skipping test_make_reds_from_simulated_simple_repeated")
    def test_make_reds_from_simulated_simple_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_reds_from_simulated_simple()

    # @ut.skip("Skipping test_make_reds_from_simulated_less_simple_repeated")
    def test_make_reds_from_simulated_less_simple_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_reds_from_simulated_less_simple()

    # @ut.skip("Skipping test_make_reds_from_simulated_complex_repeated")
    def test_make_reds_from_simulated_complex_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_reds_from_simulated_complex()

    # @ut.skip("Skipping test_make_blues_from_simulated_repeated")
    def test_make_blues_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_blues_from_simulated()

    # @ut.skip("Skipping test_gen_spend_time_from_simulated_repeated")
    def test_gen_spend_time_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_spend_time_from_simulated()

    # @ut.skip("Skipping test_gen_recipient_from_simulated_repeated")
    def test_gen_recipient_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_recipient_from_simulated()

    # @ut.skip("Skipping test_gen_recipient_distribution_from_simulated_repeated")
    def test_gen_recipient_distribution_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_recipient_distribution_from_simulated()

    # @ut.skip("Skipping test_make_coinbase_from_simulated_repeated")
    def test_make_coinbase_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_coinbase_from_simulated()

    # @ut.skip("Skipping test_gen_rings_from_simulated_fail_repeated")
    def test_gen_rings_from_simulated_fail_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_from_simulated_fail()

    # @ut.skip("Skipping test_gen_rings_from_simulated_simple_repeated")
    def test_gen_rings_from_simulated_simple_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_from_simulated_simple()

    # @ut.skip("Skipping test_gen_rings_from_simulated_complex_repeated")
    def test_gen_rings_from_simulated_complex_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_from_simulated_complex()

    # @ut.skip("Skipping test_gen_rings_distribution_from_simulated_repeated")
    def test_gen_rings_distribution_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_distribution_from_simulated()

    # @ut.skip("Skipping test_make_txn_from_simulated_repeated")
    def test_make_txn_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txn_from_simulated()

    # @ut.skip("Skipping test_make_txns_from_simulated_repeated")
    def test_make_txns_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txns_from_simulated()

    # @ut.skip("Skipping test_update_state_from_simulated_repeated")
    def test_update_state_from_simulated_simple_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_update_state_from_simulated_simple()

    # @ut.skip("Skipping test_update_state_from_simulated_repeated")
    def test_update_state_from_simulated_simple_iterated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_update_state_from_simulated_simple_iterated()

    # @ut.skip("Skipping test_make_simulated_simulator_repeated")
    def test_make_simulated_simulator_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_simulated_simulator()

    # @ut.skip("Skipping test_step_from_simulated_repeated")
    def test_step_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_step_from_simulated()

    # @ut.skip("Skipping test_run_from_simulated_repeated")
    def test_run_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_run_from_simulated()


ut.TextTestRunner(verbosity=2, failfast=True).run(ut.TestLoader().loadTestsFromTestCase(TestSimulator))
