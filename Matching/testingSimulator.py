import unittest as ut
from itertools import groupby
from graphtheory import *
from simulator import *
from copy import deepcopy
from random import sample, random, choice
from math import floor

SAMPLE_SIZE = 2

FILENAME = "data/output.txt"
STOCHASTIC_MATRIX = [[0.0, 0.9, 1.0 - 0.9], [0.125, 0.75, 0.125], [0.75, 0.25, 0.0]]
HASH_RATE_ALICE = 0.33
HASH_RATE_EVE = 0.33
HASH_RATE_BOB = 1.0 - HASH_RATE_ALICE - HASH_RATE_EVE
HASH_RATE = [HASH_RATE_ALICE, HASH_RATE_EVE, HASH_RATE_BOB]
RING_SIZE = 11
MIN_SPEND_TIME = RING_SIZE + 2  # Guarantees enough ring members for all signatures

# Expected spend-times are 20.0, 100.0, and 50.0 blocks, respectively.
SPEND_TIMES = [lambda x: 0.05 * ((1.0 - 0.05) ** (x - MIN_SPEND_TIME)),
               lambda x: 0.01 * ((1.0 - 0.01) ** (x - MIN_SPEND_TIME)),
               lambda x: 0.025 * ((1.0 - 0.025) ** (x - MIN_SPEND_TIME))]
RUNTIME = 100


def make_simulator():
    inp = dict()
    inp.update({'runtime': RUNTIME, 'hashrate': HASH_RATE, 'stochastic matrix': STOCHASTIC_MATRIX})
    inp.update({'spend times': SPEND_TIMES, 'min spend time': MIN_SPEND_TIME, 'ring size': RING_SIZE})
    inp.update({'flat': False, 'timestep': 1, 'max atomic': 2**64 - 1, 'emission': 2**-18, 'min reward': 6e11})
    label_msg = (RUNTIME, HASH_RATE[0], HASH_RATE[1], HASH_RATE[2], STOCHASTIC_MATRIX[0][0], STOCHASTIC_MATRIX[0][1], STOCHASTIC_MATRIX[0][2], STOCHASTIC_MATRIX[1][0], STOCHASTIC_MATRIX[1][1], STOCHASTIC_MATRIX[1][2], STOCHASTIC_MATRIX[2][0], STOCHASTIC_MATRIX[2][1], STOCHASTIC_MATRIX[2][2], 1.0/SPEND_TIMES[0](1), 1.0/SPEND_TIMES[1](1), 1.0/SPEND_TIMES[2](1), MIN_SPEND_TIME, RING_SIZE)
    label = str(hash(label_msg))
    label = label[-8:]
    fn = FILENAME[:-4] + str(label) + FILENAME[-4:]
    with open(fn, "w+") as wf:
        pass
    inp.update({'filename': fn})
    return Simulator(inp)


def make_simulated_simulator():
    sally = make_simulator()
    out = []
    while len(sally.buffer[sally.t])*len(sally.buffer[sally.t + 1]) == 0 or sally.t + 1 >= sally.runtime:
        while sally.t + 1 < sally.runtime and len(sally.buffer[sally.t])*len(sally.buffer[sally.t + 1]) == 0:
            out += [sally.step()]
        if sally.t + 1 >= sally.runtime or sally.t >= sally.runtime or len(sally.buffer[sally.t])*len(sally.buffer[sally.t + 1]) == 0:
            sally = make_simulator()
            out = []
    return sally


class TestSimulator(ut.TestCase):
    """ tests for simulator.py """

    #### SINGLE USE TESTS FROM EMPTY LEDGER ####

    # @ut.skip("Skipping test_init")
    def test_init(self):
        pass

    # @ut.skip("Skipping test_gen_time_step_from_empty")
    def test_gen_time_step_from_empty(self):
        sally = make_simulator()
        dt = sally.gen_time_step()
        self.assertEqual(dt, 1)

    # @ut.skip("Skipping test_gen_coinbase_owner_from_empty")
    def test_gen_coinbase_owner_from_empty(self):
        sally = make_simulator()
        results = []
        domain = list(range(len(sally.stochastic_matrix)))
        for i in range(SAMPLE_SIZE):
            results += [sally.gen_coinbase_owner()]
        self.assertTrue(all([_ in domain for _ in results]))

    # @ut.skip("Skipping test_gen_coinbase_amt_from_empty")
    def test_gen_coinbase_amt_from_empty(self):
        sally = make_simulator()

        pred_v = EMISSION_RATIO * MAX_MONERO_ATOMIC_UNITS
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

    # @ut.skip("Skipping test_make_lefts_from_empty")
    def test_make_lefts_from_empty(self):
        sally = make_simulator()
        n = 2*len(sally.stochastic_matrix)*len(sally.stochastic_matrix)
        old_ct = len(sally.g.left_nodes)
        x = []
        for sender in range(len(sally.stochastic_matrix)):
            for recip in range(len(sally.stochastic_matrix)):
                amt = random()
                x += [sally.make_lefts(sender, recip, amt)]
        new_ct = len(sally.g.left_nodes)
        self.assertEqual(old_ct + n, new_ct)
        for each in x:
            self.assertEqual(len(each), 2)
            self.assertIn(each[0], sally.g.left_nodes)
            self.assertIn(each[0], sally.ownership)
            self.assertIn(each[0], sally.amounts)
            self.assertIn(sally.ownership[each[0]], range(len(sally.stochastic_matrix)))

            self.assertIn(each[1], sally.g.left_nodes)
            self.assertIn(each[1], sally.ownership)
            self.assertIn(each[1], sally.amounts)
            self.assertIn(sally.ownership[each[1]], range(len(sally.stochastic_matrix)))

    # @ut.skip("Skipping test_make_rights_from_empty")
    def test_make_rights_from_empty(self):
        sally = make_simulator()
        for owner in range(len(sally.stochastic_matrix)):
            try:
                out = sally.make_rights(owner, [])
            except:
                self.assertTrue(False)
            else:
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), 0)

    # @ut.skip("Skipping test_make_reds_from_empty")
    def test_make_reds_from_empty(self):
        sally = make_simulator()
        try:
            out = sally.make_reds([], [])
            self.assertEqual(out, [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_blues_from_empty")
    def test_make_blues_from_empty(self):
        sally = make_simulator()
        try:
            sally.make_blues([], [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_gen_spend_time_from_empty")
    def test_gen_spend_time_from_empty(self):
        sally = make_simulator()
        # spend_times = []
        for owner in range(len(sally.stochastic_matrix)):
            for _ in range(SAMPLE_SIZE):
                i = sally.gen_spend_time(owner)
                self.assertTrue(i in range(sally.runtime))

    # @ut.skip("Skipping test_gen_recipient_from_empty")
    def test_gen_recipient_from_empty(self):
        sally = make_simulator()
        allowed_idxs = list(range(len(sally.hashrate)))
        for x in range(SAMPLE_SIZE):
            for owner in range(len(sally.stochastic_matrix)):
                for _ in range(SAMPLE_SIZE):
                    self.assertIn(sally.gen_recipient(owner), allowed_idxs)

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_empty")
    def test_add_left_node_to_buffer_from_empty(self):
        sally = make_simulator()

        for owner in range(len(sally.hashrate)):
            x = sally.g.add_node(0)
            try:
                sally.add_left_node_to_buffer(x, owner)
            except Exception:
                self.assertTrue(False)
            else:
                self.assertTrue(True)

    # @ut.skip("Skipping test_make_coinbase_from_empty")
    def test_make_coinbase_from_empty(self):
        sally = make_simulator()
        for _ in range(SAMPLE_SIZE):
            out = sally.make_coinbase()
            self.assertIn(out, sally.g.left_nodes)
            sally.t += 1

    # @ut.skip("Skipping test_gen_rings_from_empty")
    def test_gen_rings_from_empty(self):
        sally = make_simulator()
        try:
            out = sally.gen_rings([])
        except Exception:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    # @ut.skip("Skipping test_make_txn_from_empty")
    def test_make_txn_from_empty(self):
        sally = make_simulator()
        for sender in range(len(sally.stochastic_matrix)):
            for recip in range(len(sally.stochastic_matrix)):
                try:
                    out = sally.make_txn(sender, recip, [])
                except AttributeError:
                    self.assertTrue(True)

    # @ut.skip("Skipping test_make_txns_from_empty")
    def test_make_txns_from_empty(self):
        sally = make_simulator()
        self.assertEqual(sally.make_txns(), [])

    # @ut.skip("Skipping test_update_state_from_empty")
    def test_update_state_from_empty(self):
        sally = make_simulator()
        
        available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time < sally.t]
        eff_ring_size = min(sally.ringsize, len(available_ring_members))
        
        old_num_left_nodes = len(sally.g.left_nodes)
        old_num_right_nodes = len(sally.g.right_nodes)
        old_num_blue_edges = len(sally.g.blue_edges)
        old_num_red_edges = len(sally.g.red_edges)
        
        pred_num_new_rights = len(sally.buffer[sally.t+1])
        pred_num_new_lefts = 1 + 2*pred_num_new_rights
        pred_num_red_edges = eff_ring_size*pred_num_new_rights
        pred_num_blue_edges = 2*pred_num_new_rights
        
        out = sally.update_state(1)
        
        new_num_left_nodes = len(sally.g.left_nodes)
        new_num_right_nodes = len(sally.g.right_nodes)
        new_num_blue_edges = len(sally.g.blue_edges)
        new_num_red_edges = len(sally.g.red_edges)
        
        self.assertEqual(len(out), sally.dt)
        self.assertEqual(len(out[0]), 2)
        self.assertIsInstance(out[0][1], list)
        self.assertIn(out[0][0], sally.g.left_nodes)
        self.assertEqual(new_num_left_nodes, pred_num_new_lefts)
        self.assertEqual(new_num_right_nodes, pred_num_new_rights)
        self.assertEqual(new_num_blue_edges, pred_num_blue_edges)
        self.assertEqual(new_num_red_edges, pred_num_red_edges)
        
        for txn in out[0][1]:
            [new_rights, new_lefts, rings, new_reds, new_blues] = txn
            for new_right in new_rights:
                self.assertTrue(new_right in sally.g.right_nodes)
            for new_left in new_lefts:
                self.assertTrue(new_left in sally.g.left_nodes)
            for R, new_right in zip(rings, new_rights):
                for ring_member in R:
                    self.assertIn((ring_member, new_right, sally.t) in sally.g.red_edges)
                    self.assertIn((ring_member, new_right, sally.t) in new_reds)
            for new_left in new_lefts:
                for new_right in new_rights:
                    self.assertIn((new_left, new_right, sally.t) in sally.g.blue_edges)
                    self.assertIn((new_left, new_right, sally.t) in new_blues)

    #### REPETITION OF SINGLE-USE TESTS ####

    # @ut.skip("Skipping test_gen_time_step_from_empty_repeated")
    def test_gen_time_step_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_time_step_from_empty()

    # @ut.skip("Skipping test_gen_coinbase_owner_from_empty_repeated")
    def test_gen_coinbase_owner_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_owner_from_empty()

    # @ut.skip("Skipping test_gen_coinbase_amt_from_empty_repeated")
    def test_gen_coinbase_amt_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_coinbase_amt_from_empty()

    # @ut.skip("Skipping test_make_lefts_from_empty_repeated")
    def test_make_lefts_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_lefts_from_empty()

    # @ut.skip("Skipping test_make_rights_from_empty_repeated")
    def test_make_rights_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_rights_from_empty()

    # @ut.skip("Skipping test_make_reds_from_empty_repeated")
    def test_make_reds_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_reds_from_empty()

    # @ut.skip("Skipping test_make_blues_from_empty_repeated")
    def test_make_blues_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_blues_from_empty()

    # @ut.skip("Skipping test_gen_spend_time_from_empty_repeated")
    def test_gen_spend_time_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_spend_time_from_empty()

    # @ut.skip("Skipping test_gen_recipient_from_empty_repeated")
    def test_gen_recipient_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_recipient_from_empty()

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_empty_repeated")
    def test_add_left_node_to_buffer_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_add_left_node_to_buffer_from_empty()

    # @ut.skip("Skipping test_make_coinbase_from_empty_repeated")
    def test_make_coinbase_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_coinbase_from_empty()

    # @ut.skip("Skipping test_gen_rings_from_empty_repeated")
    def test_gen_rings_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_from_empty()

    # @ut.skip("Skipping test_make_txn_from_empty_repeated")
    def test_make_txn_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txn_from_empty()

    # @ut.skip("Skipping test_make_txns_from_empty_repeated")
    def test_make_txns_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txns_from_empty()

    # @ut.skip("Skipping test_update_state_from_empty_repeated")
    def test_update_state_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_update_state_from_empty()

    @ut.skip("Skipping test_run_from_empty_repeated")
    def test_run_from_empty_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_run_from_empty()
            
    #### SINGLE-USE TESTS FROM A SIMULATED LEDGER. ####

    # @ut.skip("Skipping test_gen_time_step_from_simulated")
    def test_gen_time_step_from_simulated(self):
        sally = make_simulated_simulator()
        dt = sally.gen_time_step()
        self.assertEqual(dt, 1)

    # @ut.skip("Skipping test_gen_coinbase_owner_from_simulated")
    def test_gen_coinbase_owner_from_simulated(self):
        sally = make_simulated_simulator()
        results = []
        domain = list(range(len(sally.stochastic_matrix)))
        for i in range(SAMPLE_SIZE):
            results += [sally.gen_coinbase_owner()]
        self.assertTrue(all([_ in domain for _ in results]))

    # @ut.skip("Skipping test_gen_coinbase_amt_from_simulated")
    def test_gen_coinbase_amt_from_simulated(self):
        sally = make_simulated_simulator()
        while sally.runtime - sally.t < 10:
            sally = make_simulated_simulator()

        pred_v = EMISSION_RATIO * MAX_MONERO_ATOMIC_UNITS * (1.0 - EMISSION_RATIO)**(sally.t)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)

        v = sally.gen_coinbase_amt()
        # print("Generating genesis cb, got " + str(v))
        # self.assertEqual(v, pred_v)
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)

        sally.t += 1
        v = sally.gen_coinbase_amt()
        # print("Generated next cb, got " + str(v))
        # self.assertEqual(v, pred_v)
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)

        sally.t += 1
        v = sally.gen_coinbase_amt()
        # print("Generated third cb, got " + str(v))
        # self.assertEqual(v, pred_v)
        pred_v = max(v*(1.0 - EMISSION_RATIO), MIN_MINING_REWARD)
        self.assertLessEqual(abs(sally.next_mining_reward - pred_v), 1e-1)

    # @ut.skip("Skipping test_make_lefts_from_simulated")
    def test_make_lefts_from_simulated(self):
        sally = make_simulated_simulator()
        n = 2*len(sally.stochastic_matrix)*len(sally.stochastic_matrix)
        old_ct = len(sally.g.left_nodes)
        x = []
        for sender in range(len(sally.stochastic_matrix)):
            for recip in range(len(sally.stochastic_matrix)):
                amt = random()
                x += [sally.make_lefts(sender, recip, amt)]
        new_ct = len(sally.g.left_nodes)
        self.assertEqual(old_ct + n, new_ct)
        for each in x:
            self.assertEqual(len(each), 2)
            self.assertIn(each[0], sally.g.left_nodes)
            self.assertIn(each[0], sally.ownership)
            self.assertIn(each[0], sally.amounts)
            self.assertIn(sally.ownership[each[0]], range(len(sally.stochastic_matrix)))

            self.assertIn(each[1], sally.g.left_nodes)
            self.assertIn(each[1], sally.ownership)
            self.assertIn(each[1], sally.amounts)
            self.assertIn(sally.ownership[each[1]], range(len(sally.stochastic_matrix)))

    # @ut.skip("Skipping test_make_rights_from_simulated")
    def test_make_rights_from_simulated(self):
        for owner in range(len(STOCHASTIC_MATRIX)):
            sally = make_simulated_simulator()
            available_left_nodes = [x for x in sally.g.left_nodes if x[1] + MIN_SPEND_TIME < sally.t]
            while len(available_left_nodes) == 0:
                sally = make_simulated_simulator()
                available_left_nodes = [x for x in sally.g.left_nodes if x[1] + MIN_SPEND_TIME < sally.t]
            true_spender = choice(available_left_nodes)
            try:
                out = sally.make_rights(owner, [])
            except:
                self.assertTrue(False)
            else:
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), 0)

    # @ut.skip("Skipping test_make_reds_from_simulated")
    def test_make_reds_from_simulated(self):
        sally = make_simulated_simulator()
        try:
            out = sally.make_reds([], [])
            self.assertEqual(out, [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        sally = make_simulated_simulator()
        self.assertTrue(len(sally.g.right_nodes) > 0)
        self.assertTrue(len(sally.g.left_nodes) > 0)
        y = choice(list(sally.g.right_nodes.keys()))
        x = choice(list(sally.g.left_nodes.keys()))
        while any([edge_id[0] == x and edge_id[1] == y for edge_id in sally.g.red_edges]):        
            y = choice(list(sally.g.right_nodes.keys()))
            x = choice(list(sally.g.left_nodes.keys()))
        try:
            out = sally.make_reds([[x]], [y])
        except LookupError:
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(False)
        except IndexError:
            self.assertTrue(False)
        except:
            self.assertTrue(False)
        else:
            for new_edge_id in out:
                self.assertIn(new_edge_id, sally.g.red_edges)
                self.assertEqual(new_edge_id[0], x)
                self.assertEqual(new_edge_id[1], y)
                self.assertEqual(new_edge_id[2], sally.t)

    # @ut.skip("Skipping test_make_blues_from_simulated")
    def test_make_blues_from_simulated(self):
        sally = make_simulated_simulator()
        try:
            sally.make_blues([], [])
        except LookupError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        sally = make_simulated_simulator()
        self.assertTrue(len(sally.g.right_nodes) > 0)
        self.assertTrue(len(sally.g.left_nodes) > 0)
        y = choice(list(sally.g.right_nodes.keys()))
        x = choice(list(sally.g.left_nodes.keys()))
        while any([edge_id[0] == x and edge_id[1] == y for edge_id in sally.g.blue_edges]):
            y = choice(list(sally.g.right_nodes.keys()))
            x = choice(list(sally.g.left_nodes.keys()))
        try:
            out = sally.make_blues([x], [y])
        except AttributeError:
            self.assertTrue(False)
        except LookupError:
            self.assertTrue(False)
        except IndexError:
            self.assertTrue(False)
        except:
            self.assertTrue(False)
        else:
            self.assertIn((x, y, sally.t), sally.g.blue_edges)

    # @ut.skip("Skipping test_gen_spend_time_from_simulated")
    def test_gen_spend_time_from_simulated(self):
        sally = make_simulated_simulator()
        for owner in range(len(sally.stochastic_matrix)):
            for _ in range(SAMPLE_SIZE):
                i = sally.gen_spend_time(owner)
                self.assertTrue(i in range(sally.runtime))

    # @ut.skip("Skipping test_gen_recipient_from_simulated")
    def test_gen_recipient_from_simulated(self):
        sally = make_simulated_simulator()
        allowed_idxs = list(range(len(sally.hashrate)))
        for owner in range(len(sally.stochastic_matrix)):
            for _ in range(SAMPLE_SIZE):
                self.assertIn(sally.gen_recipient(owner), allowed_idxs)

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_simulated")
    def test_add_left_node_to_buffer_from_simulated(self):
        sally = make_simulated_simulator()

        for owner in range(len(sally.hashrate)):
            x = sally.g.add_node(0)
            try:
                sally.add_left_node_to_buffer(x, owner)
            except Exception:
                self.assertTrue(False)
            else:
                self.assertTrue(True)

    # @ut.skip("Skipping test_make_coinbase_from_simulated")
    def test_make_coinbase_from_simulated(self):
        sally = make_simulated_simulator()
        old_ct = len(sally.g.left_nodes)
        out = sally.make_coinbase()
        new_ct = len(sally.g.left_nodes)
        self.assertIn(out, sally.g.left_nodes)
        self.assertEqual(old_ct + 1, new_ct)

    # @ut.skip("Skipping test_gen_rings_from_simulated")
    def test_gen_rings_from_simulated(self):
        sally = make_simulated_simulator()
        try:
            out = sally.gen_rings([])
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        sally = make_simulated_simulator()
        available = [_ for _ in sally.g.left_nodes if _[1] + MIN_SPEND_TIME < sally.t]
        x = choice(available)
        try:
            out = sally.gen_rings([x])
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), sally.ringsize)

    # @ut.skip("Skipping test_make_txn_from_simulated")
    def test_make_txn_from_simulated(self):
        sally = make_simulated_simulator()
        for sender in range(len(sally.stochastic_matrix)):
            for recip in range(len(sally.stochastic_matrix)):
                try:
                    out = sally.make_txn(sender, recip, [])
                except AttributeError:
                    self.assertTrue(True)
                else:
                    self.assertTrue(False)

        sally = make_simulated_simulator()
        while len([_ for _ in sally.g.left_nodes if _[1] + MIN_SPEND_TIME < sally.t]) == 0:
            sally = make_simulated_simulator()
        for sender in range(len(sally.stochastic_matrix)):
            for recip in range(len(sally.stochastic_matrix)):
                available = [_ for _ in sally.g.left_nodes if _[1] + MIN_SPEND_TIME < sally.t]
                x = choice(available)
                try:
                    out = sally.make_txn(sender, recip, [x])
                except AttributeError:
                    self.assertTrue(False)

    # @ut.skip("Skipping test_make_txns_from_simulated")
    def test_make_txns_from_simulated(self):
        sally = make_simulated_simulator()
        expected_number_sigs = len(sally.buffer[sally.t])
        out = sally.make_txns()
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)
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
            
    # @ut.skip("Skipping test_update_state_from_simulated")
    def test_update_state_from_simulated(self):
        # Generate simulator
        sally = make_simulated_simulator()
        
        # Collect some stats
        old_num_left_nodes = len(sally.g.left_nodes)
        old_num_right_nodes = len(sally.g.right_nodes)
        old_num_blue_edges = len(sally.g.blue_edges)
        old_num_red_edges = len(sally.g.red_edges)
        available_ring_members = [_ for _ in sally.g.left_nodes if _[1] + sally.min_spend_time < sally.t]
        eff_ring_size = min(sally.ringsize, len(available_ring_members))
        self.assertGreaterEqual(len(available_ring_members), 1)

        
        bndl = groupby(sally.buffer[sally.t+1], key=lambda x: (x[0], x[1]))
        is_empty = all(False for _ in deepcopy(bndl))
        self.assertFalse(is_empty)
        num_txns = len([(k, grp) for k, grp in bndl])

        pred_num_new_rights = len(sally.buffer[sally.t+1])
        self.assertGreaterEqual(pred_num_new_rights, 1)
        pred_num_new_lefts = 1 + 2*num_txns
        pred_num_red_edges = eff_ring_size*pred_num_new_rights
        pred_num_blue_edges = 2*pred_num_new_rights
        
        out = sally.update_state(1)  # calls make_coinbase and make_txns
        
        new_num_left_nodes = len(sally.g.left_nodes)
        new_num_right_nodes = len(sally.g.right_nodes)
        new_num_blue_edges = len(sally.g.blue_edges)
        new_num_red_edges = len(sally.g.red_edges)
        
        self.assertEqual(len(out), sally.dt)
        self.assertEqual(len(out[0]), 2)
        self.assertIsInstance(out[0][1], list)
        self.assertIn(out[0][0], sally.g.left_nodes)
        
        self.assertEqual(new_num_left_nodes, old_num_left_nodes + pred_num_new_lefts)
        self.assertEqual(new_num_right_nodes, old_num_right_nodes + pred_num_new_rights)
        self.assertEqual(new_num_blue_edges, old_num_blue_edges + pred_num_blue_edges)
        self.assertEqual(new_num_red_edges, old_num_red_edges + pred_num_red_edges)
        
        for txn in out[0][1]:
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

    # @ut.skip("Skipping test_make_simulated_simulator")
    def test_make_simulated_simulator(self):
        sally = make_simulated_simulator()
        self.assertGreater(len(sally.g.left_nodes), 0)
        self.assertGreater(len(sally.g.right_nodes), 0)
        self.assertGreater(sally.runtime, sally.t + 1)

    # @ut.skip("Skipping test_gen_rings_from_simulated")
    def test_gen_rings_from_simulated(self):
        sally = make_simulated_simulator()
        self.assertGreater(len(sally.buffer[sally.t + 1]), 0)
        signing_keys = [_[2] for _ in sally.buffer[sally.t]]
        sally.t += sally.gen_time_step()
        while len(signing_keys) == 0:
            signing_keys = [_[2] for _ in sally.buffer[sally.t]]
            sally.t += sally.gen_time_step()
        self.assertGreater(len(signing_keys), 0)
        out = sally.gen_rings(signing_keys)
        for R, y in zip(out, signing_keys):
            self.assertIn(y, R)
            self.assertIn(y, sally.g.left_nodes)
            for x in R:
                self.assertIn(x, sally.g.left_nodes)

    # @ut.skip("Skipping test_make_txn_from_simulated")
    def test_make_txn_from_simulated(self):
        sally = make_simulated_simulator()
        while sally.t + 1 >= sally.runtime or len(sally.buffer[sally.t + 1]) == 0:
            while sally.t + 1 < sally.runtime and len(sally.buffer[sally.t + 1]) == 0:
                out = sally.make_coinbase()  # out = id of new left node
                self.assertIn(out, sally.g.left_nodes)
                sally.t += 1
            if sally.t + 1 >= sally.runtime:
                sally = make_simulator()

        self.assertTrue(sally.t + 1 < sally.runtime)

        sally.t += 1
        bndl = groupby(sally.buffer[sally.t], key=lambda x: (x[0], x[1]))
        for k, grp in bndl:
            sender = choice(range(len(sally.stochastic_matrix)))
            recip = choice(range(len(sally.stochastic_matrix)))
            out = sally.make_txn(sender, recip, grp)

    # @ut.skip("Skipping test_make_txns_from_simulated")
    def test_make_txns_from_simulated(self):
        sally = make_simulated_simulator()
        while sally.t + 1 < sally.runtime and len(sally.buffer[sally.t + 1]) == 0:
            out = sally.make_coinbase()  # out = id of new left node
            self.assertIn(out, sally.g.left_nodes)
            sally.t += 1
        while sally.t + 1 >= sally.runtime or len(sally.buffer[sally.t + 1]) == 0:
            sally = make_simulator()
            while sally.t + 1 < sally.runtime and len(sally.buffer[sally.t + 1]) == 0:
                out = sally.make_coinbase()  # out = id of new left node
                self.assertIn(out, sally.g.left_nodes)
                sally.t += 1

        out = None  # clear out
        self.assertGreater(len(sally.buffer[sally.t + 1]), 0)
        sally.t += 1
        self.assertGreater(len(sally.buffer[sally.t]), 0)
        bndl = groupby(sally.buffer[sally.t], key=lambda x: (x[0], x[1]))
        k = sum([1 for key, grp in bndl])
        self.assertGreater(k, 0)
        out = sally.make_txns()
        self.assertEqual(len(out), k)
        for ovt in out:
            self.assertEqual(len(ovt), 5)
            new_rights, new_lefts, rings, new_reds, new_blues = ovt
            for new_right in new_rights:
                self.assertIn(new_right, sally.g.right_nodes)
            for new_left in new_lefts:
                self.assertIn(new_left, sally.g.left_nodes)
            for new_red in new_reds:
                self.assertIn(new_red, sally.g.red_edges)
            for new_blue in new_blues:
                self.assertIn(new_blue, sally.g.blue_edges)
            for R, y in zip(rings, new_rights):
                for x in R:
                    self.assertTrue(any([_[0] == x and _[1] == y for _ in new_reds]))
            for new_red in new_reds:
                new_left = new_red[0]
                new_right = new_red[1]
                self.assertTrue(any([x == new_left for ring in rings for x in ring]))
                self.assertIn(new_right, new_rights)

    # @ut.skip("Skipping test_step_from_simulated")
    def test_step_from_simulated(self):
        magic_numbers = [17, 10]
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

    # @ut.skip("Skipping test_update_state_from_simulated")
    def test_update_state_from_simulated(self):
        magic_numbers = [17, 10]
        sally = make_simulated_simulator()
        dt = magic_numbers[1]
        try:
            out = sally.update_state(dt)
        except Exception:
            self.assertTrue(False)
        else:
            self.assertEqual(len(out), dt)
            for x in out:
                self.assertEqual(len(x), 2)
                cb, txns = x
                self.assertIn(cb, sally.g.left_nodes)
                for txn in txns:
                    new_rights, new_lefts, rings, new_reds, new_blues = txn
                    for R, y in zip(rings, new_rights):
                        self.assertIn(y, sally.g.right_nodes)
                        for ring_member in R:
                            self.assertIn(ring_member, sally.g.left_nodes)
                            self.assertTrue(any([red_edge[0] == ring_member and red_edge[1] == y for red_edge in sally.g.red_edges]))
                    for y in new_rights:
                        for z in new_lefts:
                            self.assertIn(z, sally.g.left_nodes)
                            self.assertTrue(any([blue_edge[0] == z and blue_edge[1] == y for blue_edge in sally.g.blue_edges]))
        
    #### REPETITION OF SINGLE-USE TESTS FROM A SIMULATED LEDGER ####
    
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


    # @ut.skip("Skipping test_make_lefts_from_simulated_repeated")
    def test_make_lefts_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_lefts_from_simulated()


    # @ut.skip("Skipping test_make_rights_from_simulated_repeated")
    def test_make_rights_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_rights_from_simulated()


    # @ut.skip("Skipping test_make_reds_from_simulated_repeated")
    def test_make_reds_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_reds_from_simulated()


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

    # @ut.skip("Skipping test_add_left_node_to_buffer_from_simulated_repeated")
    def test_add_left_node_to_buffer_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_add_left_node_to_buffer_from_simulated()

    # @ut.skip("Skipping test_make_coinbase_from_simulated_repeated")
    def test_make_coinbase_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_coinbase_from_simulated()

    # @ut.skip("Skipping test_gen_rings_from_simulated_repeated")
    def test_gen_rings_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_from_simulated()

    # @ut.skip("Skipping test_make_txn_from_simulated_repeated")
    def test_make_txn_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txn_from_simulated()

    # @ut.skip("Skipping test_make_txns_from_simulated_repeated")
    def test_make_txns_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txns_from_simulated()
            
    # @ut.skip("Skipping test_update_state_from_simulated_repeated")
    def test_update_state_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_update_state_from_simulated()

    @ut.skip("Skipping test_run_from_simulated_repeated")
    def test_run_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_run_from_simulated()

    # @ut.skip("Skipping test_make_simulated_simulator_repeated")
    def test_make_simulated_simulator_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_simulated_simulator()

    # @ut.skip("Skipping test_gen_rings_from_simulated_repeated")
    def test_gen_rings_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_gen_rings_from_simulated()

    # @ut.skip("Skipping test_make_txn_from_simulated_repeated")
    def test_make_txn_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txn_from_simulated()

    # @ut.skip("Skipping test_make_txns_from_simulated_repeated")
    def test_make_txns_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_make_txns_from_simulated()

    # @ut.skip("Skipping test_step_from_simulated_repeated")
    def test_step_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_step_from_simulated()

    # @ut.skip("Skipping test_update_state_from_simulated_repeated")
    def test_update_state_from_simulated_repeated(self):
        for _ in range(SAMPLE_SIZE):
            self.test_update_state_from_simulated()

    #### INTEGRATION TESTS ####
    
    # @ut.skip("Skipping test_run_from_empty")
    def test_run_from_empty(self):
        sally = make_simulator()
        try:
            out = sally.run()
        except:
            self.assertTrue(False)
        else:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), sally.runtime)
            for step in out:
                for block in step:
                    cb, txns = block
                    self.assertIn(cb, sally.g.left_nodes)
                    for txn in txns:
                        [new_rights, new_lefts, rings, new_reds, new_blues] = txn
                        for new_right in new_rights:
                            self.assertIn(new_right, sally.g.right_nodes)
                        for new_left in new_lefts:
                            self.assertIn(new_left, sally.g.left_nodes)
                        for new_red in new_reds:
                            self.assertIn(new_red, sally.g.red_edges)
                        for new_blue in new_blues:
                            self.assertIn(new_blue, sally.g.blue_edges)
                        for R, y in zip(rings, new_rights):
                            for x in R:
                                self.assertTrue(any([edge_id[0] == x and edge_id[1] == y for edge_id in sally.g.red_edges]))
    
    @ut.skip("Skipping test_run_from_simulated")
    def test_run_from_simulated(self):
        sally = make_simulated_simulator()
        try:
            out = sally.run()
        except:
            self.assertTrue(False)
        else:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), sally.runtime)
            for step in out:
                for block in step:
                    cb, txns = block
                    self.assertIn(cb, sally.g.left_nodes)
                    for txn in txns:
                        [new_rights, new_lefts, rings, new_reds, new_blues] = txn
                        for new_right in new_rights:
                            self.assertIn(new_right, sally.g.right_nodes)
                        for new_left in new_lefts:
                            self.assertIn(new_left, sally.g.left_nodes)
                        for new_red in new_reds:
                            self.assertIn(new_red, sally.g.red_edges)
                        for new_blue in new_blues:
                            self.assertIn(new_blue, sally.g.blue_edges)
                        for R, y in zip(rings, new_rights):
                            for x in R:
                                self.assertTrue(any([edge_id[0] == x and edge_id[1] == y for edge_id in sally.g.red_edges]))

ut.TextTestRunner(verbosity=2, failfast=True).run(ut.TestLoader().loadTestsFromTestCase(TestSimulator))
