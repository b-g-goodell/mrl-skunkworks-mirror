import random
import math
import numpy as np

from graphtheory import *
from collections import deque
from copy import deepcopy

class Simulator(object):
    par = None

    def __init__(self, par = None):
        self.G = BipartiteGraph()
        self.alice_keys = []
        self.alice_sigs = []
        self.max_height = par['max_height']
        self.lock_time = 15
        self.h = 0
        self.pending_txns = [[]]*self.max_height
        self.blockChain = [[[], []]]*self.max_height
        # self.blockChain[h][0] - left_nodes at height h (signatures)
        # self.blockChain[h][1] - right_nodes at height h (one-time keys)
        self.ring_size = 11
        self.mode = "uniform"
        self.alice_hashrate = 0.01
        self.eve_hashrate = 0.32
        self.churn_number = 3
        self.corruption_rate = 0.2  # for what % many outputs is Eve the recipient?

    def run(self):
        rejected = False
        while self.h < self.max_height and not rejected:
            rejected = rejected or self._create_coinbase()
            assert not rejected
            rejected = rejected or self._spend()
            assert not rejected
            self.h += 1
        rejected = rejected or self._write_ledger_to_file()
        assert not rejected
        return rejected

    def _create_coinbase(self):
        u = random.random()
        if u < self.alice_hashrate:
            x = self.G.add_node(0)
            self.alice_keys += [(self.h, x)]
            a = self.pick_an_alice_age()
            self.pending_txns[self.h + a] += [(self.h, x)]
        else:
            x = self.G.add_node(0)
            a = self.pick_a_background_age()
            self.pending_txns[self.h + a] += [(self.h, x)]

    def _spend(self):
        alice_keys_to_spend = [pair for pair in self.pending_txns[self.h] if pair in self.alice_keys]
        background_keys_to_spend = [pair for pair in self.pending_txns[self.h] if pair not in self.alice_keys]

        while len(alice_keys_to_spend) > 0:
            num_ins_and_outs = self._get_numbers()
            num_ins = min(len(alice_keys_to_spend), num_ins_and_outs["ins"])
            num_outs = num_ins_and_outs["outs"]
            next_ins = deepcopy(alice_keys_to_spend[:num_ins])
            alice_keys_to_spend = deepcopy(alice_keys_to_spend[num_ins:])

            # for each input in next_ins, add a right node to G, pick a ring, add red edges
            these_sigs = []
            for next_in in next_ins:
                sig_idx = self.G.add_node(1)
                these_sigs += [sig_idx]
                ring_members = self._get_ring() + [next_in]
                for rm in ring_members:
                    self.G.add_edge(1, (rm, sig_idx), 0.0)

            # Add num_outs new left nodes to G
            these_outs = []
            for i in range(num_outs):
                these_outs += [self.G.add_node(0)]

            # Add blue edges from each out in these_outs to each sig in these_sigs
            for left_node in these_outs:
                for right_node in these_sigs:
                    self.G.add_edge(0, (left_node, right_node), 0.0)

    def _get_ring(self):
        return []

    def _get_numbers(self):
        return {"ins": 2, "outs": 2}

    def _write_ledger_to_file(self):
        pass
