import random
import math
import numpy as np

from graphtheory import *
from collections import deque
from copy import deepcopy

class Simulator(object):
    par = None
    def __init__(self, par = None):
        if par is None:
            par = {'max_height': 1000}
        self.G = BipartiteGraph()
        self.tainted_keys = []
        self.max_height = par['max_height']
        self.lock_time = 15
        self.h = 0
        self.spend_buffer = [[]]*self.max_height
        self.blockChain = [[[], []]]*self.max_height
        self.ring_size = 11
        self.mode = "uniform"
        # self.blockChain[h][0] - left_nodes at height h (signatures)
        # self.blockChain[h][1] - right_nodes at height h (one-time keys)
        self.shape = 1.0
        self.scale = 1.0
        self.timescale = None

    def create_ledger(self):
        rejected = False
        while self.h < self.max_height and not rejected:
            # call _spend for each block height or until we receive a rejected flag.
            rejected = rejected or self._spend()
            assert not rejected
            self.h += 1
        self._write_ledger_to_file()
        return rejected

    def _write_ledger_to_file(self):
        pass

    def _spend(self):
        rejected = False

        # Create coinbase transaction
        nums = self.get_ins_and_outs()
        rejected = rejected or self._sign(nums[1])
        assert not rejected

        # Create tainted txns
        tainted = [x for x in self.spend_buffer[h] if x in self.tainted_keys]
        while len(tainted) > 0 and not rejected:
            nums = self.get_ins_and_outs()
            num_inputs_in_next_txn = max(1, min(nums[0], len(tainted)))  # pick number of input keys for next taintxn.
            inputs_in_next_txn = deepcopy(tainted[:num_inputs_in_next_txn])
            tainted = deepcopy(tainted[num_inputs_in_next_txn:])
            rejected = rejected or self._sign(nums[1], True, inputs_in_next_txn)
            assert not rejected

        # Create untainted txns
        not_tainted = [x for x in self.spend_buffer[h] if x not in self.tainted_keys]
        while len(not_tainted) > 0 and not rejected:
            nums = self.get_ins_and_outs()  # generate number of inputs and outptus for next txn.
            num_inputs_in_next_txn = max(1, min(nums[0], len(not_tainted)))  # pick number of input keys for next taintxn.
            inputs_in_next_txn = deepcopy(not_tainted[:num_inputs_in_next_txn])
            not_tainted = deepcopy(not_tainted[num_inputs_in_next_txn:])
            rejected = rejected or self._sign(nums[1], False, inputs_in_next_txn)
            assert not rejected

        return rejected

    def _sign(self, numOuts, taint, to_be_spent=[],):
        ''' This function is badly named. This function takes a number of new outputs and a list of inputs to be spent,
        adds them to the bipartite graph, selects ring members, and draws edges. A better name for this function
        could be AddTxnToLedger or something.'''
        assert taint is True or taint is False
        # Construct output indices
        out_idxs = []
        for i in range(numOuts):
            out_idxs.append(self.G.add_node(0))
        self.blockChain[self.h][0] += out_idxs

        # Construct signature indices
        sig_idxs = []
        for onetimekey in to_be_spent:
            # Construct a ring for this key.
            ring = self._get_ring() + [onetimekey]

            # Add a right-node associated with this ring signature.
            sig_idx = self.G.add_node(1)
            sig_idxs += [sig_idx]
            # Add red edges
            for otk in ring:
                self.G.add_edge(1, (otk, sig_idx))
        self.blockChain[self.h][1] += sig_idxs

        # Add blue edges
        for out_idx in out_idxs:
            for sig_idx in sig_idxs:
                self.G.add_edge(0, (out_idx, sig_idx))

        # For each out_idx, pick a spend-time h and place into the spend-buffer.
        if taint:
            for out_idx in out_idxs:
                t = max(self.lock_time, self.getTaintDelay())
                self.spend_buffer[self.h + t] += [out_idx]
        else:
            for out_idx in out_idxs:
                t = max(self.lock_time, self.getWalletDelay())
                self.spend_buffer[self.h + t] += [out_idx]

    def _get_ring(self):
        result = []
        if self.mode == "uniform":
            result = random.sample(list(self.G.left_nodes.keys()), self.ring_size - 1)
        return result

    def get_ins_and_outs(self):
        # Pick number of inputs, numbers[0], and number of outputs, numbers[1], from an empirical distribution.
        # To randomize, pick numbers[0] from Poisson(1) and numbers[1] from Poisson(2) - this is a biased model, skewed
        # small compared to the empirical distribution, and has a light tail. Two models that are unbiased but still
        # have skew and a light tail are:
        #   (Poisson(5.41), Poisson(6.14)) - Maximum likelihood estimate
        #   (NegBinom(71.900, 0.070), NegBinom(38.697,0.137)) - Method of moments
        numbers = [1, 2]  # For now, we will do this deterministically and we will put the empirical distro in later.
        return numbers

    def getWalletDelay(self):
        # Pick a block height N+1, N+2, N+3, ... from a gamma distribution (continuous, measured in seconds) divided by
        # 120, take the ceiling fcn, but this distribution is clipped to not go below N
        x = max(self.N, math.ceil(math.exp(np.random.gamma(self.shape, self.scale)) / self.timescale))
        while x > self.T:
            x = max(self.N, math.ceil(math.exp(np.random.gamma(self.shape, self.scale)) / self.timescale))
        return x

    def getTaintDelay(self, ratio=1.0):
        # Pick a block height N+1, N+2, N+3, ... from any other distribution. Vary for testing purposes.

        # PERIODIC DAILY PURCHASE BEHAVIOR:
        x = 720

        # UNIFORM DAILY CHURNING:
        # x = random.randint(N,720)

        # GAMMA WITH SAME SHAPE BUT SMALLER RATE PARAM = SLOWER SPEND TIME
        # ratio = 0.5
        # x = min(self.N, math.ceil(math.exp(np.random.gamma(self.shape, self.scale/ratio))/self.timescale))

        # GAMMA WITH SAME SPEND TIME BUT GREATER SHAPE = SLOWER SPEND TIME
        # ratio = 2.0
        # x = min(self.N, math.ceil(math.exp(np.random.gamma(ratio*self.shape, self.scale))/self.timescale))
        # if x > self.T:
        #    x = None
        return x
