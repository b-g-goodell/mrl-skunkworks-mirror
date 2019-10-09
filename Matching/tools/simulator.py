from itertools import groupby
from graphtheory import *
from math import *
from random import *
from copy import deepcopy

MAX_MONERO_ATOMIC_UNITS = 2**64 - 1
DECAY_RATIO = 2**-18
MIN_MINING_REWARD = 6e11

class Simulator(object):
    def __init__(self, par=None):
        assert par is not None
        assert isinstance(par, dict)
        assert 'runtime' in par
        assert isinstance(par['runtime'], int)
        assert par['runtime'] > 0
        assert 'filename' in par
        assert isinstance(par['filename'], str)
        assert 'stochastic matrix' in par
        assert isinstance(par['stochastic matrix'], list)
        for row in par['stochastic matrix']:
            assert isinstance(row, list)
            for elt in row:
                assert isinstance(elt, float)
                assert 0.0 <= elt <= 1.0
            # print(row, sum(row))
            assert sum(row) == 1.0
        assert 'hashrate' in par
        assert isinstance(par['hashrate'], list)
        for elt in par['hashrate']:
            assert isinstance(elt, float)
            assert 0.0 <= elt <= 1.0
        assert sum(row) == 1.0
        assert 'spendtimes' in par
        assert isinstance(par['spendtimes'], list)
        for elt in par['spendtimes']:
            assert callable(elt)
        assert 'min spendtime' in par
        assert isinstance(par['min spendtime'], int)
        assert par['min spendtime'] > 0
        assert 'ring size' in par
        assert isinstance(par['ring size'], int)
        assert par['ring size'] > 0

        self.runtime = par['runtime']  # int, positive
        self.fn = par['filename']  # str
        self.stoch_matrix = par['stochastic matrix']  # list
        self.hashrate = par['hashrate']  # list
        self.minspendtime = 10
        self.spendtimes = par['spendtimes'] # dict with lambda functions (PMFs with support on minspendtime, minspendtime+1, minspendtime+2, ...)
        self.ringsize = par['ring size'] # int, positive
        if 'ring selection mode' not in par or par['ring selection mode'] is None:
            self.mode = "uniform" 
        else:
            self.mode = par['ring selection mode'] # str, "uniform" or "monerolink"
        self.buffer = [[]]*self.runtime
        self.ownership = dict()
        self.amounts = dict()
        self.g = BipartiteGraph()
        self.t = 0
        self.dummy_monero_mining_flat = False  # whether constant block reward has started
        open(self.fn, "w+").close()
        self.last_mining_reward = None
        self.reporting_modulus = 10 # Write to file each time these many blocks have been added

    def run(self):
        while self.t < self.runtime:
            self.make_coinbase()
            self.spend_from_buffer()
            self.report()
            self.t += 1

    def make_coinbase(self):
        owner = self.pick_coinbase_owner()
        amt = self.pick_coinbase_amt()
        dt = self.pick_spend_time(owner)
        recip = self.pick_next_recip(owner)
        node_to_spend = self.g.add_node(0)
        self.ownership[node_to_spend] = owner
        if self.t + dt < len(self.buffer):
            self.buffer[self.t + dt] += [(owner, recip, node_to_spend)]  # at block self.t + dt, owner will send new_node to recip
        self.amounts[node_to_spend] = amt
        assert node_to_spend in self.amounts and self.amounts[node_to_spend] == amt
        return node_to_spend, dt
         
    def spend_from_buffer(self):
        # results in txn_bundles, which is a list:
        #   txn_bundles[i] = ith txn included in block with height self.t
        #       txn_bundles[i][0] = list of keys (left node idents) being spent in this txn.
        #       txn_bundles[i][1] = total amount of input keys in this txn.
        #       txn_bundles[i][2] = signature nodes (right node idents) corresponding to keys being spent
        #       txn_bundles[i][3] = pair of a left node ident and an amount corresponding to change
        #       txn_bundles[i][4] = pair of a left node ident and an amount corresponding to the payment amount
        txn_bundles = [] 
        to_spend = sorted(self.buffer[self.t], key=lambda x: (x[0], x[1]))
        for k, grp in groupby(to_spend, key=lambda x: (x[0], x[1])):
            temp = deepcopy(grp)
            temp = deepcopy(grp)
            keys_to_spend = [x[2] for x in temp]
            txn_bundles += [[keys_to_spend]]
            temp = deepcopy(grp)
            assert len(keys_to_spend) > 0
            assert sum([self.amounts[x] for x in keys_to_spend]) > len(keys_to_spend)*MIN_MINING_REWARD
            tot_amt = sum([self.amounts[x] for x in keys_to_spend])
            txn_bundles[-1] += [tot_amt]
            sig_nodes = []
            for x in temp:
                sig_nodes += [self.g.add_node(1)] # add a right_node and assign ownership
                y = sig_nodes[-1]
                self.ownership[y] = k[0]
                for idx in self.get_ring(x[2]): 
                    eid = (idx, y)
                    self.g.add_edge(1, eid, 1.0)  # adds red edge
            txn_bundles[-1] += [sig_nodes]

            change_node = self.g.add_node(0)  # add a left_node and assign ownership
            self.ownership[change_node] = k[0]
            recipient_node = self.g.add_node(0)  # add a left_node and assign ownership
            self.ownership[recipient_node] = k[1]

            for snode in sig_nodes:
                eid = (recipient_node, snode)
                self.g.add_edge(0, eid, 1.0)  # adds blue edge
                eidd = (change_node, snode)
                self.g.add_edge(0, eidd, 1.0)  # adds blue edge
            
            change = random()*tot_amt
            self.amounts[change_node] = change
            txn_bundles[-1] += [(change_node, change)]
            txn_amt =  tot_amt - change
            self.amounts[recipient_node] = txn_amt
            txn_bundles[-1] += [(recipient_node, txn_amt)]
        return txn_bundles

    def report(self):
        line = "\n\n\n\nREPORTING FOR TIMESTEP" + str(self.t) + "\n\n"
        line += "LEFT NODES OF G AND OWNERSHIP AND AMOUNTS\n"
        ct = 0
        for node_idx in self.g.left_nodes:
            line += str((node_idx, self.ownership[node_idx], self.amounts[node_idx]))
            ct += 1
            if len(self.g.left_nodes) > ct:
                line += ","
        line += "\n\nRIGHT NODES OF G AND OWNERSHIP\n"
        ct = 0
        for node_idx in self.g.right_nodes:
            line += str((node_idx, self.ownership[node_idx]))
            ct += 1
            if len(self.g.right_nodes) > ct:
                line += ","
        with open(self.fn, "a") as wf:
            wf.write(line + "\n\n\n")

    def pick_coinbase_owner(self):
        i = 0
        r = random()
        u = self.hashrate[i]
        found = (u >= r)
        while not found and i < len(self.hashrate):
            u += self.hashrate[i]
            i += 1
            found = (u >= r)
        assert found
        return i
            
    def pick_coinbase_amt(self):
        ''' This function starts with a max reward, multiplies the last mining reward by a given decay ratio, until you hit a minimum. WARNING: If Simulator is not run timestep-by-timestep, i.e. if any timepoints t=0, 1, 2, ... are skipped, then the coinbase reward will be off; to fix this, we could use self.t and just return min(MAX_MONERO_ATOMIC_UNITS, max(MIN_MINING_REWARD, DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS(1.0 - DECAY_RATIO)**(self.t - 1))), but this will do for now.'''
        if self.t==0:
            self.last_mining_reward = DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS
        elif self.t > 0:
            if not self.dummy_monero_mining_flat:
                self.last_mining_reward = (1.0-DECAY_RATIO)*self.last_mining_reward
                if self.last_mining_reward < MIN_MINING_REWARD: # minimum monero mining reward is 0.6 XMR or 6e11 atomic units
                    self.last_mining_reward = MIN_MINING_REWARD
                    self.dummy_monero_mining_flat = True
            else:
                self.last_mining_reward = MIN_MINING_REWARD
        return self.last_mining_reward

    def pick_spend_time(self, owner):
        i = self.minspendtime # Minimal spend-time
        r = random()
        u = self.spendtimes[owner](i)
        found = (u >= r)
        while not found:
            u += self.spendtimes[owner](i)
            i += 1
            found = (u >= r)
        assert found
        return i

    def pick_next_recip(self, owner):
        i = 0
        r = random()
        u = self.stoch_matrix[owner][i]
        found = (u >= r)
        while not found and i < len(self.stoch_matrix[owner]):
            i += 1
            u += self.stoch_matrix[owner][i]
            found = (u >= r)
        assert found
        return i

    def get_ring(self, spender):
        k = list(self.g.left_nodes.keys())
        if self.mode == "uniform":
            ss = min(max(0,len(self.g.left_nodes)-1), self.ringsize-1)
            ring = sample(k, ss)
            while spender in ring:
                idx_of_spender = ring.index(spender)
                while ring[idx_of_spender] == spender:
                    ring[idx_of_spender] = choice(k)
            ring += [spender]
        return ring


        

