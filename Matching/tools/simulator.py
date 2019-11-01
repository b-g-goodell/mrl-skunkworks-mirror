from itertools import groupby
from graphtheory import *
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
        assert sum(par['hashrate']) == 1.0
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
        self.minspendtime = par['min spendtime']
        # dict with lambda functions 
        # PMFs with support on minspendtime, minspendtime + 1, ...
        self.spendtimes = par['spendtimes'] 
        self.ringsize = par['ring size']  # int, positive
        if 'ring selection mode' not in par or \
                par['ring selection mode'] is None:
            self.mode = "uniform" 
        else:
            # str, "uniform" or "monerolink"
            self.mode = par['ring selection mode'] 
        self.buffer = []
        for i in range(self.runtime):
            self.buffer += [[]]
        self.ownership = dict()
        self.amounts = dict()
        self.g = BipartiteGraph()
        self.t = 0
        # whether constant block reward has started
        self.dummy_monero_mining_flat = False  
        open(self.fn, "w+").close()
        self.last_mining_reward = None
        # Write to file each time these many blocks have been added
        self.reporting_modulus = par['reporting modulus']

    def halting_run(self):
        if self.t < self.runtime:
            self.make_coinbase()
            self.spend_from_buffer()
            self.t += 1

    def run(self):
        while self.t < self.runtime:
            self.make_coinbase()
            self.spend_from_buffer()
            self.t += 1

    def make_coinbase(self):
        owner = self.pick_coinbase_owner()
        assert owner in range(len(self.hashrate))
        amt = self.pick_coinbase_amt()
        dt = self.pick_spend_time(owner)
        assert isinstance(dt, int)
        assert 0 < dt
        recip = self.pick_next_recip(owner)

        node_to_spend = self.g.add_node(0, self.t)
        self.ownership[node_to_spend] = owner

        s = self.t + dt
        if s < len(self.buffer):
            # at block s, owner will send new_node to recip
            self.buffer[s].append((owner, recip, node_to_spend))
            assert s > self.t
        # assert node_to_spend in self.amounts 
        self.amounts[node_to_spend] = amt
        return node_to_spend, dt

    def spend_from_buffer(self):
        txn_bundles = []
        to_spend = sorted(self.buffer[self.t], key=lambda x: (x[0], x[1]))
        ct = 0
        num_left_nodes = len(self.g.left_nodes)
        for k, grp in groupby(to_spend, key=lambda x: (x[0], x[1])):
            txn_bundles += [self.new_ring_sigs(k, grp)]
            ct += 1
        assert num_left_nodes + 2*ct == len(self.g.left_nodes)
        return txn_bundles

    def new_ring_sigs(self, k, grp):
        temp = deepcopy(grp)
        left_nodes_being_spent = [x[2] for x in temp]
        tot_amt = sum([self.amounts[x] for x in left_nodes_being_spent])

        new_right_nodes = []
        rings = {}
        for left_node in left_nodes_being_spent:
            new_right_nodes += [self.g.add_node(1, self.t)]
            self.ownership[new_right_nodes[-1]] = (k[0], left_node)
            rings[new_right_nodes[-1]] = self.get_ring(left_node)

        change_node = self.g.add_node(0, self.t)  
        self.ownership[change_node] = k[0]

        recipient_node = self.g.add_node(0, self.t)  
        self.ownership[recipient_node] = k[1]

        change = random()*tot_amt
        txn_amt = tot_amt - change

        self.amounts[change_node] = change
        self.amounts[recipient_node] = txn_amt

        blue_eid, blue_eidd, red_eids = self.get_eids(recipient_node,
                                                      change_node,
                                                      self.ownership,
                                                      self.t, self.g,
                                                      new_right_nodes, rings)

        result = [left_nodes_being_spent, tot_amt, new_right_nodes,
                  (change_node, change), (recipient_node, txn_amt),
                  [blue_eid, blue_eidd], red_eids]
        return result

    def spnd_from_buffer(self):
        # results in txn_bundles, which is a list:
        #   txn_bundles[i] = ith txn included in block with height self.t
        #       txn_bundles[i][0] = list of keys (left node idents) being spent
        #       txn_bundles[i][1] = total amount of input keys in this txn.
        #       txn_bundles[i][2] = signature nodes (right node idents)
        #       txn_bundles[i][3] = pair of a left node ident and change amount
        #       txn_bundles[i][4] = pair of a left node ident and paym amount
        txn_bundles = [] 
        to_spend = sorted(self.buffer[self.t], key=lambda x: (x[0], x[1]))
        # print(to_spend)
        ct = 0
        for k, grp in groupby(to_spend, key=lambda x: (x[0], x[1])):
            # for each k, grp, we:
            #     add new right nodes (grp size)
            #     add two new left nodes
            #     add (grp size)*2 new blue edges
            #     add (grp size)*eff_ringsize new red edges
            temp = deepcopy(grp)
            keys_to_spend = [x[2] for x in temp]
            txn_bundles += [keys_to_spend] 
            tot_amt = sum([self.amounts[x] for x in keys_to_spend])
            txn_bundles[-1] += [tot_amt]

            temp = deepcopy(grp)
            new_right_nodes = []
            rings = dict()
            for x in temp:
                new_right_nodes += [self.g.add_node(1, self.t)]
                self.ownership[new_right_nodes[-1]] = (k[0], x[2])
                rings[new_right_nodes[-1]] = self.get_ring(x[2])
            txn_bundles[-1] += [new_right_nodes]
            # txn_bundles[-1] = [keys_to_spend, tot_amt, sig_nodes]

            change_node = self.g.add_node(0, self.t)  
            self.ownership[change_node] = k[0]

            recipient_node = self.g.add_node(0, self.t)  
            self.ownership[recipient_node] = k[1]

            for rnode in new_right_nodes:
                pair = (recipient_node, rnode)
                blue_eid = self.g.add_edge(0, pair, 1.0, self.t)
                self.ownership[blue_eid] = self.ownership[blue_eid[0]]

                pairr = (change_node, rnode)
                blue_eidd = self.g.add_edge(0, pairr, 1.0, self.t)
                self.ownership[blue_eidd] = self.ownership[blue_eidd[0]]

                red_eids = []
                for ring_member in rings[rnode]:
                    pairrr = (ring_member, rnode)
                    new_eid = self.g.add_edge(1, pairrr, 1.0, self.t)
                    red_eids += [new_eid]
                    self.ownership[new_eid] = self.ownership[new_eid[1]]
                    ct += 1

            change = random()*tot_amt
            self.amounts[change_node] = change
            txn_bundles[-1] += [(change_node, change)]
            txn_amt = tot_amt - change
            self.amounts[recipient_node] = txn_amt
            txn_bundles[-1] += [(recipient_node, txn_amt)]

        return txn_bundles

    def report(self):
        line = "\n\n\n\nREPORTING FOR TIMESTEP" + str(self.t) + "\n\n"
        line += "LEFT NODES OF G AND OWNERSHIP AND AMOUNTS\n"
        ct = 0
        for node_idx in self.g.left_nodes:
            temp = (node_idx, self.ownership[node_idx], self.amounts[node_idx])
            line += str(temp)
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
        with open(self.fn, "w+") as wf:
            wf.write(line + "\n\n\n")

    def pick_coinbase_owner(self):
        r = random()
        u = 0
        i = 0
        found = False
        while i < len(self.hashrate):
            u += self.hashrate[i]
            if u > r:
                found = True
                break
            else:
                i += 1
        assert found
        assert i in range(len(self.hashrate))
        return i
            
    def pick_coinbase_amt(self):
        """ This function starts with a max reward, multiplies the last mining 
        reward by a given decay ratio, until you hit a minimum. WARNING: If 
        Simulator is not run timestep-by-timestep, i.e. if any timepoints 
        t=0, 1, 2, ... are skipped, then the coinbase reward will be off.
        """
        if self.t == 0:
            self.last_mining_reward = DECAY_RATIO*MAX_MONERO_ATOMIC_UNITS
        elif self.t > 0:
            if not self.dummy_monero_mining_flat:
                self.last_mining_reward = 1.0-DECAY_RATIO
                self.last_mining_reward *= self.last_mining_reward
                if self.last_mining_reward < MIN_MINING_REWARD:
                    self.last_mining_reward = MIN_MINING_REWARD
                    self.dummy_monero_mining_flat = True
            else:
                self.last_mining_reward = MIN_MINING_REWARD
        return self.last_mining_reward

    def pick_spend_time(self, owner):
        try:
            assert owner in range(len(self.stoch_matrix))
        except AssertionError:
            print("Owner not found in stochastic matrix. owner = " + str(owner))
        i = self.minspendtime  # Minimal spend-time
        r = random()
        # spend times gend support on min_spendtime, min_spendtime + 1, ...
        u = self.spendtimes[owner](i)
        found = (u >= r)
        while not found and i < self.runtime:
            i += 1
            u += self.spendtimes[owner](i)
            found = (u >= r)
        assert found or i >= self.runtime
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
        ring = []
        available_keys = [x for x in list(self.g.left_nodes.keys()) if
                          x != spender]
        if self.mode == "uniform":
            ring = sample(available_keys, min(len(self.g.left_nodes),
                                              self.ringsize) - 1)
            ring += [spender]
        return ring

    @staticmethod
    def get_eids(recipient_node, change_node, ownership, t, g, new_right_nodes,
                 rings):
        for rnode in new_right_nodes:
            pair = (recipient_node, rnode)
            blue_eid = g.add_edge(0, pair, 1.0, t)  # adds blue edge
            ownership[blue_eid] = ownership[blue_eid[0]]

            pairr = (change_node, rnode)
            blue_eidd = g.add_edge(0, pairr, 1.0, t)  # adds blue edge
            ownership[blue_eidd] = ownership[blue_eidd[0]]

            red_eids = []
            for ring_member in rings[rnode]:
                pairrr = (ring_member, rnode)
                red_eids += [g.add_edge(1, pairrr, 1.0, t)]  # adds red edge
                ownership[red_eids[-1]] = ownership[red_eids[-1][1]]
        return blue_eid, blue_eidd, red_eids

