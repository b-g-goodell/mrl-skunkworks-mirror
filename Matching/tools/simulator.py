from itertools import groupby
from graphtheory import *
from random import *
from copy import deepcopy

MAX_MONERO_ATOMIC_UNITS = 2**64 - 1
EMISSION_RATIO = 2**-18
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
        self.next_mining_reward = EMISSION_RATIO*MAX_MONERO_ATOMIC_UNITS
        # Write to file each time these many blocks have been added
        self.reporting_modulus = par['reporting modulus']

    def halting_run(self):
        if self.t + 1 < self.runtime:
            self.t += 1
            self.make_coinbase()
            self.spend_from_buffer()

    def run(self):
        while self.t + 1 < self.runtime:
            self.t += 1
            self.make_coinbase()
            self.spend_from_buffer()

    def make_coinbase(self):
        owner = self.pick_coinbase_owner()
        assert owner in range(len(self.hashrate))
        amt = self.pick_coinbase_amt()
        dt = self.pick_spend_time(owner)
        assert isinstance(dt, int)
        assert 0 < dt
        recip = self.pick_next_recip(owner)

        # print("Making coinbase.")
        tomato = len(self.g.left_nodes)
        node_to_spend = self.g.add_node(0, self.t)
        # print("Coinbase with ident " + str(node_to_spend) + " created.")
        assert len(self.g.left_nodes) == tomato + 1
        self.ownership[node_to_spend] = owner

        s = self.t + dt
        if s < len(self.buffer):
            # at block s, owner will send new_node to recip
            self.buffer[s].append((owner, recip, node_to_spend))
            assert s > self.t
        # assert node_to_spend in self.amounts 
        self.amounts[node_to_spend] = amt
        return node_to_spend, dt

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

    def spend_from_buffer(self):
        """ spend_from_buffer groups the buffer by sender-recipient pairs.
        For each of these, amounts are decided, new right nodes are created,
        new left nodes are created, new red edges are created, and new blue
        edges are created.

        For convenience, we return a summary.

        :return: summary (list)
                 summary[i] = ith txn included in block with height self.t
                        idx : entry
                          0 : left nodes being spent
                          1 : sum of amounts of left nodes being spent
                          2 : right nodes being created
                          3 : pair left node and amt (change)
                          4 : pair left node and amt (recipient)

        """
        summary = []
        s = None

        old_lnids = len(self.g.left_nodes)
        old_rnids = len(self.g.right_nodes)
        old_reids = len(self.g.red_edges)
        old_beids = len(self.g.blue_edges)

        # print("Bundling transactions by sender-receiver pair")
        bndl = groupby(self.buffer[self.t], key=lambda x: (x[0], x[1]))

        # Make some predictions
        # print("Making predictions of how many graphs and edges are to be
        # added.")
        right_nodes_to_be_added = len(self.buffer[self.t])
        num_txn_bundles = sum([1 for _ in deepcopy(bndl)])
        num_true_spenders = sum([1 for k, grp in deepcopy(bndl) for _ in grp])
        assert num_true_spenders == right_nodes_to_be_added

        blue_edges_to_be_added = 2 * right_nodes_to_be_added
        left_nodes_to_be_added = 2 * num_txn_bundles  # Coinbase elsewhere
        red_edges_per_sig = min(old_lnids, self.ringsize)
        red_edges_to_be_added = red_edges_per_sig * right_nodes_to_be_added

        ct = 0

        if len(self.buffer[self.t]) > 0:
            # print("Beginning spend from buffer")

            new_lnids = len(self.g.left_nodes)
            new_rnids = len(self.g.right_nodes)
            new_reids = len(self.g.red_edges)
            new_beids = len(self.g.blue_edges)

            tot_ring_membs = 0
            rings = dict()
            new_right_nodes = []
            # TODO: FIX ORDER IN WHICH NODES AND EDGES ARE BEING ADDED
            for k, grp in bndl:
                # Add new nodes in this loop.

                # k = (sender, recipient) stochastic_matrix indices
                # grp = iterator of left node indices.
                # kk = deepcopy(k)
                # ggrp = deepcopy(grp)
                # print("Working with k = " + str(kk))
                # print("Working with grp = " + str([_ for _ in grp]))

                # Collect keys to be spent in this group.
                temp = deepcopy(grp)
                keys_to_spend = [x[2] for x in temp]
                summary += [[keys_to_spend]]

                # Compute amount for those keys.
                # print("Computing amounts")
                tot_amt = sum([self.amounts[x] for x in keys_to_spend])
                summary[-1] += [tot_amt]

                # Create new right node for each key being spent and generate a
                # ring for that node.
                # print("Creating new right nodes for this grp.")
                temp = deepcopy(grp)
                for x in temp:
                    # Pick ring members.

                    temp_ring = self.get_ring(x[2])
                    assert len(temp_ring) == red_edges_per_sig
                    rings[new_right_nodes[-1]] = temp_ring

                temp = deepcopy(grp)
                for x in temp:
                    # Add new right nodes and set ownership.
                    new_right_nodes += [self.g.add_node(1, self.t)]
                    self.ownership[new_right_nodes[-1]] = (k[0], x[2])

                    # Select ring members for each key in the group
                    # print("Picking rings for " + str(x[2]))

                    assert len(self.g.left_nodes) == new_lnids
                    assert len(self.g.right_nodes) == new_rnids + 1
                    assert len(self.g.red_edges) == new_reids
                    assert len(self.g.blue_edges) == new_beids

                    new_lnids = len(self.g.left_nodes)
                    new_rnids = len(self.g.right_nodes)
                    new_reids = len(self.g.red_edges)
                    new_beids = len(self.g.blue_edges)

                summary[-1] += [new_right_nodes]

                # Create two new left nodes
                # print("Creating two new left nodes for this group")
                change_node = self.g.add_node(0, self.t)
                self.ownership[change_node] = k[0]
                recipient_node = self.g.add_node(0, self.t)
                self.ownership[recipient_node] = k[1]

                assert len(self.g.left_nodes) == new_lnids + 2
                assert len(self.g.right_nodes) == new_rnids
                assert len(self.g.red_edges) == new_reids
                assert len(self.g.blue_edges) == new_beids

                new_lnids = len(self.g.left_nodes)
                new_rnids = len(self.g.right_nodes)
                new_reids = len(self.g.red_edges)
                new_beids = len(self.g.blue_edges)

            # Now we add edges
            assert len(list(set(new_right_nodes))) == len(new_right_nodes)

            for rnode in new_right_nodes:
                # Add blue edges from each new right node to each new left node
                # print("Adding blue edge for recipient")
                pair = (recipient_node, rnode)
                blue_eid = self.g.add_edge(0, pair, 1.0, self.t)
                self.ownership[blue_eid] = self.ownership[blue_eid[0]]

                # print("Adding blue edge for change")
                pairr = (change_node, rnode)
                blue_eidd = self.g.add_edge(0, pairr, 1.0, self.t)
                self.ownership[blue_eidd] = self.ownership[blue_eidd[0]]

                assert len(self.g.left_nodes) == new_lnids
                assert len(self.g.right_nodes) == new_rnids
                assert len(self.g.red_edges) == new_reids
                assert len(self.g.blue_edges) == new_beids + 2

                new_lnids = len(self.g.left_nodes)
                new_rnids = len(self.g.right_nodes)
                new_reids = len(self.g.red_edges)
                new_beids = len(self.g.blue_edges)

                # Add red edges to each ring member.
                # print("Adding red edges to ring members")
                for ring_member in rings[rnode]:
                    pairrr = (ring_member, rnode)
                    # print("pairrr = " + str(pairrr))
                    # print("EIDS? = " + str([eid for eid in self.g.red_edges]))
                    # print("any([pairrr[0] == eid[0] and pairrr[1] == eid[1] for eid in self.g.red_edges])? = " + str(any([pairrr[0] == eid[0] and pairrr[1] == eid[1] for eid in self.g.red_edges])))
                    new_eid = self.g.add_edge(1, pairrr, 1.0, self.t)
                    assert new_eid in self.g.red_edges
                    assert len(self.g.left_nodes) == new_lnids
                    assert len(self.g.right_nodes) == new_rnids
                    try:
                        assert len(self.g.red_edges) == new_reids + 1
                    except AssertionError:
                        print("Tried to add a single red edge and got " +
                              str(len(self.g.red_edges) - new_reids) +
                              " instead.")
                        assert False
                    assert len(self.g.blue_edges) == new_beids

                    new_lnids = len(self.g.left_nodes)
                    new_rnids = len(self.g.right_nodes)
                    new_reids = len(self.g.red_edges)
                    new_beids = len(self.g.blue_edges)

                    self.ownership[new_eid] = self.ownership[new_eid[1]]
                    ct += 1

                # Determine amounts.
                change = random()*tot_amt
                self.amounts[change_node] = change
                summary[-1] += [(change_node, change)]
                txn_amt = tot_amt - change
                self.amounts[recipient_node] = txn_amt
                summary[-1] += [(recipient_node, txn_amt)]

        new_lnids = len(self.g.left_nodes)
        new_rnids = len(self.g.right_nodes)
        new_reids = len(self.g.red_edges)
        new_beids = len(self.g.blue_edges)

        assert new_lnids == old_lnids + left_nodes_to_be_added
        assert new_rnids == old_rnids + right_nodes_to_be_added
        assert new_reids == old_reids + ct
        s = "Expected " + str(red_edges_to_be_added)
        s += " red edges to be added, but got " + str(new_reids - old_reids)
        s += " new red edges instead."
        try:
            assert new_reids == old_reids + red_edges_to_be_added
        except AssertionError:
            print(s)
            assert False
        assert new_beids == old_beids + blue_edges_to_be_added

        return summary, s

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
        result = self.next_mining_reward
        if not self.dummy_monero_mining_flat:
            self.next_mining_reward *= (1.0-EMISSION_RATIO)
            if self.next_mining_reward < MIN_MINING_REWARD:
                self.next_mining_reward = MIN_MINING_REWARD
                self.dummy_monero_mining_flat = True
        return result

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
        avail = [x for x in list(self.g.left_nodes.keys()) if x != spender]
        if self.mode == "uniform":
            # TODO: DO WE UPDATE LEFT NODES ITERATIVELY???
            k = min(len(self.g.left_nodes), self.ringsize) - 1
            ring = sample(avail, k)
            assert len(ring) == k
            ring += [spender]
            assert len(ring) == k + 1
            print("ring = " + str(len(ring)) + " , " + str(ring))
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

