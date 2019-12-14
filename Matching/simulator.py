from itertools import groupby
from graphtheory import *
from random import *
from copy import deepcopy

MAX_MONERO_ATOMIC_UNITS = 2**64 - 1
EMISSION_RATIO = 2**-18
MIN_MINING_REWARD = 6e11


class Simulator(object):
    """
    Simulator object that generates a Monero-style ledger using a Markov chain; technically not a Markov process because
    the time delay between events is not memoryless due to min spend-times.

    Attributes:
        runtime                  : positive integer
        fn                       : string
        stoch_matrix             : matrix (list of lists)
        hashrate                 : list
        minspendtime             : positive integer
        spendtimes               : list of lambda functions
        ringsize                 : positive integer
        mode                     : string, "uniform" only option presently supported TODO: add other ring selection modes
        buffer                   : list
        ownership                : dict(), GROUND TRUTH
        amounts                  : dict()
        g                        : BipartiteGraph()
        t                        : non-negative integer (time)
        dummy_monero_mining_flat : Boolean
        next_mining_reward       : positive integer, EMISSION_RATIO*MAX_MONERO_ATOMIC_UNITS only option presently
        reporting_modulus        : positive integer, how often we write to file

    Initialization: Input dictionary par has the following key-value pairs:
        'runtime'                : simulation runtime, bounds t, number of blocks
        'filename'               : stored in fn
        'stochastic matrix'      : input stochastic matrix governing the economy
        'hashrate'               : vector of hashrates, index here = index in stochastic_matrix
        'spendtimes'             : list of lambda functions
        'min spendtime'          : minimal amount of time spenders must wait before spending
        'ring size'              : size of red-neighbor set of a given right-node

    Methods:
        halting_run          : Execute next timestep of simulation.
        run                  : Execute all timesteps of simulation.
        make_coinbase        : Add a new coinbase key (left node) to the ledger with reward based on this timestep.
        spend_from_buffer    : Add new signatures (right nodes with some new left nodes) to the ledger buffered for this timestep.
        report               : Write to file TODO: Do I still use report? I don't think so...
        pick_coinbase_owner  : Sample a coinbase owner based on hashrate vector.
        pick_coinbase_amt    : Compute coinbase amount based on block height.
        pick_spend_time      : Sample a spendtime for a new left node
        pick_next_recip      : Sample a recipient for the transaction using stochastic matrix.
        get_ring             : Sample a ring

    Example Usage:
        sm = [[2**(-8), 2**(-3), 1.0 - 2**(-3) - 2**(-8)],
              [2**(-4), 2**(-3), 1.0 - 2**(-3) - 2**(-4)],
              [2**(-1), 2**(-2), 1.0 - 2**(-1) - 2**(-2)]]
        hr = [2**-5, 2**-5, 1.0 - 2**-5 - 2**-4]
        mst = 10
        st = [lambda x: (1.0/2.0)*((1.0 - (1.0/2.0)**(x - mst))),
              lambda x: (1.0/5.0)*((1.0 - (1.0/5.0)**(x - mst))),
              lambda x: (1.0/6.0)*((1.0 - (1.0/6.0)**(x - mst)))]
        par = {'runtime': 100, 'filename': "output.csv", 'stochastic matrix': sm, 'hashrate': hr, 'spendtimes': st,
               'min spendtime': mst, 'ring size': 11}
        sally = Simulator(par)
        sally.run()

    """
    def __init__(self, par=None):
        """ See help(Simulator) """
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
            try:
                assert sum(row) == 1.0
            except AssertionError:
                print("Woops, row didn't sum to 1.0 : " + str(row))
                assert False
        assert 'hashrate' in par
        assert isinstance(par['hashrate'], list)
        for elt in par['hashrate']:
            assert isinstance(elt, float)
            assert 0.0 <= elt <= 1.0
        try:
            assert sum(par['hashrate']) == 1.0
        except AssertionError:
            print("Woopsie! Tried to use a hashrate vector that doesn't sum to 1.0... offending vector = " + str(par['hashrate']))
            assert False
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
        """ halting_run executes a single timestep and returns t+1 < runtime. """
        if self.t + 1 < self.runtime:
            if self.t % 100 == 0:
                print(".", end='')
            self.t += 1
            self.make_coinbase()
            self.spend_from_buffer()
        return self.t+1 < self.runtime

    def run(self):
        """ run iteratively execute all timesteps, returning nothing. """
        while self.t + 1 < self.runtime:
            if self.t % 100 == 0:
                print(".", end='')
            self.t += 1
            self.make_coinbase()
            self.spend_from_buffer()

    def make_coinbase(self):
        """ make_coinbase creates a new coinbase with reward based on the time. """
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
            assert node_to_spend[1] + self.minspendtime <= s
            assert s >= self.t + self.minspendtime
        # assert node_to_spend in self.amounts 
        self.amounts[node_to_spend] = amt
        return node_to_spend, dt

    def spend_from_buffer(self):
        """ spend_from_buffer groups the buffer by sender-recipient pairs.
        For each of these, amounts are decided, new right nodes are created,
        new left nodes are created, new red edges are created, and new blue
        edges are created.

        For convenience, we return a summary (list):
            idx : summary[idx]
              0 : left nodes being spent
              1 : sum of amounts of left nodes being spent
              2 : right nodes being created
              3 : pair left node and amt (change)
              4 : pair left node and amt (recipient)

        WARNING: We assume if a buffer keys has not yet entered the ring_member_choices set, then the key owner
        delays until the next block.
        """
        summary = []
        s = None

        ring_member_choices = [x for x in list(self.g.left_nodes.keys()) if x[1] + self.minspendtime <= self.t]
        num_rmc = len(ring_member_choices)
        red_edges_per_sig = min(num_rmc, self.ringsize)
        # subset = all([any([y[2] == x for y in ring_member_choices]) for x in self.buffer[self.t]])
        # A is a subset of B if for all x in A, there exists some y in B such that x = y
        if len(ring_member_choices) >= self.ringsize:
            init_lnids = len(self.g.left_nodes)
            init_rnids = len(self.g.right_nodes)
            init_reids = len(self.g.red_edges)
            init_beids = len(self.g.blue_edges)

            old_lnids = len(self.g.left_nodes)
            old_rnids = len(self.g.right_nodes)
            old_reids = len(self.g.red_edges)
            old_beids = len(self.g.blue_edges)

            # Ensure buffer elements are spent after minspendtime
            # TODO: Buffer elements are appearing before they should be legally spent, or the checks for this are wrong
            right_nodes_to_be_pushed = [x for x in self.buffer[self.t] if x[2] not in ring_member_choices]
            if self.t + 1 < self.runtime:
                self.buffer[self.t] += right_nodes_to_be_pushed

            right_nodes_to_be_added = [x for x in self.buffer[self.t] if x[2] in ring_member_choices]

            bndl = groupby(right_nodes_to_be_added, key=lambda x: (x[0], x[1]))
            num_txn_bundles = len([x for x in deepcopy(bndl)])
            left_nodes_to_be_added = 2 * num_txn_bundles  # Coinbase elsewhere

            right_nodes_to_be_added = len(right_nodes_to_be_added)
            blue_edges_to_be_added = 2 * right_nodes_to_be_added
            red_edges_to_be_added = red_edges_per_sig * right_nodes_to_be_added

            ct = 0

            if right_nodes_to_be_added > 0:
                # print("Beginning spend from buffer")

                new_lnids = old_lnids
                new_rnids = old_rnids
                new_reids = old_reids
                new_beids = old_beids

                tot_ring_membs = 0
                rings = dict()
                new_right_nodes = []

                for k, grp in bndl:
                    # Collect keys to be spent in this group.
                    temp = deepcopy(grp)
                    keys_to_spend = [x[2] for x in temp if x[2] in ring_member_choices]

                    summary += [[keys_to_spend]]

                    # Compute amount for those keys.
                    # print("Computing amounts")
                    tot_amt = sum([self.amounts[x] for x in keys_to_spend])
                    summary[-1] += [tot_amt]

                    # Create new right node for each key being spent and generate a
                    # ring for that node.

                    temp = deepcopy(grp)
                    for x in temp:
                        if x[2] in ring_member_choices:
                            # x = (sender, receiver, node_id)
                            # Add new right nodes and set ownership.
                            new_right_nodes += [self.g.add_node(1, self.t)]
                            self.ownership[new_right_nodes[-1]] = (k[0], x[2])
                            temp_ring = self.get_ring(x[2], ring_member_choices)
                            assert len(temp_ring) == red_edges_per_sig
                            rings[new_right_nodes[-1]] = temp_ring

                            # Select ring members for each key in the group
                            # print("Picking rings for " + str(x[2]))

                            old_lnids = new_lnids
                            old_rnids = new_rnids
                            old_reids = new_reids
                            old_beids = new_beids

                            new_lnids = len(self.g.left_nodes)
                            new_rnids = len(self.g.right_nodes)
                            new_reids = len(self.g.red_edges)
                            new_beids = len(self.g.blue_edges)

                            assert new_lnids == old_lnids
                            assert new_rnids == old_rnids + 1
                            assert new_reids == old_reids
                            assert new_beids == old_beids


                    summary[-1] += [new_right_nodes]
                    summary[-1] += [rings]

                    old_lnids = new_lnids
                    old_rnids = new_rnids
                    old_reids = new_reids
                    old_beids = new_beids

                    # Create two new left nodes
                    # print("Creating two new left nodes for this group")
                    change_node = self.g.add_node(0, self.t)
                    self.ownership[change_node] = k[0]
                    recipient_node = self.g.add_node(0, self.t)
                    self.ownership[recipient_node] = k[1]

                    new_lnids = len(self.g.left_nodes)
                    new_rnids = len(self.g.right_nodes)
                    new_reids = len(self.g.red_edges)
                    new_beids = len(self.g.blue_edges)

                    assert new_lnids == old_lnids + 2
                    assert new_rnids == old_rnids
                    assert new_reids == old_reids
                    assert new_beids == old_beids

                    old_lnids = new_lnids
                    old_rnids = new_rnids
                    old_reids = new_reids
                    old_beids = new_beids

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

                    new_lnids = len(self.g.left_nodes)
                    new_rnids = len(self.g.right_nodes)
                    new_reids = len(self.g.red_edges)
                    new_beids = len(self.g.blue_edges)

                    assert new_lnids == old_lnids
                    assert new_rnids == old_rnids
                    assert new_reids == old_reids
                    assert new_beids == old_beids + 2
                    assert len(rings[rnode]) == len(list(set(rings[rnode])))

                    old_lnids = new_lnids
                    old_rnids = new_rnids
                    old_reids = new_reids
                    old_beids = new_beids

                    # Add red edges to each ring member.
                    # print("Adding red edges to ring members")
                    # print("rnode = " + str(rnode))
                    # print("rings[rnode] = " + str(rings[rnode]))
                    for ring_member in rings[rnode]:
                        pairrr = (ring_member, rnode)
                        expected_new_eid = (pairrr[0], pair[1], self.t)
                        if expected_new_eid not in self.g.red_edges:
                            new_eid = self.g.add_edge(1, pairrr, 1.0, self.t)
                        assert expected_new_eid == new_eid

                        new_lnids = len(self.g.left_nodes)
                        new_rnids = len(self.g.right_nodes)
                        new_reids = len(self.g.red_edges)
                        new_beids = len(self.g.blue_edges)

                        if new_eid in self.g.red_edges:
                            assert new_lnids == old_lnids
                            assert new_rnids == old_rnids
                            assert new_reids == old_reids + 1
                            assert new_beids == old_beids
                        else:
                            assert new_lnids == old_lnids
                            assert new_rnids == old_rnids
                            assert new_reids == old_reids
                            assert new_beids == old_beids

                        old_lnids = new_lnids
                        old_rnids = new_rnids
                        old_reids = new_reids
                        old_beids = new_beids

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

            assert new_lnids == init_lnids + left_nodes_to_be_added
            assert new_rnids == init_rnids + right_nodes_to_be_added
            # assert new_reids == init_reids + ct
            s = "Expected " + str(red_edges_to_be_added)
            s += " red edges to be added, but got " + str(new_reids - old_reids)
            s += " new red edges instead."
            try:
                assert new_reids == init_reids + red_edges_to_be_added
            except AssertionError:
                print(s)
                assert False
            assert new_beids == init_beids + blue_edges_to_be_added
        else:
            # In this case, there are not enough ring members to construct full ring signatures; we assume
            # spenders decide to wait till the next block.
            self.buffer[self.t + 1] += self.buffer[self.t]
            summary = []
            s = ""
        return summary, s

    def report(self):
        """ report writes a summary of the graph and the ground truth of ownership and amounts to file. """
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
        """ pick_coinbase_owner uses the hashrate vector to determine the owner of the next coinbase output."""
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
        """ pick_coinbase_amt starts with a max reward, multiplies the last mining
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
        """ sample a random spend-time from the owner's spend-time distribution function. """
        try:
            assert owner in range(len(self.stoch_matrix))
        except AssertionError:
            print("Owner not found in stochastic matrix. owner = " + str(owner))
        try:
            assert owner in range(len(self.spendtimes))
        except AssertionError:
            print("Owner not found in spendtimes. owner = " + str(owner))
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
        """ pick_next_recip uses the stochastic matrix and owner information to determine next recipient. """
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

    def get_ring(self, spender, ring_member_choices):
        """ get_ring selects ring members. Presently the only mode is uniform. TODO: Expand modes. """
        ring = []
        assert spender in ring_member_choices
        if self.mode == "uniform":
            k = min(len(ring_member_choices), self.ringsize)
            ring = sample(ring_member_choices, k)
            if spender not in ring:
                i = choice(range(len(ring)))
                ring[i] = spender
            assert len(ring) == k
            assert len(list(set(ring))) == len(ring)
            # print("ring = " + str(len(ring)) + " , " + str(ring))
        return ring

