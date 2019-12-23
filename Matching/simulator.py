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
    def __init__(self, par=None, verbosity=False):
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
        with open(self.fn, "w+") as wf:
            pass
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
        self.buffer = [list() for i in range(self.runtime)]
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

        self.verbosity = verbosity
        self.owner_names = {0: "Alice", 1: "Eve", -2: "Eve", 2: "Bob", -1: "Bob"}

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
            self.next_mining_reward *= (1.0 - EMISSION_RATIO)
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

    def look_for_dupes(self):
        whole_buffer_list = [x for entry in self.buffer for x in entry]
        whole_buffer_set = list(set(whole_buffer_list))
        result = (len(whole_buffer_list) == len(whole_buffer_set))
        try:
            assert result
        except AssertionError:
            counts = dict()
            for x in whole_buffer_set:
                counts.update({x: 0})
            for x in whole_buffer_list:
                counts[x] += 1
            repeats = [x for x in counts if counts[x] > 1]
            print("\n\n ERROR: LOOKS LIKE A DUPE MADE IT INTO THE BUFFER. repeats = " + str(repeats))
            assert result
        return result

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
            assert not any([(owner, recip, node_to_spend) in buff for buff in self.buffer])
            self.look_for_dupes()
            self.buffer[s] += [(owner, recip, node_to_spend)]
            self.look_for_dupes()
            assert node_to_spend[1] + self.minspendtime <= s
            assert s >= self.t + self.minspendtime
        # assert node_to_spend in self.amounts
        self.amounts[node_to_spend] = amt
        return node_to_spend, dt

    def get_ring(self, spender, ring_member_choices):
        """ get_ring selects ring members. Presently the only mode is uniform. """
        # TODO: Expand modes.
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

    def spend_from_buffer(self):
        """
        spend_from_buffer

        spend_from_buffer takes outputs pending "to be spent"  in the buffer at the current index, adds them to
        the graph, and adds appropriate edges. It does so in the following way:
            FIRST:  If there are not enough ring members available, buffer rolls over and spend_from_buffer returns an
                    empty string. Otherwise, the buffer is inspected for any outputs that are being spent before their min spend time
                    (assumed to be a concensus-enforced locktime) and rolls these over.
            SECOND: The remaining buffer is grouped by sender-recipient pair.
            THIRD:  New nodes are added for each sender-recipient-pair:
                    (i)   2 new left nodes are added to the graph and
                    (ii)  2 new spendtimes are samples for these and they are added to the buffer and
                    (iii) 1 new right node (signature node) is added to the graph and
                    (iv)  ring members are sampled and
                    (v)   transaction amounts are decided.
            FOURTH: New edges are added likewise:
                    (i)   New red edges from each ring member to each corresponding new right node
                    (ii)  2 new blue edges from each new right node to each corresponding pair of new output left nodes

        WARNING: We assume if a buffer keys has not yet entered the ring_member_choices set, then the key owner
        delays until the next block. This should not occur, but if it does it should not cause more than a single
        block delay.
        """

        ring_member_choices = [x for x in self.g.left_nodes if x[1] + self.minspendtime <= self.t]
        assert all([x[2] in ring_member_choices for x in self.buffer[self.t]])
        num_rmc = len(ring_member_choices)
        red_edges_per_sig = min(num_rmc, self.ringsize)

        # We will return new_left, new_right, new_blue, new_red
        new_right_nodes = []
        new_left_nodes = []
        new_blue_edges = []
        new_red_edges = []
        next_txn = []
        true_spender_ring_sig_relation = {}

        if len(self.buffer[self.t]) > 0 and red_edges_per_sig != self.ringsize and self.t + 1 < len(self.buffer):
            self.buffer[self.t + 1] += self.buffer[self.t]
            self.buffer[self.t] = list()

        elif red_edges_per_sig == self.ringsize:
            to_be_spent = deepcopy(self.buffer[self.t])
            node_ids_to_be_spent = [x[2] for x in to_be_spent]

            # print("~~~~~> right_nodes_remaining " + str(right_nodes_remaining))
            if len(to_be_spent) > 0:
                bndl = groupby(to_be_spent, key=lambda x: (x[0], x[1]))
                rings = dict()

                for k, grp in deepcopy(bndl):
                    temp = deepcopy(grp)
                    keys_to_spend = [x[2] for x in temp]
                    tot_amt = sum([self.amounts[x] for x in keys_to_spend])

                    # New right nodes
                    # print("keys_to_spend " + str(keys_to_spend))
                    for x in keys_to_spend:
                        # print("about to add a right node")
                        new_right_nodes += [self.g.add_node(1, self.t)]
                        self.ownership.update({new_right_nodes[-1]: k[0]})
                        rings[new_right_nodes[-1]] = self.get_ring(x, ring_member_choices)
                        true_spender_ring_sig_relation.update({new_right_nodes[-1]: x})

                    u = random()

                    # New left nodes
                    new_left_nodes += [self.g.add_node(0, self.t)]
                    recip_node = new_left_nodes[-1]
                    self.ownership[recip_node] = k[1]
                    recip_dt = self.pick_spend_time(k[1])
                    recip_next_recip = self.pick_next_recip(k[1])
                    recip_s = self.t + recip_dt
                    if recip_s < len(self.buffer):
                        assert not any([(k[1], recip_next_recip, recip_node) in buff for buff in self.buffer])
                        self.buffer[recip_s] += [(k[1], recip_next_recip, recip_node)]
                    self.amounts[recip_node] = u * tot_amt

                    new_left_nodes += [self.g.add_node(0, self.t)]
                    change_node = new_left_nodes[-1]
                    self.ownership[change_node] = k[0]
                    change_dt = self.pick_spend_time(k[0])
                    change_next_recip = self.pick_next_recip(k[0])
                    change_s = self.t + change_dt
                    if change_s < len(self.buffer):
                        assert not any([(k[0], change_next_recip, change_node) in buff for buff in self.buffer])
                        self.buffer[change_s] += [(k[0], change_next_recip, change_node)]
                    self.amounts[change_node] = tot_amt - self.amounts[recip_node]

                for rnode in new_right_nodes:
                    recip_pair = (recip_node, rnode)
                    new_blue_edges += [self.g.add_edge(0, recip_pair, 1.0, self.t)]
                    self.ownership[new_blue_edges[-1]] = self.ownership[new_blue_edges[-1][0]]

                    change_pair = (change_node, rnode)
                    new_blue_edges += [self.g.add_edge(0, change_pair, 1.0, self.t)]
                    self.ownership[new_blue_edges[-1]] = self.ownership[new_blue_edges[-1][0]]

                    for ring_member in rings[rnode]:
                        ring_member_pair = (ring_member, rnode)
                        new_red_edges += [self.g.add_edge(1, ring_member_pair, 1.0, self.t)]
                        self.ownership[new_red_edges[-1]] = self.ownership[new_red_edges[-1][1]]

                    with open(self.fn, "a") as wf:
                        # Note: generally more than one ring signature is used to create new left nodes, so expect
                        # overlap in the resulting file.
                        line = self.owner_names[self.ownership[rnode]] + " constructs new ring signature " + str(
                            rnode) + " with ring members " + str(
                            rings[rnode]) + " and true spender " + str(true_spender_ring_sig_relation[rnode]) + ", authorizing the creation of new left node " + str(
                            recip_node) + " owned by " + self.owner_names[self.ownership[recip_node]] + " and new left node " + str(
                            change_node) + " owned by " + self.owner_names[self.ownership[change_node]] + ".\n"
                        wf.write(line)

                self.look_for_dupes()
            
        return [len(new_left_nodes), len(new_right_nodes), len(new_red_edges), len(new_blue_edges)]

    def halting_run(self):
        """ halting_run executes a single timestep and returns t+1 < runtime. """
        if self.t + 1 < self.runtime:
            # Make predictions
            old_predictions = [0, 0, 0, 0]

            # Take old stats
            old_bndl = groupby(self.buffer[self.t + 1], key=lambda x: (x[0], x[1]))
            old_r = len(self.buffer[self.t + 1])
            old_l = sum([1 for _ in deepcopy(old_bndl)])
            old_rmc = [x for x in self.g.left_nodes if x[1] + self.minspendtime <= self.t]
            old_stats = [self.t, len(self.g.left_nodes), len(self.g.right_nodes), len(self.g.red_edges),
                         len(self.g.blue_edges), len(old_rmc)]

            # Look: old_r = len(self.buffer[self.t + 1]) = number of new right nodes to be added in the next block
            # Also: old_l = number of txn bundles, each producing two output (new left nodes) in the next block

            # Do a thing
            # if self.t % 100 == 0:
            #     print(".", end='')
            self.t += 1
            with open(self.fn, "a") as wf:
                wf.write("Beginning block height " + str(self.t) + ".\n")

            # Take new stats
            new_rmc = [x for x in self.g.left_nodes if x[1] + self.minspendtime <= self.t]
            new_stats = [self.t, len(self.g.left_nodes), len(self.g.right_nodes), len(self.g.red_edges),
                         len(self.g.blue_edges), new_rmc]

            # Check predictions
            assert new_stats[0] == old_stats[0] + 1
            assert new_stats[1] == old_stats[1] + old_predictions[0]
            assert new_stats[2] == old_stats[2] + old_predictions[1]
            assert new_stats[3] == old_stats[3] + old_predictions[2]
            assert new_stats[4] == old_stats[4] + old_predictions[3]
            self.look_for_dupes()

            # Make predictions
            new_predictions = [1, 0, 0, 0]

            # Reset old stats
            old_stats = new_stats

            # Do a thing
            node_id, time_interval = self.make_coinbase()
            ow = self.ownership[node_id]
            with open(self.fn, "a") as wf:
                wf.write(self.owner_names[ow] + " mines a new coinbase output with node_id " + str(node_id) + ".\n")



            # Take new stats
            new_rmc = [x for x in self.g.left_nodes if x[1] + self.minspendtime <= self.t]
            new_stats = [self.t, len(self.g.left_nodes), len(self.g.right_nodes), len(self.g.red_edges),
                         len(self.g.blue_edges), len(new_rmc)]

            # Check predictions
            assert new_stats[0] == old_stats[0]
            assert new_stats[1] == old_stats[1] + new_predictions[0]
            assert new_stats[2] == old_stats[2] + new_predictions[1]
            assert new_stats[3] == old_stats[3] + new_predictions[2]
            assert new_stats[4] == old_stats[4] + new_predictions[3]
            self.look_for_dupes()

            new_predictions = [0, 0, 0, 0]
            num_rmc = len(new_rmc)
            red_edges_per_sig = min(num_rmc, self.ringsize)

            if red_edges_per_sig == self.ringsize:
                right_nodes_remaining = [x for x in self.buffer[self.t] if x[2] in new_rmc]
                if len(right_nodes_remaining) > 0:
                    bndl = groupby(right_nodes_remaining, key=lambda x: (x[0], x[1]))
                    num_txns = sum([1 for k, grp in bndl])
                    num_new_rights = len(right_nodes_remaining)
                    new_predictions = [2 * num_txns, num_new_rights, self.ringsize * num_new_rights, 2 * num_new_rights]

            # print("HR Predictions = " + str(new_predictions))
            # print("HR Current num left nodes = " + str(len(self.g.left_nodes)))
            # print("HR Current num right nodes = " + str(len(self.g.right_nodes)))
            # print("HR Current num red edges = " + str(len(self.g.red_edges)))
            # print("HR Current num blue edges = " + str(len(self.g.blue_edges)))
            # print("HR Predicted num left nodes = " + str(len(self.g.left_nodes) + new_predictions[0]))
            # print("HR Predicted num right nodes = " + str(len(self.g.right_nodes) + new_predictions[1]))
            # print("HR Predicted num red edges = " + str(len(self.g.red_edges) + new_predictions[2]))
            # print("HR Predicted num blue edges = " + str(len(self.g.blue_edges) + new_predictions[3]))
            
            # Reset old stats
            old_stats = new_stats

            # Do a thing
            new_stuff = self.spend_from_buffer()
            # print("New stuff = " + str(new_stuff))
            
            assert new_stuff[0] == new_predictions[0]  # left
            assert new_stuff[1] == new_predictions[1]  # right
            assert new_stuff[2] == new_predictions[2]  # red
            assert new_stuff[3] == new_predictions[3]  # blue

            # Take new stats
            new_rmc = [x for x in self.g.left_nodes if x[1] + self.minspendtime <= self.t]
            new_stats = [self.t, len(self.g.left_nodes), len(self.g.right_nodes), len(self.g.red_edges),
                         len(self.g.blue_edges), new_rmc]

            # Check predictions
            assert new_stats[0] == old_stats[0]
            # print("HR Observed num left nodes = " + str(new_stats[1]))
            # print("HR Observed num right nodes = " + str(new_stats[2]))
            # print("HR Observed num red edges = " + str(new_stats[3]))
            # print("HR Observed num blue edges = " + str(new_stats[4]))
            assert new_stats[1] == old_stats[1] + new_predictions[0]
            assert new_stats[2] == old_stats[2] + new_predictions[1]
            assert new_stats[3] == old_stats[3] + new_predictions[2]
            assert new_stats[4] == old_stats[4] + new_predictions[3]
            self.look_for_dupes()

        return self.t+1 < self.runtime

    def run(self):
        """ run iteratively execute all timesteps, returning nothing. """
        alpha = 0
        keep_going = self.halting_run()
        while keep_going:
            alpha += 1
            keep_going = self.halting_run()

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
