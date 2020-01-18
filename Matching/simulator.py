from itertools import groupby
from graphtheory import *
from random import random, sample, randrange
from copy import deepcopy

# Some Monero-specific constants
MAX_MONERO_ATOMIC_UNITS = 2**64 - 1
EMISSION_RATIO = 2**-18
EMISSION_RATIO_COMPLEMENT = 1.0 - EMISSION_RATIO
MIN_MINING_REWARD = 6e11
# A useful dictionary for human-readable files
OWNERSHIP_DICT = {0: "Alice", 1: "Eve", 2: "Bob"}


class Simulator(object):
    """
    Simulator object.

    Simulates an economy with a Markov chain between Alice (single party), Eve (a KYC large institution), Bob (all
    other background players). Stores transactions in a Monero-style ledger by using a Bipartite Graph and a
    dictionary tracking ground truth ownership and amounts. Writes the ledger to file with its ground truth
    in a human_readable format.
    
    Attributes:
        runtime            : simulation runtime in blocks
        hashrate           : list of 3 hashrates summing to 1.0
        stochastic_matrix  : stochastic 3x3 matrix for transition of ownership
        spend_times        : list of 3 lambda functions that are PMFs
        min_spend_time     : positive integer
        ringsize           : positive integer
        flat               : boolean (has monero reward become flat?)
        dt                 : positive integer (timestep width)
        max_atomic_units   : positive integer, in Monero 2**64 - 1
        emission           : float, in Monero 2**-18
        min_reward         : positive integer, in Monero 6e11
        filename           : filename
        next_mining_reward : positive integer
        t                  : positive integer, next block height
        ownership          : dict() tracking ownership
        amounts            : dict() tracking amounts
        real_red_edges     : dict() tracking true spending edges
        buffer             : spending buffer
        g                  : BipartiteGraph visualizing the Monero ledger.

    Methods in this class are ordered first according to dependency upon other methods and second according to elegance.

    """
    def __init__(self, inp=None):
        self.runtime = inp['runtime']
        self.hashrate = inp['hashrate']
        self.stochastic_matrix = inp['stochastic matrix']
        self.spend_times = inp['spend times']
        self.min_spend_time = inp['min spend time']
        self.ringsize = inp['ring size']
        self.flat = inp['flat']  # boolean : are block rewards flat yet?
        self.dt = inp['timestep']  # = 1
        assert self.dt >= 1
        self.max_atomic_units = inp['max atomic']  # 2**64 - 1 in monero
        self.emission = inp['emission']  # 2**-18 in monero
        self.min_reward = inp['min reward']  # 6e11 in Monero
        self.filename = inp['filename']
        with open(self.filename, "w+"):
            pass
        self.next_mining_reward = self.emission*self.max_atomic_units
        self.t = 0
        self.ownership = dict()
        self.amounts = dict()
        self.real_red_edges = dict()
        self.buffer = []
        for i in range(self.runtime):
            self.buffer += [[]]
        self.g = BipartiteGraph()

    def gen_time_step(self):
        """ Return next timestep (deterministic). """
        return min(1, self.dt)

    def gen_coinbase_owner(self):
        """ Generate a random coinbase owner based on hashrate. """
        # Simple selection from a PMF.
        out, r = 0, random()
        u = self.hashrate[out]
        while r > u and out + 1 < len(self.hashrate):
            out += 1
            u += self.hashrate[out]
        if out >= len(self.hashrate):
            raise LookupError("Error in gen_coinbase_owner: tried to generate an index with pmf " +
                              str(self.hashrate) + " but got " + str(out))
        return out

    def gen_coinbase_amt(self):
        """ Get next block reward and update. """
        if self.t >= self.runtime:
            raise RuntimeError("Error in gen_coinbase_amt: can't gen coinbase amt for a block outside of runtime")
        # If time hasn't run out, get the next mining reward.
        out = deepcopy(self.next_mining_reward)
        # If block rewards aren't yet flat, drop next reward and check if flatness should begin.
        if not self.flat:
            self.next_mining_reward = max(self.next_mining_reward*(1.0 - self.emission), self.min_reward)
            self.flat = self.next_mining_reward == MIN_MINING_REWARD
        return out

    def add_left_node_to_buffer(self, x, owner):
        """ Mark a left_node/one-time Monero key x for spending by owner """
        if x not in self.g.left_nodes:
            raise LookupError("Tried to add " + str(x) + " to buffer but is not an element of g.left_nodes.")

        # Pick a spend time
        dt = self.gen_spend_time(owner)
        if self.t + dt < self.runtime:
            # If the spend time is too long, this method does nothing.
            # Otherwise, a recipient is determined and x is placed in the buffer.
            recip = self.gen_recipient(owner)
            old_ct = len(self.buffer[self.t + dt])
            self.buffer[self.t + dt] += [(owner, recip, x)]
            new_ct = len(self.buffer[self.t + dt])
            if old_ct + 1 != new_ct:
                raise AttributeError("Error in make_coinbase: Tried to append to buffer, length unchanged.")

    def make_lefts(self, sender, recip, amt):
        """ Make 2 new left nodes indicating change and recipient output whose amounts sum to amt. """
        # Split amounts across two new left nodes
        old_ct = len(self.g.left_nodes)
        out, r = None, random()
        ch_amt, recip_amt = r * amt, amt - r*amt
        ch_out, recip_out = self.g.add_node(0, self.t), self.g.add_node(0, self.t)
        out = [ch_out, recip_out]
        new_ct = len(self.g.left_nodes)
        if old_ct + 2 != new_ct:
            raise AttributeError("Error in make_lefts: called add_node twice but did not get 2 new nodes.")

        # Set ownerships and amounts. 
        self.amounts[ch_out] = ch_amt
        self.ownership[ch_out] = sender
        self.amounts[recip_out] = recip_amt
        self.ownership[recip_out] = recip
        return out

    def make_rights(self, sender, true_spenders):
        """ Construct signatures by signer with the keys in true_spenders. """
        old_ct = len(self.g.right_nodes)
        old_old_ct = old_ct
        n = len(true_spenders)
        out = []
        for true_spender in true_spenders:
            # For each true spender, create new right node, mark ownership, note the real red edge.
            out += [self.g.add_node(1, self.t)]
            self.real_red_edges[out[-1]] = true_spender
            new_ct = len(self.g.right_nodes)
            if old_ct + 1 != new_ct:
                raise AttributeError("Error in make_rights: called add_node but did not get a single new node.")
            old_ct = new_ct
            self.ownership[out[-1]] = sender
        new_ct = len(self.g.right_nodes)
        if old_old_ct + n != new_ct:
            raise AttributeError("Error in make_rights: called add_node " + str(n) +
                                 " times, didn't get " + str(n) + " new nodes.")
        return out

    def make_reds(self, rings, new_rights):
        """ Make new red edges between rings and new_rights. """
        old_ct = len(self.g.red_edges)
        old_old_ct = old_ct
        # Compute an expected number of red edges we should be gaining.
        expected_new_red_edges = sum([len(ring) for ring in rings])
        out = []
        if not len(new_rights) == len(rings):
            raise AttributeError("Error in make_reds: number of sigs and number of rings do not match")
        if len(rings) == 0:
            raise LookupError("Error in make_reds: cannot make reds without ring members. Is the graph is empty?")
        if any([len(ring) != len(set(ring)) for ring in rings]):
            raise IndexError("Error in make_reds: Duplicate ring members present.")
        if len(new_rights) != len(set(new_rights)):
            raise IndexError("Error in make_reds: Duplicate new right nodes present.")

        for R, y in zip(rings, new_rights):
            for x in R:
                if any([edge_id[0] == x and edge_id[1] == y for edge_id in self.g.red_edges]):
                    raise LookupError("Error in make_reds: tried to make a red edge that already exists.")
                # For each ring member-signature pair that is already a red edge, create it.
                out += [self.g.add_edge(1, (x, y), 1.0, self.t)]
                new_ct = len(self.g.red_edges)
                if old_ct + 1 != new_ct:
                    raise AttributeError("Error in make_reds: called add_edge, didn't get a single new red edge.")
                old_ct = new_ct

        new_ct = len(self.g.red_edges)
        if old_old_ct + expected_new_red_edges != new_ct:
            raise AttributeError("Error in make_reds: called add_edge " + str(expected_new_red_edges) +
                                 " times but got a different number of new red edges.")
        return out

    def make_blues(self, new_lefts, new_rights):
        """ Make new blue edges between new_lefts and new_rights. """
        old_ct = len(self.g.blue_edges)
        old_old_ct = old_ct
        expected_new_blue_edges = len(new_lefts)*len(new_rights)

        out = []
        if len(set(new_lefts)) != len(new_lefts):
            raise IndexError("Error in make_blues: duplicate new left nodes present.")
        if len(set(new_rights)) != len(new_rights):
            raise IndexError("Error in make_blues: duplicate new right nodes present.")
        if len(new_lefts) == 0 or len(new_rights) == 0:
            raise LookupError("Error in make_blues: cannot make blue edges without some new_lefts or new_rights.")

        old_ct = len(self.g.blue_edges)
        for x in new_lefts:
            for y in new_rights:
                out += [self.g.add_edge(0, (x, y), 1.0, self.t)]
                new_ct = len(self.g.blue_edges)
                if old_ct + 1 != new_ct:
                    raise AttributeError("Error in make_blues: called add_edge but did not get a single new edge.")
                old_ct = new_ct
        new_ct = len(self.g.blue_edges)

        if old_old_ct + expected_new_blue_edges != new_ct:
            raise AttributeError  # ("Error in make_blues: called add_edge " + str(expected_new_blue_edges) +
            # " times but did not get this many new blue edges.")
        return out

    def gen_spend_time(self, owner):
        """ Generate wait time for buffer placement. """
        f = self.spend_times[owner]
        i, u, r = 0, f(0), random()
        while r > u and i + 1 < self.runtime - self.t - self.min_spend_time:
            i += 1
            u += f(i)
        i = i + self.min_spend_time
        # if i >= self.runtime - self.t:
        #     # Not really an error, and will be removed!
        #     raise Exception("Error in gen_spend_time: generated time beyond runtime.")
        return i

    def gen_recipient(self, owner):
        """ Generate a recipient based on the stochastic matrix. """
        f = self.stochastic_matrix[owner]
        i, u, r = 0, f[0], random()
        while r > u and i + 1 < len(self.stochastic_matrix):
            i += 1
            u += f[i]
        if i < 0 or i >= len(self.stochastic_matrix):
            raise Exception  # ("Error in gen_recipient: Tried to generate an index in " +
            # str(range(len(self.stochastic_matrix))) + " but got " + str(i))
        return i

    def make_coinbase(self):
        """ Make a new coinbase output. """
        # Pick a coinbase owner based on hashrate
        owner = self.gen_coinbase_owner()

        # Pick coinbase amount based on emission
        v = self.gen_coinbase_amt()

        # Add new left node
        old_ct = len(self.g.left_nodes)
        out = self.g.add_node(0, self.t)
        new_ct = len(self.g.left_nodes)
        if old_ct + 1 != new_ct:
            raise RuntimeError("Error in make_coinbase: called add_node but did not get a single new node.")

        # Set left node's ownership and amount
        self.ownership[out] = owner
        self.amounts[out] = v
        self.add_left_node_to_buffer(out, owner)
        return out

    def gen_rings(self, signing_keys):
        """ Select ring members uniformly at random from available members. """
        if len(signing_keys) == 0:
            raise AttributeError  # ("Error in gen_rings: Tried to sign without any keys.")
        if any([_[1] + self.min_spend_time > self.t for _ in signing_keys]):
            raise AttributeError  # ("Error in gen_rings: Some signing keys have not matured past min lock time yet.")

        out = []
        valid_ring_members = [x for x in self.g.left_nodes if x[1] + self.min_spend_time <= self.t]
        actual_ring_size = min(self.ringsize, len(valid_ring_members))
        if actual_ring_size == 0:
            raise AttributeError  # ("Error in gen_rings: No available ring members.")
        if any([x not in valid_ring_members for x in signing_keys]):
            raise AttributeError  # ("Error in gen_rings: Signing key not available ring member.")

        for _ in signing_keys:
            next_ring = sample(valid_ring_members, actual_ring_size)
            if _ not in next_ring:
                i = randrange(actual_ring_size)
                next_ring[i] = _            
            out += [next_ring]
            
        for R, y in zip(out, signing_keys):
            assert y in R
            
        return out

    def make_txn(self, sender, recip, grp):
        """ Construct a transaction from sender to recipient whose true signers are in the iterator grp. """ 
        if sender not in range(len(self.stochastic_matrix)):
            raise AttributeError
        if recip not in range(len(self.stochastic_matrix)):
            raise AttributeError
        if all(False for _ in deepcopy(grp)):
            # In this case, the input iterator is empty.
            raise AttributeError

        tmp_grp = deepcopy(grp)
        out = []

        # Make a new signature node for each signing key
        signing_keys = [_[2] for _ in tmp_grp]
        out += [self.make_rights(sender, signing_keys)]

        # Make two new outputs with appropriately chosen amounts
        amt = sum([self.amounts[_] for _ in signing_keys])
        out += [self.make_lefts(sender, recip, amt)]

        for new_left in out[-1]:
            self.add_left_node_to_buffer(new_left, recip)

        # Select ring members
        out += [self.gen_rings(signing_keys)]

        # Make new red edges
        out += [self.make_reds(out[-1], out[0])]

        # Make new blue edges
        out += [self.make_blues(out[1], out[0])]

        # Return list with [new rights, new lefts, rings, new reds, new blues]
        return out

    def make_txns(self):
        """ Makes all transactions sitting in the buffer. """
        out = []
        bndl = groupby(self.buffer[self.t], key=lambda x: (x[0], x[1]))
        is_empty = all(False for _ in deepcopy(bndl))
        if not is_empty:
            for k, grp in bndl:
                sender, recip = k  # Each grp in bndl consists of left nodes being spent with this sender-recip pair.
                out += [self.make_txn(sender, recip, deepcopy(grp))]
        return out

    def update_state(self, dt):
        """ If time hasn't run out, call make_coinbase and make_txns """
        # Check if dt is beyond the time horizon. If not, make a coinbase and spend from the buffer.
        out = []
        target_time = self.t + dt
        if target_time >= self.runtime:
            # Simulation comes to an end.
            return out
        while self.t < target_time:
            self.t += self.dt
            out += [[self.make_coinbase(), self.make_txns()]]
        return out

    def step(self):
        """  Generate a timestep and update the state. """
        dt = self.gen_time_step()
        return self.update_state(dt)

    def human_parse(self, txn):
        """ Describe txn in human-readable terms. """
        [new_rights, new_lefts, rings] = txn[:3]
        senders = [self.ownership[_] for _ in new_rights]
        if any(x != y for x in senders for y in senders):
            raise AttributeError
        # if len(list(set(senders))) != len(senders):
        #    raise AttributeError
        sender = senders[0]
        sender_name = str(OWNERSHIP_DICT[sender])

        true_spenders = [self.real_red_edges[_] for _ in new_rights]  # string.
        input_amt = sum([self.amounts[_] for _ in true_spenders])  # not string

        ch_out = new_lefts[0]
        r_out = new_lefts[1]
        ch_recip = self.ownership[ch_out]
        r_recip = self.ownership[r_out]

        recip_names = [OWNERSHIP_DICT[ch_recip], OWNERSHIP_DICT[r_recip]]
        residual = self.amounts[ch_out] + self.amounts[r_out] - input_amt
        return "Also, " + sender_name + " spends " + str(input_amt) + " with one-time keys " + str(true_spenders) + \
               ", respective ring members " + str(rings) + ", outputting change key " + str(ch_out) + \
               " and recipient key " + str(r_out) + ", respectively owned by " + recip_names[0] + " and " + \
               recip_names[1] + " and resp. amounts " + str(self.amounts[ch_out]) + " and " + \
               str(self.amounts[r_out]) + ", note INPUT - OUTPUT = " + str(residual) + ".\n"

    def record(self, out):
        """ Write ledger and its ground truth to file. """
        h = 1
        with open(self.filename, "w+") as wf:
            for steps in out:
                for block in steps:
                    cb = block[0]
                    txns = block[1]
                    line = ""
                    line += str(h) + "^th block: \n"
                    line += "The coinbase with left_node_identity " + str(cb) + " was mined by "
                    line += OWNERSHIP_DICT[self.ownership[cb]] + " for block reward " + str(self.amounts[cb]) + ".\n"
                    for txn in txns:
                        line += self.human_parse(txn)
                    line += "\n"
                    h += 1

                    wf.write(line)

    def run(self):
        """ Execute self.step until time runs out and then record results. """
        out = [self.step()]
        while self.t + self.dt <= self.runtime:
            out += [self.step()]
        self.record(out)
        return out
