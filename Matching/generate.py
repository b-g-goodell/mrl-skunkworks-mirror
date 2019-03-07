import random
from graphtheory import *
from math import *

par = {'data': 1.0, 'ident':'the name of the graph', 'left': [], 'right': [], 'edges': []}
G = BipartiteGraph(par)

R = 11 # ring size
T = 100 # number blocks
M = 20 # trigger block height for weird behavior.
N = 10 # locktime for monero transactions measured in blocks
p = 0.1 # proportion of outputs sent to Alice

timesUp = [[] for i in range(T)] #timesUp[h] = outputs that shall be spent at block height h.
blockChain = [[[], []] for i in range(T)] #blockChain[h] = [[], []], blockChain[h][0] = idents for new outputs created at block height h, blockChain[h][1] = signature idents for new signatures created at block height h.
taint = [] # list of tainted outputs
true_edges = []

#### Various functions for ease of use ####

def addCoinbase(h, ct):
    # Create a new coinbase transaction, add it to the graph, record it on the blockchain, pick a time delay for spending, determine if
    # the miner is taintman, output our rolling index ct for naming nodes.
    next_ident = str(ct)
    ct = str(int(ct)+1)
    blockChain[t][0].append(next_ident)
    node_to_add = Node({'data':None, 'ident':next_ident, 'edges':[]})
    G._add_left(node_to_add)
    
    s = getWalletDelay() # measured in number of blocks
    if h+s < T:
        timesUp[h+s].append(next_ident)

    u = random.random()
    if u < p:
        taints.append(next_ident)

    return ct

def Sign(h, list_of_inputs, ct, numOuts):
    # place new signatures, one for each input, record true edge, pick ring members, add fake edges
    sig_idents = []
    for i in list_of_inputs:
        # Each input requires a ring signature node to be added on the right side of the graph
        next_ident = str(ct)
        sig_idents.append(next_ident)
        ct = str(int(ct)+1)
        blockChain[h][1].append(ct)
        node_to_add = Node({'data':None, 'ident':next_ident, 'edges':[]})
        G._add_right(node_to_add)
        
        # We can immediately assign the true edge
        edge_ident = str((i,next_ident))
        true_edges.append(edge_ident) # Record true edge and place edge on graph
        G._add_edge({'data':None, 'ident':edge_ident, 'left':i, 'right':next_ident, 'weight':0.0})

        # Now we pick ring members and assign those edges.
        next_ring = getRing(i) # For this signature, pick a ring
        for ringMember in next_ring:
            edge_ident = str((ringMember, next_ident))
            G._add_edge({'data':None, 'ident':edge_ident, 'left':ringMember, 'right':next_ident, 'weight':0.0})

    # place new output nodes on the left of G and determine their delays and whether they are tainted.
    out_idents = []
    for i in range(numOuts):
        # create next new output
        next_ident = str(ct)
        out_idents.append(next_ident)
        ct = str(int(ct)+1)        
        blockChain[h][0].append(ct) # add to blockchain list
        node_to_add = Node({'data':None, 'ident':next_ident, 'edges':[]})
        G._add_left(node_to_add)

        # Determine delays of next new output
        if h > M and list_of_inputs[0] in taint:
            s = getTaintDelay()
        else:
            s = getWalletDelay()
        if h+s < T:
            timesUp[h+s].append(next_ident)

        # Determine if tainted
        u = random.random()
        if u < p:
            taints.append(next_ident)

        # add edges from signatures to these new outputs
        for sig_ident in sig_idents:
            edge_ident = str((next_ident, sig_ident))
            G._add_edge({'data':None, 'ident':edge_ident, 'left':next_ident, 'right':sig_ident, 'weight':0.0})
    return ct

def Spend(h, list_of_inputs, ct):
    # breaks list_of_inputs into pieces called transactions
    # then calls Sign(-) on each piece

    # First: create tainted transactions.
    tainted = [x for x in list_of_inputs if x in taint]
    while(len(tainted) > 0):
        nums = getNumbers()
        nums[0] = max(1, min(nums[0], len(tainted))) # pick number of outs in next tainted txn.
        inputs_to_be_used = tainted[:nums]
        tainted = tainted[nums:]
        ct = Sign(h, inputs_to_be_used, ct, nums[1])

    # Second: create not-tainted transactions:
    not_tainted = [x for x in list_of_inputs if x not in taint]
    while(len(not_tainted)>0):
        nums = getNumbers() # generate number of inputs and outptus for next txn.
        nums[0] = max(1, min(nums[0], len(not_tainted))
        inputs_to_be_used = not_tainted[:nums]
        not_tainted = not_tainted[nums:]
        ct = Sign(h, inputs_to_be_used, ct, nums[1])
    return ct
        
def getWalletDelay():
    # Pick a block height N+1, N+2, N+3, ... from a gamma distribution (continuous, measured in seconds) divided by 120, take the ceiling fcn, adding N
    pass

def getTaintDelay():
    # Pick a block height N+1, N+2, N+3, ... from any other distribution. Vary for testing purposes.
    pass

def getNumbers():
    # Pick number of inputs, numbers[0], and number of outputs, numbers[1], from an empirical distribution.
    numbers = [2,2] # For now, we will do this deterministically and we will put the empirical distro in later.
    return numbers
    

#### The main loop ####

t = 0
ct = 0 # counter for indexing new nodes; incremented every time a new node is added.

while(t < T):
    ct = addCoinbase(t, ct)
    ct = Spend(t, timesUp[t], ct)
    t += 1
    # Simple, right?

