import random
from graphtheory import *
from math import *

x = 'the name of the graph'
par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'edges': []}
G = BipartiteGraph(par)

R = 11 # ring size
T = 100 # number blocks
M = 20 # trigger block height for weird behavior.
p = 0.1 # proportion of outputs sent to Alice

timesUp = [[] for i in range(T)] #timesUp[h] = outputs that shall be spent at block height h.
blockChain = [[[], []] for i in range(T)] #blockChain[h] = [[], []], blockChain[h][0] = inputs idents for new outputs created at block height h, blockChain[h][1] = signature idents for new signatures created at block height h.
taints = [[] for i in range(T)] # taints[h] = tainted outputs containted in block height h.

def createTxn(h, numbers, input_list, ct):
    [ct, sigNodes] = addSignatureNodes(input_list, ct) # add a ring signature node on the right for each input
    numbers[0] = min(numbers[0], len(input_list))
    possible_signers = getRings(numbers[0], input_list) # select ring members for each input
    in_edgeIds = addEdges(possible_signers, sigNodes) # add edges between all ring member nodes and their ring signature node
    [ct, newOuts] = addNewOutputs(numbers[1], ct) # add numbers[1] output nodes on the left
    out_edgeIds = addEdges(newOuts, sigNodes) # add edges between ring signature node and the new output nodes.
    input_list = input_list[numbers[0]:]
    done = (len(input_list)==0)
    return [ct, sigNodes, in_edgeIds, newOuts, out_edgeIDs, input_list, done]

def addSignatureNodes(h, input_list, ct):
    sigNodes = []
    for x in input_list:
        ct = str(int(ct)+1)
        par = {'data':None, 'ident':ct, 'edges':[]}
        next_right = Node(par)
        G._add_right(next_right)
        sigNodes.append(ct)
    return [ct, sigNodes]

def addNewOutputs(number_new_outs, ct):
    outNodes = []
    for i in range(number_new_outs):
        ct = str(int(ct) + 1)
        par = {'data':None, 'ident':ct, 'edges':[]}
        next_node = Node(par)
        G._add_left(next_node)
        outNodes.append(next_node.ident)
    return [ct, outNodes]

def addEdges(lefties, righties):
    new_idents = []
    for l in lefties:
        for r in righties:
            next_ident = str((l.ident, r.ident))
            par = {'data':None, 'ident':next_ident, 'left':G.left[l], 'right':G.right[r], 'weight':0.0}
            e = Edge(par)
            G._add_edge(e)
            new_idents.append(next_ident)
    return new_idents

def getRings(h, number, input_list):
    rings = []
    i = 0
    while i < number:
        next_input = input_list[i]
        rings.append(getRing(next_input))
    return rings

def getRing(input_ident):
    ring = []
    ring.append(input_ident)
    i = 1
    while i < R: # ring size
        i += 1
        s = getWalletDelay()
    

def Spend(inputs, ct):
    tainted = [x for x in inputs if x in taints]
    non_tainted = [x for x in inputs if x not in taints]
    numbers = getNumbers()
    [ct, sigNodes, in_edgeIds, newOuts, out_edgeIDs, input_list, done] = createTxn(numbers, tainted, ct)
    # assert len(input_list)==0
    while not done:
        numbers = getNumbers()
        numbers[0] = min(numbers[0], len(non_tainted))
        [ct, sigNodes, in_edgeIds, newOuts, out_edgeIDs, non_tainted, done] = createTxn(numbers, non_tainted, ct)

# Need: addSignatureNodes, getRings, addEdges, addNewOutputs, getNumbers, getWalletDelay
    
t = 0
ct = 0

while(t < T):
    # First, add a coinbase output.
    [ct, coinBase] = addNewOutputs(1, ct)
    blockChain[t]
    s = getWalletDelay() # measured in number of blocks
    u = random.random()
    if t+s < T:
        timesUp[t+s].append(x)
        blockChain[t].append(x)
        par = {'data':None, 'ident':x, 'edges':[]}
        node_to_add = Node(par)
        G._add_left(node_to_add)
        if u < p:
            taints.append(x)
    # Next, execute Spend(timesUp[t], ct)
    ct = Spend(timesUp[t], ct)
    t += 1


