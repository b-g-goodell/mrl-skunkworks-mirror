import random, math
from graphtheory import *
import numpy as np

class Simulator(object):
    par = None
    def __init__(self, par=None):
        if par is None:
            par = {'ring size':3, 'lock time':5, 'experiment start time':7, 'max sim time':41, 'prop taint':0.1, 'shape':19.28, 'rate':1.61}
        if 'ring size' in par:
            self.R = par['ring size']
        else:
            self.R = 11
        if 'lock time' in par:
            self.N = par['lock time']
        else:
            self.N = 10
        if 'experiment start time' in par:
            self.M = par['experiment start time']
        else:
            self.M = 100
        if 'max sim time' in par:
            self.T = par['max sim time']
        else:
            self.T = 10000
        if 'prop taint' in par:
            self.p = par['prop taint']
        else:
            self.p = 0.01
        if 'shape' in par:
            self.shape = par['shape']
        else:
            self.shape = 19.28
        if 'rate' in par:
            self.rate = par['rate']
        else:
            self.rate = 1.61
        if 'timescale' in par:
            self.timescale = par['timescale']
        else:
            self.timescale = 120.0 # number of seconds per block
        self.scale = 1.0/self.rate
        self.G = BipartiteGraph({'data':None, 'ident':None, 'left':[], 'right':[], 'in_edges':[], 'out_edges':[]})
        self.timesUp = [[] for i in range(self.T)] #timesUp[h] = outputs that shall be spent at block height h.
        self.blockChain = [[[], []] for i in range(self.T)] #blockChain[h] = [[], []], blockChain[h][0] = idents for new outputs created at block height h, blockChain[h][1] = signature idents for new signatures created at block height h.
        self.taint = [] # list of tainted outputs
        self.trueEdges = [] 
        self.currentHeight = 0
        self.label = 0
        
    def runSimulation(self):   
        while(self.currentHeight < self.T):
            self.addCoinbase()
            self.Spend(self.timesUp[self.currentHeight])
            self.currentHeight += 1
        # Simple, right?
        with open("output.txt","w") as wf:
            wf.write("Description of G ====\n\n")
            wf.write("Left-nodes key list:\n")
            ordList = sorted(list(self.G.left.keys()))
            wf.write(str(ordList)+ "\n\n")
            wf.write("Right-nodes key list:\n")
            ordList = sorted(list(self.G.right.keys()))
            wf.write(str(ordList)+ "\n\n")
            wf.write("in_edges:\n")
            ordList = sorted(list(self.G.in_edges.keys()))
            wf.write("out_edges:\n")
            ordList = sorted(list(self.G.out_edges.keys()))
            wf.write(str(ordList)+ "\n\n")
            wf.write("Blockchain representation:\n")
            for h in range(len(self.blockChain)):
                block = self.blockChain[h]
                line = "\tHeight = " + str(h) + ":\n"
                wf.write(line)
                wf.write("\t\t OTKeys :\n")
                for entry in self.blockChain[h][0]:
                    wf.write("\t\t\t" + str(entry) + "\n")
                wf.write("\t\t Signatures :\n")
                for entry in self.blockChain[h][1]:
                    wf.write("\t\t\t" + str(entry) + "\n")
            wf.write("\n")
            wf.write("Ring signature representation:\n")
            for sig_node in self.G.right:
                wf.write("Signature with ident = " + str(sig_node) + "," + str(type(sig_node)) + "  has ring members:\n")
                for edge_ident in self.G.in_edges:
                    #print(" edge ident = " + str(edge_ident) + " has type " + str(type(edge_ident)) + "\n")
                    if sig_node==edge_ident[1]:
                        wf.write("\t" + str(edge_ident[0])+"\n")
                wf.write("\t and is associated with outputs:\n")
                for edge_ident in self.G.out_edges:
                    #print(" edge ident = " + str(edge_ident) + " has type " + str(type(edge_ident)) + "\n")
                    if sig_node==edge_ident[1]:
                        wf.write("\t" + str(edge_ident[0])+"\n")
        pass

    def addCoinbase(self):
        # Create a new coinbase transaction, add it to the graph, record it on the blockchain, pick a time delay for spending, determine if
        # the miner is taintman, output our rolling index ct for naming nodes.
        next_ident = self.label
        self.label += 1        
        node_to_add = Node({'data':None, 'ident':next_ident, 'edges':[]})
        self.G._add_left(node_to_add)
        self.blockChain[self.currentHeight][0].append(next_ident)
        
        s = self.getWalletDelay() # measured in number of blocks
        if s is not None:
            if self.currentHeight+s < self.T:
                self.timesUp[self.currentHeight+s].append(next_ident)

        u = random.random()
        if u < self.p:
            self.taint.append(next_ident)
        pass

    def Sign(self, list_of_inputs, numOuts):
        # place new signatures, one for each input, record true edge, pick ring members, add fake edges
        sig_idents = []
        for i in list_of_inputs:
            # Each input requires a ring signature node to be added on the right side of the graph
            next_ident = self.label
            self.label += 1
            sig_idents.append(next_ident)
            sig_node_to_add = Node({'data':None, 'ident':next_ident, 'edges':[]})
            self.G._add_right(sig_node_to_add)
            self.blockChain[self.currentHeight][1].append(next_ident)
            
            # Now we pick ring members and assign those edges.
            next_ring = self.getRing(i) # For this signature, pick a ring
            #print("next_ring = ", next_ring)
            assert i in next_ring
            for ringMember in next_ring:
                edge_ident = (int(ringMember), int(next_ident))
                e = Edge({'data':None, 'ident':edge_ident, 'left':self.G.left[ringMember], 'right':self.G.right[next_ident], 'weight':0.0})
                temp = len(self.G.in_edges)
                self.G._add_in_edge(e)
                assert len(self.G.in_edges) - temp > 0
                
            # We can immediately assign the true edge
            true_edge_ident = (int(i),int(next_ident))
            self.trueEdges.append(true_edge_ident) # Record true edge

        # place new output nodes on the left of G and determine their delays and whether they are tainted.
        out_idents = []
        for i in range(numOuts):
            # create next new output
            next_ident = self.label
            self.label += 1
            out_idents.append(next_ident)
            
            out_node_to_add = Node({'data':None, 'ident':next_ident, 'edges':[]})
            self.G._add_left(out_node_to_add)
            self.blockChain[self.currentHeight][0].append(next_ident) # add to blockchain list

            # Determine if tainted
            u = random.random()
            if u < self.p:
                self.taint.append(next_ident)
                if self.currentHeight >= self.M:
                    s = self.getTaintDelay()
                else:
                    s = self.getWalletDelay()
            else:
                s = self.getWalletDelay()
            if s is not None:
                if self.currentHeight+s < self.T:
                    self.timesUp[self.currentHeight+s].append(next_ident)

            # add edges from signatures to these new outputs
            for sig_ident in sig_idents:
                edge_ident = (int(next_ident), int(sig_ident))
                e = Edge({'data':None, 'ident':edge_ident, 'left':self.G.left[next_ident], 'right':self.G.right[sig_ident], 'weight':0.0})
                self.G._add_out_edge(e)
        pass

    def Spend(self, list_of_inputs):
        # breaks list_of_inputs into pieces called transactions
        # then calls Sign(-) on each piece

        # First: create tainted transactions.
        tainted = [x for x in list_of_inputs if x in self.taint]
        while(len(tainted) > 0):
            nums = self.getNumbers()
            nums[0] = max(1, min(nums[0], len(tainted))) # pick number of keys signed for in next tainted txn.
            inputs_to_be_used = tainted[:nums[0]]
            tainted = tainted[nums[0]:]
            self.Sign(inputs_to_be_used, nums[1])

        # Second: create not-tainted transactions:
        not_tainted = [x for x in list_of_inputs if x not in self.taint]
        while(len(not_tainted)>0):
            nums = self.getNumbers() # generate number of inputs and outptus for next txn.
            nums[0] = max(1, min(nums[0], len(not_tainted)))
            inputs_to_be_used = not_tainted[:nums[0]]
            not_tainted = not_tainted[nums[0]:]
            self.Sign(inputs_to_be_used, nums[1])
        pass

    def getWalletDelay(self):
        # Pick a block height N+1, N+2, N+3, ... from a gamma distribution (continuous, measured in seconds) divided by 120, take the ceiling fcn, but this distribution is clipped to not go below N
        x = max(self.N,math.ceil(math.exp(np.random.gamma(self.shape, self.scale))/self.timescale))
        if x > self.T:
            x = None
        return x

    def getTaintDelay(self, ratio=1.0):
        # Pick a block height N+1, N+2, N+3, ... from any other distribution. Vary for testing purposes.

        # PERIODIC DAILY PURCHASE BEHAVIOR:
        x = 720

        # UNIFORM DAILY CHURNING:
        #x = random.randint(N,720)

        # GAMMA WITH SAME SHAPE BUT SMALLER RATE PARAM = SLOWER SPEND TIME
        #ratio = 0.5
        #x = min(self.N, math.ceil(math.exp(np.random.gamma(self.shape, self.scale/ratio))/self.timescale))

        # GAMMA WITH SAME SPEND TIME BUT GREATER SHAPE = SLOWER SPEND TIME
        #ratio = 2.0
        #x = min(self.N, math.ceil(math.exp(np.random.gamma(ratio*self.shape, self.scale))/self.timescale))
        if x > self.T:
            x = None
        return x

    def getNumbers(self):
        # Pick number of inputs, numbers[0], and number of outputs, numbers[1], from an empirical distribution.
        numbers = [2,2] # For now, we will do this deterministically and we will put the empirical distro in later.
        return numbers

    def getRing(self, signers_ident):
        result = []
        result.append(signers_ident)
        while(len(result)<self.R):
            age = self.getWalletDelay() # always >= self.N
            if age is not None:
                bottom = self.currentHeight+max(-self.T,-age-self.N)
                top = self.currentHeight+min(-age+self.N,-self.N)
                choices = []
                for i in range(bottom,top+1):
                    for y in self.blockChain[i][0]:
                        try:
                            assert y in self.G.left
                        except AssertionError:
                            li = [str(z) for z in sorted(list(self.G.left.keys()))]
                            print("A ring member y was selected that is not in G.left; an element in blockChain[" + str(i) + "][0] is not in G.left.")
                            print("Here is G.left.keys: \n")
                            for entry in li:
                                print(entry)
                            print("And here is the entry of blockChain: ")
                            print(self.blockChain[i][0])
                        choices.append(y)
                if len(choices)>0:
                    result.append(random.choice(choices))
        result = list(dict.fromkeys(result)) # dedupe
        return result

sally = Simulator(None)
sally.runSimulation()

