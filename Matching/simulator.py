import random, math
from graphtheory import *
import numpy as np

class Simulator(object):
    par = None
    def __init__(self, par=None):
        if par is None or len(par) == 0:
            par = {'ring size':2, 'lock time':5, 'experiment start time':7, 'max sim time':200, 'prop taint':0.1, 'shape':19.28, 'rate':1.61}
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
            self.T = 1000000
        if 'prop taint' in par:
            self.p = par['prop taint']
        else:
            self.p = 0.001
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
        self.G = BipartiteGraph({'data':None, 'ident':None, 'left':[], 'right':[], 'red_edges':[], 'blue_edges':[]})
        self.timesUp = [[] for i in range(self.T)] #timesUp[h] = outputs that shall be spent at block height h.
        self.blockChain = [[[], []] for i in range(self.T)] #blockChain[h] = [[], []], blockChain[h][0] = idents for new outputs created at block height h, blockChain[h][1] = signature idents for new signatures created at block height h.
        self.taint = [] # list of tainted outputs
        self.trueEdges = []  # list of edge identities (not edges)
        self.currentHeight = 0
        self.label = 0
        
    def runSimulation(self, Tau=None):   
        if Tau is None:
            Tau = self.T
        rejected = False
        while(self.currentHeight < self.T and self.currentHeight < Tau and not rejected):
            rejected = rejected or self._spend()
            assert not rejected
            self.currentHeight += 1   
        self._report()
        return rejected

    def _spend(self):
        # breaks list_of_inputs into pieces called transactions
        # then calls Sign(-) on each piece.

        rejected = False

        # Collect the outputs to be spent in this block
        to_be_spent = self.timesUp[self.currentHeight]

        # Create coinbase transaction
        nums = self.getNumbers()
        rejected = rejected or self._sign(nums[1])
        assert not rejected

        # Next: create tainted transactions.
        tainted = [x for x in to_be_spent if x in self.taint]
        while(len(tainted) > 0 and not rejected):
            nums = self.getNumbers()
            nums[0] = max(1, min(nums[0], len(tainted))) # pick number of keys signed for in next tainted txn.
            inputs_to_be_used = tainted[:nums[0]]
            tainted = tainted[nums[0]:]
            rejected = rejected or self._sign(nums[1], inputs_to_be_used)
            assert not rejected

        # Last: create not-tainted transactions:
        not_tainted = [x for x in to_be_spent if x not in self.taint]
        while(len(not_tainted)>0 and not rejected):
            nums = self.getNumbers() # generate number of inputs and outptus for next txn.
            nums[0] = max(1, min(nums[0], len(not_tainted)))
            inputs_to_be_used = not_tainted[:nums[0]]
            not_tainted = not_tainted[nums[0]:]
            rejected = rejected or self._sign(nums[1], inputs_to_be_used)
            assert not rejected

        return rejected

    def _sign(self, numOuts, to_be_spent=[]):
        ''' This function is badly named. This function takes a number of new outputs and a list of inputs to be spent,
        adds them to the bipartite graph, selects ring members, and draws edges. A better name for this function 
        could be AddTxnToLedger or something.'''

        rejected = False
        out_idents = []
        sig_idents = []
        if len(to_be_spent) > 0: # 0 input transactions
            print("number of outputs to be spent okay")
            for i in to_be_spent:
                print("signing for input ", i)
                # Each input requires a ring signature node to be added on the right side of the graph
                self.label += 1
                print("label incremented")
                next_ident = self.label
                print("next ident set")
                sig_idents.append(next_ident)
                print("next ident appended")
                rejected = rejected or self.G.add_right_node(next_ident)
                assert not rejected
                ct = len(self.blockChain[self.currentHeight][1])
                self.blockChain[self.currentHeight][1].append(next_ident)
                assert ct + 1 == len(self.blockChain[self.currentHeight][1])
                print("new sigs added to blcokchain")
               
                # Now we pick ring members and assign those edges.
                print("calling get ring")
                next_ring = self.getRing(i) # For this signature, pick a ring
                print("Checking true spender is in ring")
                assert i in next_ring
                for ringMember in next_ring:
                    print("making new edge identity")
                    edge_ident = (int(ringMember), int(next_ident))
                    rejected = rejected or self.G.add_red_edge(edge_ident, 0.0)
                    assert not rejected
                    
                # We assign the true edge
                print("Assigning true edge")
                true_edge_ident = (int(i),int(next_ident))
                self.trueEdges.append(true_edge_ident) # Record true edge

        
            # place new output nodes on the left of G and determine their delays and whether they are tainted.
            print("place new output nodes on the left of G and determine their delays and whether they are tainted.")
            for i in range(numOuts):
                # create next new output
                print("Picking identity")
                self.label += 1
                next_ident = self.label
                print("Appending identity")
                out_idents.append(next_ident)
                
                print("Adding left node to G")
                rejected = rejected or self.G.add_left_node(next_ident)
                assert not rejected
                print("Adding to blockchain")
                self.blockChain[self.currentHeight][0].append(next_ident) # add to blockchain list

                # Determine if tainted
                print("Determining if tainted")
                u = random.random()
                s = None
                if u < self.p:
                    self.taint.append(next_ident)
                    if self.currentHeight >= self.M:
                        s = self.getTaintDelay()
                    else:
                        s = self.getWalletDelay()
                else:
                    s = self.getWalletDelay()

                print("Adding to timesUp")
                if s is not None and self.currentHeight+s < self.T:
                    self.timesUp[self.currentHeight+s].append(next_ident)
                if len(sig_idents)>0:
                    # add edges from signatures to these new outputs
                    print("Adding output (blue) edges")
                    # rando_counter = 0
                    for sig_ident in sig_idents:
                        # rando_counter += 1
                        edge_ident = (int(next_ident), int(sig_ident))
                        # e = Edge({'data':None, 'ident':edge_ident, 'left':self.G.left[next_ident], 'right':self.G.right_nodes[sig_ident], 'weight':0.0})
                        rejected = rejected or self.G.add_blue_edge(edge_ident, 0.0)
                        # print(rando_counter)
                        assert not rejected
        return rejected    

    def getWalletDelay(self):
        # Pick a block height N+1, N+2, N+3, ... from a gamma distribution (continuous, measured in seconds) divided by 120, take the ceiling fcn, but this distribution is clipped to not go below N
        x = max(self.N,math.ceil(math.exp(np.random.gamma(self.shape, self.scale))/self.timescale))
        while x > self.T:
            x = max(self.N,math.ceil(math.exp(np.random.gamma(self.shape, self.scale))/self.timescale))
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
        #if x > self.T:
        #    x = None
        return x

    def getNumbers(self):
        # Pick number of inputs, numbers[0], and number of outputs, numbers[1], from an empirical distribution.
        # To randomize, pick numbers[0] from Poisson(1) and numbers[1] from Poisson(2) - this is a biased model, skewed
        # small compared to the empirical distribution, and has a light tail. Two models that are unbiased but still
        # have skew and a light tail are:
        #   (Poisson(5.41), Poisson(6.14)) - Maximum likelihood estimate
        #   (NegBinom(71.900, 0.070), NegBinom(38.697,0.137)) - Method of moments
        numbers = [1,2] # For now, we will do this deterministically and we will put the empirical distro in later.
        return numbers

    def getRing(self, signers_ident):
        ''' Pick a ring containing signers_ident '''
        result = []
        result.append(signers_ident)
        done = False
        while(len(result)<self.R and not done):
            #print("Picking age of next output")
            age = None
            while age is None:
                age = self.getWalletDelay() # always >= self.N
            assert age is not None
            bottom = self.currentHeight-max(self.T,age+self.N)
            top = self.currentHeight-min(age-self.N,self.N)
            choices = []
            assert len(range(bottom,top+1))>0
            choices = [y for i in range(bottom, top+1) for y in self.blockChain[i][0]]
            #for i in range(bottom,top+1):
            #    for y in self.blockChain[i][0]:
            #        #print("Verifying choice ", y, "in blockchain is in left-nodes")                    
            #        assert y in self.G.left_nodes
            #        #try:
            #        #    assert y in self.G.left_nodes
            #        #except AssertionError:
            #        #    li = [str(z) for z in sorted(list(self.G.left_nodes.keys()))]
            #        #    #print("A ring member y was selected that is not in G.left_nodes; an element in blockChain[" + str(i) + "][0] is not in G.left.")
            #        #    #print("Here is G.left_nodes.keys: \n")
            #        #    #for entry in li:
            #        #    #    print(entry)
            #        #    #print("And here is the entry of blockChain: ")
            #        #    #print(self.blockChain[i][0])
            #        #print("Appending choice to choices")
            #        choices.append(y)
            if len(choices)>0:
                #print("Picking from choices")
                x = random.choice(choices)
                while x in result:
                    #print("Woops, choice has already been selected as a ring member. Reselect.")
                    x = random.choice(choices)
                assert x not in result
                result.append(x)
                #print("Deduplicating as a precaution")
                result = list(dict.fromkeys(result)) # dedupe
            else:
                done = True
        return result

    def _report(self):
        with open("output.txt","w") as wf:
            wf.write("Description of G ====\n\n")
            wf.write("Left-nodes key list:\n")
            ordList = sorted(list(self.G.left_nodes.keys()))
            wf.write(str(ordList)+ "\n\n")
            wf.write("Right-nodes key list:\n")
            ordList = sorted(list(self.G.right_nodes.keys()))
            wf.write(str(ordList)+ "\n\n")
            wf.write("in_edges:\n")
            ordList = sorted(list(self.G.red_edge_weights.keys()))
            wf.write("out_edges:\n")
            ordList = sorted(list(self.G.blue_edge_weights.keys()))
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
            for sig_node in self.G.right_nodes:
                wf.write("Signature with ident = " + str(sig_node) + "," + str(type(sig_node)) + "  has ring members:\n")
                for edge_ident in self.G.red_edge_weights:
                    #print(" edge ident = " + str(edge_ident) + " has type " + str(type(edge_ident)) + "\n")
                    if sig_node==edge_ident[1]:
                        wf.write("\t" + str(edge_ident[0])+"\n")
                wf.write("\t and is associated with outputs:\n")
                for edge_ident in self.G.blue_edge_weights:
                    #print(" edge ident = " + str(edge_ident) + " has type " + str(type(edge_ident)) + "\n")
                    if sig_node==edge_ident[1]:
                        wf.write("\t" + str(edge_ident[0])+"\n")


sally = Simulator(None)
sally.runSimulation()

