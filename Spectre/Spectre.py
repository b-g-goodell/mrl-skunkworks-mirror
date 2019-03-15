from math import *
from collections import *
import random
from copy import *

class Node(object):

    def __init__(self,params):
        # Passes unit tests
        self.label = params['label']
        self.payload = params['payload']
        self.parents = params['parents']
        self.children = {}

class DirAcyGraph(object):

    def __init__(self,params=None):
        # Passes unit tests
        self.nodes = {}
        self.leaves = {}

    def _add_node(self, params):
        # Passes unit tests
        assert params['label'] not in self.nodes and params['label'] not in self.leaves
        parentsExistOrHasNoParents = True
        for p in params['parents']:
            parentsExistOrHasNoParents = parentsExistOrHasNoParents and p in self.nodes
        if parentsExistOrHasNoParents:
            x = Node(params)
            self.nodes.update({params['label']:x})
            self.leaves.update({params['label']:x})
            for p in x.parents:
                if p in self.leaves:
                    del self.leaves[p]
                if params['label'] not in self.nodes[p].children:
                    self.nodes[p].children.update({params['label']:x})

    def _rem_leaf(self, leafLabel):
        # Passes unit tests
        assert leafLabel in self.leaves and leafLabel in self.nodes
        for p in self.nodes[leafLabel].parents:
            assert p in self.nodes
            if leafLabel in self.nodes[p].children:
                del self.nodes[p].children[leafLabel]
            if len(self.nodes[p].children) == 0:
                self.leaves.update({p:self.nodes[p]})
        del self.leaves[leafLabel]
        del self.nodes[leafLabel]

    def _get_past(self, startLabel):
        # Passes unit tests
        allowableLabels = []
        Q = deque()
        Q.append(startLabel)
        while(len(Q)>0):
            nextLabel = Q.popleft()
            for p in self.nodes[nextLabel].parents:
                if p not in allowableLabels:
                    allowableLabels.append(p)
                Q.append(p)
        G = deepcopy(self)
        touched = []
        for nextLabel in G.leaves:
            Q.append(nextLabel)
        while(len(Q)>0):
            nextLabel = Q.popleft()
            if nextLabel not in touched and nextLabel not in allowableLabels:
                assert nextLabel in G.leaves
                touched.append(nextLabel)
                for p in G.nodes[nextLabel].parents:
                    Q.append(p)
                G._rem_leaf(nextLabel)
        return G

    def _short_spec(self, state):
        # Not functional yet
        pass


    def _spec(self):
        # Passing a basic unit test
        result = {}

        if len(self.nodes)==1:
            assert len(self.leaves)==len(self.nodes)
            onlyNode = list(self.nodes.keys())[0]
            result = {(onlyNode, onlyNode):1}
        else:
            Q = deque()
            touched = []
            votes = {}
            futureVotes = {}
            for z in self.leaves:
                if z not in touched:
                    touched.append(z)
                    Q.append(z)
            while(len(Q)>0):
                z = Q.popleft()
                for p in self.nodes[z].parents:
                    if p not in touched:
                        touched.append(p)
                        Q.append(p)
                pastZ = self._get_past(z)
                rVote = pastZ._spec()
                for y in self.nodes:
                    for x in self.nodes:
                        votes.update({(z,x,y):0, (z,y,x):0})
                        if x == y:
                            votes.update({(z,x,x):1})
                        else:
                            if x in pastZ.nodes and y in pastZ.nodes:
                                assert rVote[(x,y)] in [-1, 0, 1] and rVote[(y,x)] in [-1,0,1]
                                votes.update({(z,x,y):rVote[(x,y)], (z,y,x):rVote[(y,x)]})
                            elif x in pastZ.nodes and y not in pastZ.nodes:
                                votes.update({(z,x,y):1, (z,y,x):-1})
                            elif x not in pastZ.nodes and y in pastZ.nodes:
                                votes.update({(z,x,y):-1, (z,x,y):1})
                            elif (z,x,y) in futureVotes or (z,y,x) in futureVotes:
                                eventZero = (z,x,y) in futureVotes and (z,y,x) in futureVotes
                                eventOne = (futureVotes[(z,x,y)] > 0 and futureVotes[(z,y,x)] < 0)
                                eventTwo = (futureVotes[(z,x,y)] < 0 and futureVotes[(z,y,x)] > 0)
                                eventThree = (futureVotes[(z,x,y)]==0 and futureVotes[(z,x,y)]==0)
                                try:
                                    assert eventZero or eventOne or eventTwo or eventThree 
                                except AssertionError:
                                    print(eventZero, eventOne, eventTwo, eventThree, futureVotes[(z,x,y)])
                                if futureVotes[(z,x,y)] > 0 and futureVotes[(z,y,x)] < 0:
                                    votes.update({(z,x,y):1, (z,y,x):-1})
                                elif futureVotes[(z,x,y)] < 0 and futureVotes[(z,y,x)] > 0:
                                    votes.update({(z,x,y):-1, (z,y,x):1})
                                elif futureVotes[(z,x,y)]==0 and futureVotes[(z,y,x)]==0:
                                    if x < y:
                                        votes.update({(z,x,y):1, (z,y,x):-1})
                                    elif x > y:
                                        votes.update({(z,x,y):-1, (z,y,x):1})
                                    else:
                                        #print("SHIT")
                                        pass
                            else:
                                # No blocks from future of z vote on x,y: z has no opinion.
                                pass
                        for w in pastZ.nodes:
                            if (w,x,y) not in futureVotes:
                                futureVotes.update({(w,x,y):votes[(z,x,y)]})
                            else:
                                futureVotes[(w,x,y)] += votes[(z,x,y)]
                            if (w,y,x) not in futureVotes:
                                futureVotes.update({(w,y,x):votes[(z,y,x)]})
                            else:
                                futureVotes[(w,y,x)] += votes[(z,y,x)]

            for x in self.nodes:
                for y in self.nodes:
                    s = 0
                    for z in self.nodes:
                        s += votes[(z,x,y)]
                    if s > 0 or s == 0 and x < y:
                        result.update({(x,y):1, (y,x):-1})
                    elif s < 0 or s == 0 and x > y:
                        result.update({(x,y):-1, (y,x):1})
            
        return result
                                    
                
