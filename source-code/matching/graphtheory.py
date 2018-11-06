from collections import deque
import random

# I am lazy and use AssertionErrors for everything.

def sym_dif(x, y):
    ''' Compute symmetric difference between two lists'''
    z = []
    for e in x:
        if e not in y:
            z.append(e)
    for e in y:
        if e not in x:
            z.append(e)
    return z

class Node(object):
    par = None
    e = None

    def __init__(self, par):
        self.data = par['data']  # any
        self.ident = par['ident']  # str
        self.edges = []

    def _add_edge(self, edges):
        if type(edges) == type([]):
            for e in edges:
                if e not in self.edges:
                    self.edges.append(e)
        elif isinstance(edges, Edge):
            if edges not in self.edges:
                self.edges.append(edges)

    def _del_edge(self, e):
        # Remove an edge from self.edges. Return True if successful, false if edge was already removed.
        if e in self.edges:
            new_edges = [f for f in self.edges if f != e]
            self.edges = new_edges

 #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

class Edge(object):
    par = None

    def __init__(self, par):
        # Initialize with par = {'data':data, 'ident':ident, 'endpoints':[Node, Node], 'weight':weight}
        self.data = par['data']           # str
        self.ident = par['ident']         # str
        self.endpoints = par['endpoints'] # []
        self.weight = par['weight']       # None or float or int

 #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

class Graph(object):
    '''
    Graph object with attributes self.data, self.ident, self.nodes, self.edges, self.depth and with functions
    _add_node, _del_node, _add_edge, _del_edge, maximal_matching, _get_bigger_matching, _get_augmenting_paths
    '''
    par = None
    new_node = None
    old_node = None
    new_edge = None
    old_edge = None
    match = None

    def __init__(self, par):
        # Initialize with par = {'data':data, 'ident':ident, 'nodes':[], 'edges':[]
        self.data = par['data']   # str
        self.ident = par['ident'] # str
        self.nodes = {}
        for n in par['nodes']:
            self._add_node(n)
        self.edges = {}
        for e in par['edges']:
            self._add_edge(e)
        self.depth = 0

    def _add_node(self, new_node):
        # Add a new_node to self.nodes
        if new_node.ident not in self.nodes:
            self.nodes.update({new_node.ident:new_node})

    def _del_node(self, old_node):
        # Delete an old_node from self.nodes
        if old_node.ident in self.nodes:
            for e in old_node.edges:
                #print(e.ident)
                self._del_edge(e)
            del self.nodes[old_node.ident]

    def _add_edge(self, new_edge):
        # Add a new_edge to self.edges (and possibly its endpoints)
        #print("entering _add_edge")
        if new_edge.endpoints[0].ident not in self.nodes:
            #print("left endpoint not in nodes, adding to nodes")
            self._add_node(new_edge.endpoints[0])
        if new_edge.endpoints[1].ident not in self.nodes:
            #print("right endpoint not in nodes, adding to nodes")
            self._add_node(new_edge.endpoints[1])
        if new_edge.ident not in self.edges:
            #print("edge ident not in edges, adding to edges")
            self.edges.update({new_edge.ident:new_edge})
            new_edge.endpoints[0]._add_edge(new_edge)
            new_edge.endpoints[1]._add_edge(new_edge)

    def _del_edge(self, old_edge):
        # Remove an old_edge from self.edge
        if old_edge.ident in self.edges:
            del self.edges[old_edge.ident]

    def _get_bigger_matching(self, match=None):
        # Take as input some matching, find some augmenting paths, sym_dif them, and output the (bigger) result
        #print("beginning get_bigger_matching")
        if match is None:
            #print("I see we are starting with an emtpy match. random edge is added.")
            n = len(self.edges)
            i = random.randrange(n)
            ek = list(self.edges.keys())
            match = []
            match.append(self.edges[ek[i]])
        else:
            #print("I see we have a match. let's find an augmenting path.")
            aug_paths = self._get_augmenting_paths(match)
            for p in aug_paths:
                #print("sym-diffing augmenting path.")
                #print("match before = " + str([e.ident for e in match]) + " and path = " + str([e.ident for e in p]))
                match = sym_dif(p, match)
                #print("match after = " + str([e.ident for e in match]))
        return match

    def _get_augmenting_paths(self, match):
        # Want to find a shortest path starting with unmatched
        # edges with unmatched left endpoint, alternating between
        # matched and unmatched edges, terminating in an unmatched
        # edge whose right endpoint is unmatched.

        # Breadth-first search for an odd-length path terminating in an unmatched endpoint.

        result = []
        #print("Assemble matched nodes\n")
        matched_nodes_0 = [e.endpoints[0] for e in match]
        matched_nodes_1 = [e.endpoints[1] for e in match]
        #print("Shuffle unmatched nodes\n")
        unmatched_nodes = [self.nodes[n] for n in self.nodes if self.nodes[n] not in matched_nodes_0 and self.nodes[n] not in matched_nodes_1]
        s_unmatched_nodes_0 = []
        s_unmatched_nodes_1 = []
        disc_nodes = []
        for n in unmatched_nodes:
            if len(n.edges)==0:
                disc_nodes.append(n)
            else:
                if n == n.edges[0].endpoints[0] and n != n.edges[0].endpoints[1]:
                    s_unmatched_nodes_0.append(n)
                elif n != n.edges[0].endpoints[0] and n == n.edges[0].endpoints[1]:
                    s_unmatched_nodes_1.append(n)
                else:
                    print("WOOPWISE")
        random.shuffle(s_unmatched_nodes_0)
        random.shuffle(s_unmatched_nodes_1)

        #print("Filling queue")
        q = deque() # current level queue
        levs = deque() # queue of levels
        if len(s_unmatched_nodes_0) > 0:
            for n in s_unmatched_nodes_0:
                s_edges = n.edges
                random.shuffle(s_edges)
                for e in s_edges:
                    assert n == e.endpoints[0]
                    if e not in match:
                        this_path = []
                        this_path.append(e)
                        q.append(this_path)
            levs.append(q)

        #print("Beginning inductive loop")
        found = False
        while len(levs) > 0 and not found:
            q = levs.popleft()
            newq = deque()
            while len(q) > 0:
                this_path = q.popleft()
                b = len(this_path) % 2
                n = this_path[-1].endpoints[b]
                if b == 0:
                    s_edges = random.shuffle(n.edges)
                    for e in s_edges:
                        if e not in match:
                            next_path = this_path
                            next_path.append(e)
                            newq.append(next_path)
                elif b == 1:
                    if n not in matched_nodes_1:
                        assert n in s_unmatched_nodes_1
                        found = True
                        result.append(this_path)
                    else:
                        s_local_edges = n.edges
                        random.shuffle(s_local_edges)
                        for e in s_local_edges:
                            if e in match:
                                next_path = this_path
                                next_path.append(e)
                                newq.append(next_path)
            if len(newq) > 0:
                levs.append(newq)
        ## The following block of code tests the result is a valid augmenting path.
        for resulting_path in result:
            assert len(resulting_path)%2 == 1
            idx = 0
            while(idx < len(resulting_path)):
                assert resulting_path[idx] not in match
                if idx+1 < len(resulting_path):
                    assert resulting_path[idx] in match
                idx += 2
            assert resulting_path[0].endpoints[0] not in matched_nodes_0
            assert resulting_path[-1].endpoints[1] not in matched_nodes_1
        return result

    def maximal_matching(self):
        match = self._get_bigger_matching(None)
        assert type(match)==type([])
        assert len(match)==1
        next_match = self._get_bigger_matching(match)
        #print(match, type(match))
        #print(next_match, type(match))
        assert len(next_match) > len(match)
        while len(match) < len(next_match):
            match = next_match
            self.depth += 1
            #print("\n\n\t\t\t\tDEPTH = " + str(self.depth))
            next_match = self._get_bigger_matching(match)
        return match
