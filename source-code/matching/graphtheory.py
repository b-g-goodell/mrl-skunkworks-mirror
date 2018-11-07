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
    ''' Node object representing a vertex in a graph
    Attributes:
        data : arbitrary
        ident : string
        edges : list
    Functions:
        _add_edge : take edge(s) as input and add to self.edges
        _del_edge : take edge as input and remove from self.edges
    '''
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
    ''' Edge object representing an edge in a graph. Essentially a container.
    Attributes:
        data : arbitrary
        ident : string
        endpoints : list length 2
        weight : None, float, int
    '''
    par = None

    def __init__(self, par):
        # Initialize with par = {'data':data, 'ident':ident, 'endpoints':[Node, Node], 'weight':weight}
        self.data = par['data']           # str
        self.ident = par['ident']         # str
        self.endpoints = par['endpoints'] # []
        self.weight = par['weight']       # None or float or int

 #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

class Graph(object):
    ''' Graph object representing a graph
    Attributes:
        data : arbitrary
        ident : string
        self.nodes : dict
        edges : dict
        depth : int
    Functions:
        _add_node : take node as input and add to self.nodes
        _del_node : take node as input, removes incident edges from self.edges, removes input from self.nodes
        _add_edge : take edge(s) as input and add to self.edges, adding endpoints to self.nodes if necessary
        _del_edge : take edge as input and remove from self.edges
        _get_augmenting_path : take a match as input and return an augmenting path as output.
        _get_bigger_matching : take match as input and return a larger match
        maximal_matching : iteratively call _get_bigger_matching until matches of same size are returned
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
        if new_edge not in new_edge.endpoints[0].edges:
            new_edge.endpoints[0]._add_edge(new_edge)
        if new_edge not in new_edge.endpoints[1].edges:
            new_edge.endpoints[1]._add_edge(new_edge)
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
        # Implements Hopcroft-Karp algorithm.
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
        # Implements Ford-Fulkerson method for bipartite graphs (citation?)

        # Want to find a shortest path whose endpoints are unmatched and edges alternate between matched and
        # unmatched. Breadth-first search for an odd-length path beginning and ending in an unmatched endpoint.

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
        # Iteratively call _get_bigger_matching until you stop getting bigger matches.
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


def make_graph(i,r):
    # Create a bipartite graph with 2*i nodes such that all right-hand nodes are r-regular.
    s = random.random()
    x = str(hash(str(1.0) + str(s)))
    par = {'data': 1.0, 'ident': x, 'nodes': [], 'edges': []}
    G = Graph(par)
    N = i  # nodeset size
    K = r  # neighbor size

    ct = 0
    while len(G.nodes) < 2 * N:
        while str(ct) in G.nodes:
            ct += 1
        par = {'data': 1.0, 'ident': str(ct)}
        n = Node(par)
        G._add_node(n)

    nodekeys = list(G.nodes.keys())
    for i in range(N):
        sig_idx = nodekeys[i]
        right_node = G.nodes[sig_idx]
        idxs = random.sample(range(N), K)
        assert len(idxs) == K
        for j in idxs:
            otk_idx = nodekeys[j + N]
            left_node = G.nodes[otk_idx]
            x = left_node.ident + "," + right_node.ident
            par = {'data': 1.0, 'ident': x, 'endpoints': [left_node, right_node], 'weight': 0}
            e = Edge(par)
            G._add_edge(e)
    return G