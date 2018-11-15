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
            for n in old_edge.endpoints:
                n._del_edge(old_edge)
            del self.edges[old_edge.ident]

    def _check_match(self, alleged_match):
        tn = []
        ismatch = True
        if alleged_match is not None and len(alleged_match) > 0:
            for e in alleged_match:
                if e.endpoints[0] in tn or e.endpoints[1] in tn:
                    ismatch = False
                else:
                    tn.append(e.endpoints[0])
                    tn.append(e.endpoints[1])
        return ismatch

    def _disjoint(self, set_one, set_two):
        set_one_nodes = [e.endpoints[0] for e in set_one] + [e.endpoints[1] for e in set_one]
        set_two_nodes = [e.endpoints[0] for e in set_two] + [e.endpoints[1] for e in set_two]
        isect = [x for x in set_one_nodes if x not in set_two_nodes]
        return len(isect)==0

    def maximal_matching(self, wt=False):
        #print("Finding match.")
        # Iteratively call _get_bigger_matching until you stop getting bigger matches.
        assert self.check_match(None)
        match = self._get_bigger_matching(None, wt)
        assert self.check_match(match)
        next_match = self._get_bigger_matching(match, wt)
        assert self.check_match(next_match)
        while len(match) < len(next_match):
            match = next_match
            self.depth += 1
            next_match = self._get_bigger_matching(match, wt)
            assert self.check_match(next_match)
        return match

    def _get_bigger_matching(self, match=None, wt=False):
        # Implements Hopcroft-Karp algorithm.
        # Take as input some matching, find some augmenting paths, sym_dif them, and output the (bigger) result
        touched_nodes = []
        touched_nodes = None
        #print("beginning get_bigger_matching")
        result = []
        if match is None:
            #print("I see we are starting with an emtpy match. random edge is added.")
            if wt:
                weighted_edges = sorted([(e.weight, e) for e in self.edges], key=lambda x:x[0], reverse=True)
                result.append(weighted_edges[0])
                assert self.check_match(result)
            else:
                i = random.randrange(len(self.edges))
                ek = list(self.edges.keys())[i]
                e = self.edges[ek]
                result.append(e)
                assert self.check_match(result)
        else:
            #print("I see we have a match. let's find an augmenting path.")
            assert self.check_match(match)
            aug_paths = self._get_augmenting_paths(match, wt)
            if len(aug_paths) == 0:
                result = match
                assert self.check_match(result)
            elif len(aug_paths) > 0:
                result = match
                assert self.check_match(result)
                for p in aug_paths:
                    for q in aug_paths:
                        assert self.disjoint(p,q)
                for p in aug_paths:
                    assert self.check_match(result)
                    result = sym_dif(result, p)
                    assert self.check_match(result)
                assert self.check_match(result)
        return result

    def _get_augmenting_paths(self, match, wt=False):
        bfs = [] # results from breadth-first search
        matched_nodes_left = [e.endpoints[0] for e in match]
        matched_nodes_right = [e.endpoints[1] for e in match]
        unmatched_nodes = [n for n in list(self.nodes.values()) if n not in matched_nodes_left and n not in matched_nodes_right]

        q = deque()

        for n in unmatched_nodes:
            for e in n.edges:
                if e.endpoints[0] == n:
                    assert e not in match
                    this_path = []
                    this_path.append(e)
                    q.append(this_path)

        found = False
        while len(q) > 0 and not found:
            this_path = q.popleft()
            if len(this_path)%2 == 1:
                endpt = this_path[-1].endpoints[1]
                if endpt in unmatched_nodes:
                    found = True
                    bfs.append(this_path)
                else:
                    assert endpt in matched_nodes_right
                    for e in endpt.edges:
                        if e in match:
                            next_path = this_path
                            next_path.append(e)
                            q.append(next_path)
            if len(this_path)%2 == 0:
                endpt = this_path[-1].endpoints[0]
                assert endpt in matched_nodes_left
                for e in endpt.edges:
                    if e not in match:
                        next_path = this_path
                        next_path.append(e)
                        q.append(next_path)

        if wt:
            weighted_bfs = []
            for p in bfs:
                s = 0.0
                for e in p:
                    s += e.weight
                weighted_bfs.append([s,p])
            sorted_weighted_bfs = sorted(weighted_bfs, key=lambda x:x[0], reverse=True)
            bfs = [x[1] for x in sorted_weighted_bfs]
        else:
            random.shuffle(bfs)

        results = []
        touched_nodes = []
        for p in bfs:
            bad_path = False
            for e in p:
                if e.endpoints[0] in touched_nodes or e.endpoints[1] in touched_nodes:
                    bad_path = True
                    break
            for i in range(len(p)):
                j = i % 2
                if j == 0 and p[i] not in match:
                    bad_path = True
                    print("WHAT")
                if j == 1 and p[i] in match:
                    bad_path = True
                    print("WHAT")
            if not bad_path:
                results.append(p)
                for e in p:
                    touched_nodes.append(e.endpoints[0])
                    touched_nodes.append(e.endpoints[1])
        return results








    def _old_get_augmenting_paths(self, match):
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

        touched_nodes = []
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
        touched_edges = {}
        found = False
        while len(levs) > 0 and not found:
            #print("Currently have " + str(len(levs)) + " levels stored in queue")
            q = levs.popleft()
            newq = deque()
            while len(q) > 0:
                #print("current length of newq=" + str(len(newq)))
                this_path = q.popleft()
                b = len(this_path) % 2
                n = this_path[-1].endpoints[b]
                if b == 0:
                    if len(n.edges) > 0:
                        s_edges = n.edges
                        random.shuffle(s_edges)
                        for e in s_edges:
                            if e not in match and e.ident not in touched_edges:
                                next_path = this_path
                                next_path.append(e)
                                newq.append(next_path)
                                touched_edges.update({e.ident:e})
                    #else: # In this case, the path is a failure and is not re-appended to the queue.
                elif b == 1:
                    if n not in matched_nodes_1:
                        assert n in s_unmatched_nodes_1
                        found = True
                        result.append(this_path)
                    else:
                        s_local_edges = n.edges
                        random.shuffle(s_local_edges)
                        for e in s_local_edges:
                            if e in match and e.ident not in touched_edges:
                                next_path = this_path
                                next_path.append(e)
                                newq.append(next_path)
                                touched_edges.update({e.ident:e})
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

    def _get_augmenting_cycles(self, match):
        # Pettie and Sanders method
        # Find all max-gain augmenting 4-cycles

        result = []
        uncycled_edges = deque(sorted([self.edges[e] for e in self.edges if self.edges[e] not in match], key=lambda x:x.weight, reverse=True))
        touched_nodes = []
        touched_edges = []
        while(len(uncycled_edges) > 0):
            next_edge = uncycled_edges.popleft()
            uleft_endpoint = next_edge.endpoints[0]
            uright_endpoint = next_edge.endpoints[1]

            if uleft_endpoint not in touched_nodes and uright_endpoint not in touched_nodes and next_edge not in touched_edges:
                # Find max-gain 4-cycle passing through this edge.
                left_edge_candidates = []
                for e in uleft_endpoint.edges:
                    if e in match and e.endpoints[1] not in touched_nodes and e not in touched_edges:
                        left_edge_candidates.append(e)
                right_edge_candidates = []
                for e in uright_endpoint.edges:
                    if e in match and e.endpoints[1] not in touched_nodes and e not in touched_edges:
                        right_edge_candidates.append(e)

                cycles_of_max_gain = [[], None]
                for le in left_edge_candidates:
                    for re in right_edge_candidates:
                        lleft_endpoint = re.endpoints[0]
                        lright_endpoint = le.endpoints[1]
                        if lleft_endpoint not in touched_nodes and lright_endpoint not in touched_nodes:
                            for e in lleft_endpoint.edges:
                                if e not in match and e.endpoints[1] == lright_endpoint and e not in touched_edges:
                                    gain = e.weight + next_edge.weight - le.weight - re.weight
                                    cycle = [next_edge, re, e, le]
                                    if(len(cycles_of_max_gain[0]) == 0):
                                        cycles_of_max_gain[0].append(cycle)
                                        cycles_of_max_gain[1] = gain
                                    elif(cycles_of_max_gain[1] < gain):
                                        cycles_of_max_gain[0] = []
                                        cycles_of_max_gain[0].append([next_edge, re, e, le])
                                        cycles_of_max_gain[1] = gain
                                    elif(cycles_of_max_gain[1] == gain):
                                        cycle_to_add = [next_edge, re, e, le]
                                        cycles_of_max_gain[0].append(cycle_to_add)
                                    else:
                                        pass

                # Touch edges and nodes of this cycle
                if len(cycles_of_max_gain[0]) > 0:
                    chosen_cycle = random.choice(cycles_of_max_gain[0])
                    result.append(chosen_cycle)
                    for e in chosen_cycle:
                        if e not in touched_edges:
                            touched_edges.append(e)
                        for v in e.endpoints:
                            if v not in touched_nodes:
                                touched_nodes.append(v)
                            # Also touch all edges adjacent to v to ensure
                            # vertex disjointness
                            for f in v.edges:
                                if f not in touched_edges:
                                    touched_edges.append(f)
        return result

def make_graph(i,r,wt=None):
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

    if wt is not None:
        for eid in G.edges:
            e = G.edges[eid]
            e.weight = random.random()

    return G