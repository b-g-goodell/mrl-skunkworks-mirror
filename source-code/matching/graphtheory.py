from collections import deque
import random

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
        self.data = par['data']     # str
        self.ident = par['ident']   # str
        self.left = par['left']     # Node
        self.right = par['right']   # Node
        self.weight = par['weight'] # None or float or int

 #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

class BipartiteGraph(object):
    ''' Graph object representing a graph
    Attributes:
        data : arbitrary
        ident : string
        left : dict
        right : dict
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

    def __init__(self, par):
        # Initialize with par = {'data':data, 'ident':ident, 'nodes':[], 'edges':[]
        self.data = par['data']   # str
        self.ident = par['ident'] # str
        self.left = {}
        for n in par['left']:
            self._add_left(n)
        self.right = {}
        for n in par['right']:
            self._add_right(n)
        self.edges = {}
        for e in par['edges']:
            self._add_edge(e)
        self.depth = 0

    def _add_left(self, new_node):
        # Add a new_node to self.nodes
        if new_node.ident not in self.left:
            self.left.update({new_node.ident:new_node})

    def _add_right(self, new_node):
        # Add a new_node to self.nodes
        if new_node.ident not in self.right:
            self.right.update({new_node.ident:new_node})

    def _del_node(self, old_node):
        # Delete an old_node from self.nodes
        if old_node.ident in self.left:
            for e in old_node.edges:
                #print(e.ident)
                self._del_edge(e)
            del self.left[old_node.ident]
        elif old_node.ident in self.right:
            for e in old_node.edges:
                #print(e.ident)
                self._del_edge(e)
            del self.right[old_node.ident]

    def _add_edge(self, new_edge):
        # Add a new_edge to self.edges (and possibly its endpoints)
        #print("entering _add_edge")
        if new_edge not in new_edge.left.edges:
            new_edge.left._add_edge(new_edge)
        if new_edge not in new_edge.right.edges:
            new_edge.right._add_edge(new_edge)
        if new_edge.left.ident not in self.left:
            #print("left endpoint not in nodes, adding to nodes")
            self._add_left(new_edge.left)
        if new_edge.right.ident not in self.right:
            #print("right endpoint not in nodes, adding to nodes")
            self._add_right(new_edge.right)
        if new_edge.ident not in self.edges:
            #print("edge ident not in edges, adding to edges")
            self.edges.update({new_edge.ident:new_edge})

    def _del_edge(self, old_edge):
        # Remove an old_edge from self.edge
        if old_edge.ident in self.edges:
            old_edge.left._del_edge(old_edge)
            old_edge.right._del_edge(old_edge)
            del self.edges[old_edge.ident]

    def _check_match(self, alleged_match):
        tn = []
        ismatch = True
        if alleged_match is not None and len(alleged_match) > 0:
            for e in alleged_match:
                if e.left in tn or e.right in tn:
                    ismatch = False
                else:
                    tn.append(e.left)
                    tn.append(e.right)
        return ismatch

    def _disjoint(self, set_one, set_two):
        # Take two edge sets as input and return as output a bit indicating whether they are vertex-disjoint
        set_one_nodes = [e.left for e in set_one] + [e.right for e in set_one]
        set_two_nodes = [e.left for e in set_two] + [e.right for e in set_two]
        # Note if vertex-disjoint, then set_one_nodes intersected with set_two_nodes is the empty set, so the sym_dif
        # is a disjoint union!
        return len(set_one_nodes)+len(set_two_nodes)==len(sym_dif(set_one_nodes, set_two_nodes))

    def _bfs(self, match=None):
        #print("initiating bfs")
        if match is None:
            match = []
        result = []

        matched_lefts = []
        matched_rights = []
        for e in match:
            assert e.ident in self.edges
            if e.left not in matched_lefts:
                matched_lefts.append(e.left)
            if e.right not in matched_rights:
                matched_rights.append(e.right)
        unmatched_lefts = [self.left[n] for n in self.left if self.left[n] not in matched_lefts]
        unmatched_rights = [self.right[n] for n in self.right if self.right[n] not in matched_rights]
        assert len(unmatched_lefts) + len(matched_lefts) == len(self.left)
        assert len(unmatched_rights) + len(matched_rights) == len(self.right)

        q = deque()

        #print("constructing first potential paths")
        potential_paths = []
        tn = [] # for tracking unmatched
        for n in unmatched_lefts:
            for e in n.edges:
                assert e not in match
                assert e.left == n
                if [e] not in potential_paths and e.right not in tn:
                    potential_paths += [[e]]
        for p in potential_paths:
            for e in p:
                if e.left not in tn:
                    tn.append(e.left)
                if e.right not in tn:
                    tn.append(e.right)
        q.append(potential_paths)

        #print("potential paths = " + str(potential_paths))
        #print("queue = " + str(q))

        spf = False # shortest path found yet?

        #print("entering main loop")
        while(len(q) > 0 and not spf):
            potential_paths = q.popleft()
            #print("next entry in queue: " + str(potential_paths))
            next_paths = []
            for p in potential_paths:
                parity = len(p)%2
                if parity == 0:
                    endpt = p[-1].left
                    for e in endpt.edges:
                        if e not in match and e.right not in tn:
                            pp = [f for f in p] + [e]
                            next_paths += [pp]
                elif parity == 1:
                    endpt = p[-1].right
                    if endpt in unmatched_rights:
                        spf = True
                        result += [p]
                    else:
                        for e in endpt.edges:
                            if e in match and e.left not in tn:
                                pp = [f for f in p] + [e]
                                next_paths += [pp]
                else:
                    print("lolwut")
            if len(next_paths)>0:
                for p in next_paths:
                    for e in p:
                        if e.left not in tn:
                            tn.append(e.left)
                        if e.right not in tn:
                            tn.append(e.right)
                q.append(next_paths)

        # check alternating and augmenting
        for p in result:
            assert len(p) > 0
            idx = 0
            while(idx < len(p)):
                parity = idx%2
                if parity == 0:
                    assert p[idx] not in match
                    assert p[idx].ident in self.edges
                else:
                    assert p[idx] in match
                    assert p[idx].ident in self.edges
                idx += 1
        return result

    def _get_augmenting_paths(self, match=None, wt=True, verbose=False):
        ### returns VERTEX-DISJOINT augmenting paths
        vertex_disjoint_choices = []
        if verbose:
            line = [e.ident for e in match]
            print("Beginning _get_augmenting_paths with match = " + str(line))

            shortest_paths = self._bfs(match)

            line = [[e.ident for e in p] for p in shortest_paths]
            print("found shortest paths = " + str(line))
            print("verifying correct form")
            for p in shortest_paths:
               assert len(p) % 2 == 1
               line = [e.ident for e in p]
               print("checking alt and aug for p = " + str(line))
               #print(self.edges)
               for i in range(len(p)):
                   parity = i % 2
                   if parity == 0:
                       assert p[i] not in match
                       assert p[i].ident in self.edges
                   else:
                       assert p[i] in match
                       assert p[i].ident in self.edges

            print("Putting paths into an order either by weight or randomly.")
            sorted_paths = None
            if wt:
                weighted_paths = []
                for p in shortest_paths:
                    s = 0.0
                    for e in p:
                        s += e.weight
                    weighted_paths += [[s,p]]
                weighted_sorted_paths = sorted(weighted_paths, key=lambda x:x[0], reverse=True)
                sorted_paths = [x[1] for x in weighted_sorted_paths]
            else:
                x = [i for i in range(len(shortest_paths))]
                random.shuffle(x)
                sorted_paths = [shortest_paths[i] for i in x]


            line = [[e.ident for e in p] for p in sorted_paths]
            print("found sorted paths = " + str(line))
            print("verifying correct form")
            for p in sorted_paths:
                assert len(p) % 2 == 1
                line = [e.ident for e in p]
                print("checking alt and aug for p = " + str(line))
                # print(self.edges)
                for i in range(len(p)):
                    parity = i % 2
                    if parity == 0:
                        assert p[i] not in match
                        assert p[i].ident in self.edges
                    else:
                        assert p[i] in match
                        assert p[i].ident in self.edges

            print("Making vertex-disjoint according to above order")
            for p in sorted_paths:
                d = True
                for pp in vertex_disjoint_choices:
                    d = d and self._disjoint(p,pp)
                if d:
                    vertex_disjoint_choices += [p]

            line = [[e.ident for e in p] for p in vertex_disjoint_choices]
            print("found vdc = " + str(line))
            print("verifying correct form")
            for p in vertex_disjoint_choices:
                assert len(p) % 2 == 1
                line = [e.ident for e in p]
                print("checking alt and aug for p = " + str(line))
                # print(self.edges)
                for i in range(len(p)):
                    parity = i % 2
                    if parity == 0:
                        assert p[i] not in match
                        assert p[i].ident in self.edges
                    else:
                        assert p[i] in match
                        assert p[i].ident in self.edges
        else:
            shortest_paths = self._bfs(match)
            sorted_paths = None
            if wt:
                weighted_paths = []
                for p in shortest_paths:
                    s = 0.0
                    for e in p:
                        s += e.weight
                    weighted_paths += [[s, p]]
                weighted_sorted_paths = sorted(weighted_paths, key=lambda x: x[0], reverse=True)
                sorted_paths = [x[1] for x in weighted_sorted_paths]
            else:
                x = [i for i in range(len(shortest_paths))]
                random.shuffle(x)
                sorted_paths = [shortest_paths[i] for i in x]
            for p in sorted_paths:
                d = True
                for pp in vertex_disjoint_choices:
                    d = d and self._disjoint(p, pp)
                if d:
                    vertex_disjoint_choices += [p]
        return vertex_disjoint_choices

    def _get_improving_cycles(self, match):
        weighted_edges = []
        for e in self.edges:
            if e not in match:
                this_edge = self.edges[e]
                assert isinstance(this_edge, Edge)
                weighted_edges.append((this_edge.weight, self.edges[e]))
        sorted_weighted_unmatched_edges = sorted(weighted_edges, key=lambda x:x[0], reverse=True)
        swue = [x[1] for x in sorted_weighted_unmatched_edges]
        fourcycles = []
        sixcycles = []
        tn = []
        for e in swue:
            le = e.left
            re = e.right
            if le not in tn and re not in tn:
                for f in le.edges:
                    if e != f:
                        lf = f.left
                        assert lf == e.left
                        rf = f.right
                        if rf not in tn and f in match:
                            for g in re.edges:
                                if g != e and g != f:
                                    lg = g.left
                                    rg = g.right
                                    assert rg == e.right
                                    if lg not in tn and g in match:
                                        missing_ident = lg.ident + "," + rf.ident
                                        if missing_ident in self.edges:
                                            h = self.edges[missing_ident]
                                            if h != e and h != g and h != f:
                                                assert h not in match
                                                assert h.left == g.left
                                                assert h.right == f.right
                                                assert f.left == e.left
                                                assert g.right == e.right
                                                gain = e.weight + h.weight - f.weight - g.weight
                                                if gain > 0:
                                                    newcycle = [gain, [e, f, h, g]]
                                                    fourcycles.append(newcycle)
                                                    tn += [e.left, e.right, g.left, f.right]
                                        else:
                                            for fp in lg.edges:
                                                if fp != e and fp != f and fp != g:
                                                    lfp = fp.left
                                                    assert lfp == lg
                                                    rfp = fp.right
                                                    if rfp not in tn and fp not in match:
                                                        for gp in rf.edges:
                                                            if gp != fp and gp != e and gp != f and gp != g:
                                                                lgp = gp.left
                                                                rgp = gp.right
                                                                assert rgp == rf
                                                                if lgp not in tn and gp not in match and lgp != e.left and rgp != e.right:
                                                                    missing_ident = lgp.ident + "," + rfp.ident
                                                                    if missing_ident in self.edges:
                                                                        h = self.edges[missing_ident]
                                                                        if h != gp and h != fp and h != e and h != f and h != g:
                                                                            assert h in match
                                                                            assert h.left == gp.left
                                                                            assert h.right == fp.right
                                                                            assert gp.right == f.right
                                                                            assert fp.left == g.left
                                                                            assert f.left == e.left
                                                                            assert g.right == e.right
                                                                            gain = e.weight + fp.weight + gp.weight - f.weight - g.weight - h.weight
                                                                            if gain > 0:
                                                                                newcycle = [gain, [e, f, gp, h, fp, g]]
                                                                                sixcycles.append(newcycle)
                                                                                tn += [e.left, e.right, g.left, fp.right, gp.left, f.right]
        return [fourcycles, sixcycles]

    def _get_bigger_matching(self, match=None, wt=True):
        #print("Beginning _get_bigger_matching")
        vdc = self._get_augmenting_paths(match, wt)
        if match is None:
            match = []
        for p in vdc:
            #line1 = [e.ident for e in p]
            #line2 = [e.ident for e in match]
            #print("sym-diffing p = " + str(line1) + " with match = " + str(line2))
            match = sym_dif(match, p)
            #line = [e.ident for e in match]
            #print("result = " + str(line))
        return match

    def max_matching(self, match=None, wt=True):
        #line = [e.ident for e in match]
        #print("\n\nBeginning max_matching with match = " + str(line))
        next_match = self._get_bigger_matching(match, wt)
        #line = [e.ident for e in next_match]
        #print("First iteration gets us " + str(line))
        while(match is None or len(next_match)>len(match)):
            match = next_match
            next_match = self._get_bigger_matching(match, wt)
            #line = [e.ident for e in next_match]
            #print("Next iteration gets us " + str(line))
        return match

    def opt_matching(self, level_of_cycles=10):
        ct = 0
        result = self.max_matching(match=None, wt=True)
        #print("Result = " + str(result))
        wt = 0.0
        for e in result:
            wt += e.weight
        cycles = self._get_improving_cycles(result)
        fourcycles = cycles[0]
        sixcycles = cycles[1]
        cycle_list = [c[1] for c in fourcycles] + [c[1] for c in sixcycles]
        ct += 1
        newResult = result
        if len(cycle_list)>0:
            #print("found a cycle, augmenting")
            for c in cycle_list:
                newResult = sym_dif(newResult, c)
                #print("New result = " + str(newResult))
        newWt = 0.0
        for e in newResult:
            newWt += e.weight
        while(newWt > wt and ct < level_of_cycles):
            wt = newWt
            result = newResult

            cycles = self._get_improving_cycles(result)
            fourcycles = cycles[0]
            sixcycles = cycles[1]
            cycle_list = [c[1] for c in fourcycles] + [c[1] for c in sixcycles]

            ct += 1
            newResult = result
            if len(cycle_list) > 0:
                #print("found a cycle, augmenting")
                for c in cycle_list:
                    #print("Checking cycle = " + str(c))
                    #print("Current match = " + str(newResult))
                    newResult = sym_dif(newResult, c)
                    #print("newResult = " + str(newResult))
            newWt = 0.0
            for e in newResult:
                newWt += e.weight
        return newResult

def make_graph(i,r,wt=None):
    # Create a bipartite graph with 2*i nodes such that all right-hand nodes are r-regular.
    s = random.random()
    x = str(hash(str(1.0) + str(s)))
    par = {'data': 1.0, 'ident': x, 'left': [], 'right':[], 'edges': []}
    G = BipartiteGraph(par)
    N = i  # nodeset size
    K = r  # neighbor size
    ct = 0
    while len(G.left) <  N:
        while str(ct) in G.left:
            ct += 1
        par = {'data': 1.0, 'ident': str(ct)}
        n = Node(par)
        G._add_left(n)
        ct += 1
    while len(G.right) <  N:
        while str(ct) in G.right:
            ct += 1
        par = {'data': 1.0, 'ident': str(ct)}
        n = Node(par)
        G._add_right(n)
        ct += 1

    leftkeys = list(G.left.keys())
    rightkeys = list(G.right.keys())
    for i in range(N):
        sig_idx = rightkeys[i]
        right_node = G.right[sig_idx]
        assert isinstance(right_node, Node)
        idxs = random.sample(range(N), K)
        assert len(idxs) == K
        for j in idxs:
            otk_idx = leftkeys[j]
            left_node = G.left[otk_idx]
            assert isinstance(left_node, Node)
            x = left_node.ident + "," + right_node.ident
            par = {'data': 1.0, 'ident': x, 'left':left_node, 'right':right_node, 'weight': 0}
            e = Edge(par)
            G._add_edge(e)

    if wt is not None:
        for eid in G.edges:
            e = G.edges[eid]
            e.weight = random.random()

    return G