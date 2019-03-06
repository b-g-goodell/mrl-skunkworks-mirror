from collections import deque
import random
import sys

if sys.version_info[0] != 3:
    print("This script requires Python3")
    sys.exit(1)

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

    def _report(self):
        # Return a string summary of the graph. Spans multiple lines.
        line = ""
        line += "\n\t\t\t=====Reporting on Graph " + str(self.ident) + " ====="
        line += "\nLeft nodes:"
        for l in self.left:
            line += "\n\t" + str(l) + ",\t\t\t" + str(self.left[l])
        line += "\nRight nodes:"
        for r in self.right:
            line += "\n\t" + str(r) + ",\t\t\t" + str(self.right[r])
        line += "\nEdges:"
        for e in self.edges:
            line += "\n\t" + str(e) + ",\t\t\t" + str(self.edges[e]) +",\t\t\twt = " + str(self.edges[e].weight)
        return line

    def _add_left(self, new_node):
        # Add a new_node to self.left
        if new_node.ident not in self.left:
            self.left.update({new_node.ident:new_node})

    def _add_right(self, new_node):
        # Add a new_node to self.right
        if new_node.ident not in self.right:
            self.right.update({new_node.ident:new_node})

    def _del_node(self, old_node):
        # Delete an old_node from self.left or self.right
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
        if new_edge not in new_edge.left.edges:
            new_edge.left._add_edge(new_edge)
        if new_edge not in new_edge.right.edges:
            new_edge.right._add_edge(new_edge)
        if new_edge.left.ident not in self.left:
            self._add_left(new_edge.left)
        if new_edge.right.ident not in self.right:
            self._add_right(new_edge.right)
        if new_edge.ident not in self.edges:
            self.edges.update({new_edge.ident:new_edge})

    def _del_edge(self, old_edge):
        # Remove an old_edge from self.edges
        if old_edge.ident in self.edges:
            old_edge.left._del_edge(old_edge)
            old_edge.right._del_edge(old_edge)
            del self.edges[old_edge.ident]

    def _check_match(self, alleged_match):
        # Return boolean indicating whether alleged_match is truly a match
        # The constraint here is that each node adjacent to any edge in the
        # match is adjacent to only one edge in the match.
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
        # Return boolean indicating whether set_one and set_two are vertex-disjoint lists of edges
        set_one_nodes = [e.left for e in set_one] + [e.right for e in set_one]
        set_two_nodes = [e.left for e in set_two] + [e.right for e in set_two]
        # Note if vertex-disjoint, then set_one_nodes intersected with set_two_nodes is the empty set, so the sym_dif
        # is a disjoint union!
        return len(set_one_nodes)+len(set_two_nodes)==len(sym_dif(set_one_nodes, set_two_nodes))

    def _bfs(self, match=None):
        # Carry out a breadth-first search with respect to input match.
        # Seeking a vertex-disjoint subset of the set of all shortest paths whose initial endpoint is an unmatched
        # left endpoint and whose terminal endpoint is an unmatched right endpoint.
        if match is None:
            match = []
        result = []

        # Assemble node sets for touchin'
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

        q = deque()

        #print("constructing first potential paths")
        potential_paths = []
        # Filling potential_paths with initial possible paths.
        for n in unmatched_lefts:
            for e in n.edges:
                assert e not in match
                assert e.left == n
                if [e] not in potential_paths:
                    potential_paths += [[e]]
        q.append(potential_paths)

        # Entering main loop.
        tn = [] # for touching nodes
        spf = False  # shortest path found yet?
        while(len(q) > 0 and not spf):
            # Take current list of potential paths
            potential_paths = q.popleft()

            # Start next level of potential paths
            next_paths = []
            for p in potential_paths:
                parity = len(p)%2
                if parity == 0:
                    # In this case, the path has an even number of edges and needs at least one more to find
                    # an unmatched right node (it started on the left)
                    endpt = p[-1].left
                    for e in endpt.edges:
                        if e not in match and e not in p and e.right not in tn:
                            pp = [f for f in p] + [e]
                            next_paths += [pp]
                elif parity == 1:
                    endpt = p[-1].right
                    if endpt in unmatched_rights:
                        # In this case, the path is a success! We add to our result and touch nodes on the path.
                        spf = True
                        result += [p]
                        for e in p:
                            if e.left not in tn:
                                tn.append(e.left)
                            if e.right not in tn:
                                tn.append(e.right)
                    else:
                        # In this case, the path is not a success but has an odd number of edges. We add a matched
                        # edge to get to the left.
                        for e in endpt.edges:
                            if e in match and e not in p and e.left not in tn:
                                pp = [f for f in p] + [e]
                                next_paths += [pp]
                else:
                    # This should never occur
                    print("lolwut")
            if len(next_paths)>0 and not spf:
                # if any potential paths are in next_paths, they are added to the queue
                # Of course, if spf = True, then we won't be going into that loop anyway.
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

    def _get_improving_cycles(self, match, cutoff=6):
        # breadth-first search for alternating cycles inside matching
        # stop if cycle length > 6 (or maybe 8?)
        # Only works with weighted graphs

        # We first find all unmatched edges and sort them by weight
        weighted_edges = []
        for e in self.edges:
            if self.edges[e] not in match:
                this_edge = self.edges[e]
                assert isinstance(this_edge, Edge)
                weighted_edges.append((this_edge.weight, self.edges[e]))

        sorted_weighted_unmatched_edges = sorted(weighted_edges, key=lambda x:x[0], reverse=True)
        swue = [x[1] for x in sorted_weighted_unmatched_edges]

        # Each such edge may be the start of a potential_cycle
        potential_cycles = []
        q = deque()
        tn = []
        result = []
        for e in swue:
            if [e] not in potential_cycles:
                if e.left not in tn and e.right not in tn:
                    potential_cycles += [[e]]
        q.append(potential_cycles)


        # Beginning main loop
        current_path_length = 1
        scpgf = False # Shortest cycle with positive gain found? I.e. should we stop the while loop?
        while(len(q) > 0 and not scpgf and current_path_length < cutoff):

            # Take current potential cycles, find their length, start next_cycles list
            these_cycles = q.popleft()
            current_path_length = len(these_cycles[0])
            next_cycles = []

            parity = current_path_length % 2
            for c in these_cycles:
                # We will not keep adding onto this cycle if it has any touched nodes.
                any_touched_nodes = False
                for e in c:
                    any_touched_nodes = any_touched_nodes or e.left in tn or e.right in tn
                if not any_touched_nodes:
                    # Target vertex that completes the cycle
                    target_vertex = c[0].left
                    if parity == 1:
                        # We began with an unmatched edge, no right node can be the target! This cycle needs
                        # at least one more (matched) edge to become an alternating cycle
                        current_endpt = c[-1].right
                        for e in current_endpt.edges:
                            if not scpgf and e not in c and e in match:
                                cp = c + [e]
                                if cp not in next_cycles:
                                    next_cycles += [cp]
                    elif parity == 0:
                        current_endpt = c[-1].left
                        if target_vertex.ident == current_endpt.ident:
                            # In this case, the cycle has been completed.
                            # If it has positive gain, we add it to our results and touch its nodes.
                            gain = 0.0
                            length_of_cycle = len(c)
                            num_idxs = length_of_cycle//2
                            for i in range(num_idxs):
                                gain += c[2*i].weight - c[2*i+1].weight
                            if gain > 0.0:
                                scpgf = True
                                result += [c]
                                for e in c:
                                    if e.left not in tn:
                                        tn.append(e.left)
                                    if e.right not in tn:
                                        tn.append(e.right)
                        elif not scpgf:
                            # in this case, we have an even length path that is not a cycle, beginning with an unmatched
                            # edge. So we need to add at least one more unmatched edge to find a cycle
                            for e in current_endpt.edges:
                                if e not in c and e not in match:
                                    cp = c + [e]
                                    if cp not in next_cycles:
                                        next_cycles += [cp]
                    else:
                        print("lolwut")
            if len(next_cycles) > 0 and not scpgf:
                q.append(next_cycles)

        # check each cycle is alternating and augmenting
        for c in result:
            assert len(c) > 0
            idx = 0
            while(idx < len(c)):
                parity = idx%2
                if parity == 0:
                    assert c[idx] not in match
                    assert c[idx].ident in self.edges
                else:
                    assert c[idx] in match
                    assert c[idx].ident in self.edges
                idx += 1
        return result

    def _get_augmenting_paths(self, match=None, wt=True):
        # Returns VERTEX-DISJOINT augmenting paths
        vertex_disjoint_choices = []

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
            if len(vertex_disjoint_choices) > 0:
                for pp in vertex_disjoint_choices:
                    d = d and self._disjoint(p, pp)
            if d:
                vertex_disjoint_choices += [p]
        return vertex_disjoint_choices

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
        ct += 1
        newResult = result
        if len(cycles)>0:
            #print("found a cycle, augmenting")
            for c in cycles:
                newResult = sym_dif(newResult, c)
                #print("New result = " + str(newResult))
        newWt = 0.0
        for e in newResult:
            newWt += e.weight
        while(newWt > wt and ct < level_of_cycles):
            wt = newWt
            result = newResult

            cycles = self._get_improving_cycles(result)
            ct += 1
            newResult = result
            if len(cycles) > 0:
                #print("found a cycle, augmenting")
                for c in cycles:
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
        if wt=="random":
            for eid in G.edges:
                e = G.edges[eid]
                e.weight = random.random()
        else:
            # Assume wt is dictionary of the form {edge.ident:edge.weight}
            for eid in wt:
                assert eid in G.edges
                G.edges[eid].weight = wt[eid]
    return G