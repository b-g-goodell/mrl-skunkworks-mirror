from collections import deque
from copy import deepcopy
import random
import sys

# Check version number
if sys.version_info[0] != 3:
    print("This script requires Python3")
    sys.exit(1)

def disjoint(set_one, set_two):
    """ Return boolean indicating whether set_one and set_two are vertex-disjoint lists.

    Keyword arguments:
        set_one -- first set
        set_two -- second set
    """
    return len(set_one) + len(set_two) == len(symdif(set_one, set_two))

def symdif(x=None, y=None):
    """ Compute symmetric difference of two lists, return a list.

    Keyword arguments:
        x -- first list (default [])
        y -- second list (default [])
    """
    if x is None:
        x = []
    if y is None:
        y = []
    return [e for e in x if e not in y] + [e for e in y if e not in x]

class BipartiteGraph(object):
    """ Graph object representing a graph

    Attributes:
        data : arbitrary
        ident : string
        left_nodes : dict, keys = node_idents, vals = node_idents
        right_nodes : dict, keys = node_idents, vals = node_idents
        red_edge_weights : dict, keys = (node_ident, node_ident) tuples (pairs), vals = weights
        blue_edge_weights : dict, keys = (node_ident, node_ident) tuples (pairs), vals = weights

    Functions:
        add_left_node : take node_ident as input and add to self.left_nodes
        add_right_node : take node_ident as input and add to self.right_nodes
        del_node : take node_ident as input, deletes related edges with del_edge, 
            removes input from self.right_nodes or self.left_nodes
        add_red_edge : take tuple (pair) of node_idents and a weight (int or float) 
            as input. Add to self.red_edge_weights, adding endpoints to self.left_nodes 
            or self.right_nodes if necessary
        add_blue_edge : take tuple (pair) of node_idents and a weight (int or float) 
            as input. Add to self.red_edge_weights, adding endpoints to self.left_nodes 
            or self.right_nodes if necessary
        del_edge : take tuple (pair) of node_idents and sets weight to str("-infty")
        get_augmenting_path_of_reds : take a match of red edges as input and return an 
            augmenting path of red edges as output.
        get_bigger_red_matching : take match of red edges as input and return a 
            larger match of red edges as ouptut
        maximal_matching : iteratively call get_bigger_red_matching until matches of same 
            size are returned.
        optimize_max_matching : find a highest-weight matching.
    """

    def __init__(self, par):
        # par = {'data':data, 'ident':ident, 'left':list of Nodes, 'right':list of Node, 'edges':list of Edge}
        self.data = par['data']   # str
        self.ident = par['ident']  # str
        self.left_nodes = {}
        self.left_nodes.update(par['left'])
        self.right_nodes = {}
        self.right_nodes.update(par['right'])
        self.red_edge_weights = {}
        self.red_edge_weights.update(par['red_edges'])
        self.blue_edge_weights = {}
        self.blue_edge_weights.update(par['blue_edges'])

    def add_left_node(self, new_nid):
        # Add a new_node to self.left
        rejected = False
        # print(self.left_nodes, type(self.left_nodes))
        rejected = rejected or new_nid in self.left_nodes
        rejected = rejected or new_nid in self.right_nodes
        if not rejected:
            self.left_nodes.update({new_nid:new_nid})
        return rejected

    def add_right_node(self, new_nid):
        # Add a new_node to self.left
        rejected = False
        rejected = rejected or new_nid in self.right_nodes
        rejected = rejected or new_nid in self.left_nodes
        if not rejected:
            self.right_nodes.update({new_nid:new_nid})
        return rejected

    def del_node(self, old_nid):
        # Delete an old_node from self.left or self.right
        rejected = old_nid not in self.left_nodes and old_nid not in self.right_nodes
        if not rejected:
            to_be_deleted = []
            for eid in self.red_edge_weights:
                if old_nid == eid[0] or old_nid == eid[1]:
                    to_be_deleted.append(eid)
            new_red_edges = {}
            for eid in self.red_edge_weights:
                if eid not in to_be_deleted:
                    new_red_edges[eid] = self.red_edge_weights[eid]
            self.red_edge_weights = new_red_edges

            to_be_deleted = []
            for eid in self.blue_edge_weights:
                if old_nid == eid[0] or old_nid == eid[1]:
                    to_be_deleted.append(eid)
            new_blue_edges = {}
            for eid in self.blue_edge_weights:
                if eid not in to_be_deleted:
                    new_blue_edges[eid] = self.blue_edge_weights[eid]
            self.blue_edge_weights = new_blue_edges

            if old_nid in self.left_nodes:  
                del self.left_nodes[old_nid]
            if old_nid in self.right_nodes:  
                del self.right_nodes[old_nid]
        return rejected

    def add_red_edge(self, eid, new_edge_weight):
        ''' Throw a new red edge with edge identity eid into the graph. '''
        rejected = (len(eid)!=2)
        assert not rejected
        rejected = rejected or eid in self.red_edge_weights 
        assert not rejected
        rejected = rejected or eid in self.blue_edge_weights 
        assert not rejected
        rejected = rejected or eid[0] not in self.left_nodes 
        assert not rejected
        rejected = rejected or eid[1] not in self.right_nodes
        assert not rejected
        if not rejected:
            self.red_edge_weights.update({eid:new_edge_weight})
        return rejected

    def add_blue_edge(self, eid, new_edge_weight):
        ''' Throw a new blue edge with edge identity eid into the graph. '''
        rejected = (len(eid)!=2)
        assert not rejected
        rejected = rejected or eid in self.red_edge_weights 
        assert not rejected
        rejected = rejected or eid in self.blue_edge_weights 
        assert not rejected
        rejected = rejected or eid[0] not in self.left_nodes 
        assert not rejected
        rejected = rejected or eid[1] not in self.right_nodes
        assert not rejected
        if not rejected:
            self.blue_edge_weights.update({eid:new_edge_weight})
        return rejected

    def del_edge(self, old_eid):
        # Remove an old_edge from self.edges
        rejected = old_eid not in self.red_edge_weights and old_eid not in self.blue_edge_weights
        if not rejected:
            if not rejected and old_eid in self.red_edge_weights:
                del self.red_edge_weights[old_eid]
            if not rejected and old_eid in self.blue_edge_weights:
                del self.blue_edge_weights[old_eid]
        return rejected

    def _check_red_match(self, alleged_match):
        # Return boolean indicating whether alleged_match is truly a match
        # from the red-edges.
        # The constraints:
        #  1) Each node adjacent to any edge in the match is adjacent to only one edge in the match.
        #  2) All edges in the match are red_edges
        tn = []
        ismatch = True
        if alleged_match is not None and len(alleged_match) > 0:
            for eid in alleged_match:
                if eid not in self.red_edge_weights or eid[0] in tn or eid[1] in tn:
                    ismatch = False
                else:
                    tn.append(eid[0])
                    tn.append(eid[1])
        return ismatch

    def _check_blue_match(self, alleged_match):
        # Return boolean indicating whether alleged_match is truly a match
        # from the blue-edges.
        # The constraints:
        #  1) Each node adjacent to any edge in the match is adjacent to only one edge in the match.
        #  2) All edges in the match are blue_edges
        tn = []
        ismatch = True
        if alleged_match is not None and len(alleged_match) > 0:
            for eid in alleged_match:
                if eid not in self.blue_edge_weights or eid[0] in tn or eid[1] in tn:
                    ismatch = False
                else:
                    tn.append(eid[0])
                    tn.append(eid[1])
        return ismatch

    def redd_bfs(self, match=[]):
        ''' Find all augmenting paths with respect to non-empty input match using breadth-first search. '''
        # TODO: Rename to reflect search from unmatched lefts to unmatched rights.
        assert len(match) > 0
        result = None # Receiving None as output means input was not a matching on the nodes.
        q = deque()
        found_shortest_path = False

        if self._check_red_match(match):
            # At least pass back an empty list if match is a matching
            result = []

            # Assemble a few node lists for convenience
            matched_lefts = [eid[0] for eid in match]
            matched_rights = [eid[1] for eid in match]
            unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
            unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]

            if len(unmatched_rights) == 0 or len(unmatched_lefts) == 0:
                found_shortest_path = True
                #result = [[eid] for eid in match]
            else:
                # prepare for main loop
                for eid in self.red_edge_weights:
                    if eid[0] in unmatched_lefts:
                        if eid[1] in unmatched_rights:
                            found_shortest_path = True
                            result.append([eid])
                        else:
                            q.append([eid])
                # some pre-loop checks
                assert len(result) > 0 or len(q) > 0
                assert not found_shortest_path or len(result) > 0  # Check if len(result) > 0 then found_shortest_path

                # entering main loop
                if not found_shortest_path:
                    while len(q) > 0:
                        # get next path in queue
                        this_path = q.popleft()
                        next_path = None

                        # extract some info about path
                        parity = len(this_path) % 2  # parity of path
                        last_edge = this_path[-1]  # last edge in path
                        last_node = last_edge[parity]  # last node in path
                        tn = [eid[0] for eid in this_path] + [eid[1] for eid in this_path]  # touched nodes
                        tn = list(dict.fromkeys(tn))  # deduplicate

                        if not parity:
                            edge_set = [x for x in self.red_edge_weights if x not in match if x not in this_path]
                        elif parity:
                            edge_set = match
                        for eid in edge_set:
                            if eid[1-parity] not in tn and eid not in this_path and eid[parity] == last_node:
                                # if parity == 1, add nonpath edge from match adj to the last edge with untchd endpoint
                                # if parity == 0, add nonpath red edge adj to the last edge with untchd endpoint
                                assert (this_path[-1] in match and eid not in match) or (this_path[-1] not in match and eid in match)
                                next_path = this_path + [eid]
                                # next_path.append(eid)
                                extensible = True
                                if parity and eid in match and not found_shortest_path:
                                    q.append(next_path)
                                elif not parity and eid not in match:
                                    if eid[1] in unmatched_rights:
                                        found_shortest_path = True
                                        result.append(next_path)
                                    else:
                                        q.append(next_path)
                        if not extensible:
                            print("Error: We found a path that cannot be extended to a path that terminates at an " +
                                    "unmatched right node. Path found = ", this_path)
            # print("lasting result 1 = ", result)
            # print("lasting result 2 = ", result)
            # Before returning our result, we check each path in the result is alternating
            # with respect to the input match and all have the same length.
            assert isinstance(result, list)  # check result is a list
            if len(result) > 0:
                # If any paths are found, verify they are augmenting
                # Check: if found_shortest_path then result is not None and len(result) > 0
                assert not found_shortest_path or (result is not None and len(result) > 0)
                # Check: if result is not None and len(result) > 0 then found_shortest_path
                assert not (result is not None and len(result) > 0) or found_shortest_path
                if found_shortest_path:
                    path_length = len(result[0])
                    for p in result:
                        assert len(p) == path_length  # check all shortest paths have the same length
                        for i in range(path_length):
                            parity = i % 2  # 0th edge is not in match, 1st edge is in match, 2nd edge is not, alternating
                            if not parity:
                                assert p[i] not in match
                            elif parity:
                                assert p[i] in match
        return result

    def get_augmenting_red_paths(self, match=[]):
        # Returns list of vertex disjoint augmenting paths of red-edges
        result = [] # vertex-disjoint choices
        shortest_paths = self.redd_bfs(match) # These are augmenting by construction
        if shortest_paths is not None and len(shortest_paths) > 0:
            # sort paths by weight
            sorted_paths = None
            weighted_paths = []
            for p in shortest_paths:
                s = 0.0
                for eid in p:
                    s += self.red_edge_weights[eid]
                weighted_paths += [[s, p]]
            weighted_sorted_paths = sorted(weighted_paths, key=lambda x: x[0], reverse=True)
            sorted_paths = [x[1] for x in weighted_sorted_paths]

            # Add each path to vertex_disjoint_choices if they are vertex disjoint!
            if sorted_paths is not None and len(sorted_paths) > 0:
                for p in sorted_paths:        
                    vertices_in_p = [eid[0] for eid in p] + [eid[1] for eid in p]
                    d = True
                    if len(result) > 0:
                        for pp in result:
                            # print(pp, len(pp))
                            vertices_in_pp = [eid[0] for eid in pp] + [eid[1] for eid in pp]
                            d = d and disjoint(vertices_in_p, vertices_in_pp)
                    if d:
                        result += [p]
        return result

    def get_bigger_red_matching(self, match=[]):
        ''' Call get_augmenting_red_paths and XOR/symdif with curren'''
        # print("Beginning _get_bigger_red_matching")
        result = None
        if self._check_red_match(match):
            result = match
            vdc = self.get_augmenting_red_paths(match)
            if len(vdc) > 0:
                for p in vdc:
                    result = symdif(result, p)
        return result

    def get_max_red_matching(self, match=[]):
        ''' Call get_bigger_red_matching until results stop getting bigger. '''
        assert len(match) > 0
        next_match = self.get_bigger_red_matching(match)
        while len(next_match) > len(match):
            match = next_match
            next_match = self.get_bigger_red_matching(match)
        return match

    def get_shortest_improving_paths_wrt_max_match(self, match=[]):
        ''' Find all paths with respect to non-empty and maximal input match using breadth-first search. We call such a
        path improving if it is half-augmenting in the sense that the path must alternate and begin with a nonmatch
        edge but can terminate with a match edge. '''
        # TODO: Rename to reflect search starts with unmatched lefts and terminates upon self-intersection
        assert len(match) > 0
        result = None # Receiving None as output means input was not a matching on the nodes.
        q = deque()
        found_shortest_path = False
        critical_path_length = None

        if self._check_red_match(match):
            # At least pass back an empty list if match is a matching
            result = []

            # Assemble a few node lists for convenience
            matched_lefts = [eid[0] for eid in match]
            matched_rights = [eid[1] for eid in match]
            unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
            unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]

            # Check for maximality
            assert len(unmatched_rights)==0 or len(unmatched_lefts)==0
            # Note that for a monero or zcash-style application, len(unmatched_lefts) is never zero (in practice)
            # and in these cases the goal is to match all right nodes (not necessarily all left nodes).

            # prepare for main loop
            for eid in self.red_edge_weights:
                if eid[0] in unmatched_lefts:
                    if eid[1] in unmatched_rights:
                        found_shortest_path = True
                        old_len = len(result)
                        result.append([eid])
                        new_len = len(result)
                        assert new_len - old_len > 0
                        assert len(result) > 0
                    else:
                        q.append([eid])
            # some pre-loop checks
            assert len(result) > 0 or len(q) > 0
            assert not found_shortest_path or len(result) > 0  # Check that len(result) > 0 implies found_shortest_path = True

            # entering main loop
            if not found_shortest_path:
                while len(q) > 0:
                    # get next path in queue
                    this_path, next_path = q.popleft(), None
                    if not found_shortest_path or (found_shortest_path and len(this_path) <= critical_path_length):
                        # extract some info about path
                        parity = len(this_path) % 2  # parity of path
                        last_edge = this_path[-1]  # last edge in path
                        last_node = last_edge[parity]  # last node in path
                        tn = [eid[0] for eid in this_path] + [eid[1] for eid in this_path]  # touched nodes
                        tn = list(dict.fromkeys(tn))  # deduplicate

                        path_is_done = False
                        edge_set = None

                        if not parity:
                            # Only need to check touched nodes since matching is already maximal, there are no unmatched
                            # rightnodes (since a maximal match covers all rightnodes)
                            edge_set = [x for x in self.red_edge_weights if x not in match and x not in this_path and x[1-parity] not in tn and x[parity]==last_node]
                        elif parity:
                            edge_set = [x for x in match if x not in this_path and x[1-parity] not in tn and x[parity]==last_node]
                        path_is_done = (edge_set is None or len(edge_set) == 0)
                        if path_is_done:
                            gain = sum([self.red_edge_weights[eid] for eid in this_path if eid not in match])
                            gain = gain - sum([self.red_edge_weights[eid] for eid in this_path if eid in match])
                            if gain > 0:
                                found_shortest_path = True
                                if critical_path_length is None:
                                    critical_path_length = len(this_path)
                                else:
                                    assert critical_path_length <= len(this_path)
                                    assert found_shortest_path
                                result.append((this_path, gain))
                        else:
                            for eid in edge_set:
                                q.append(this_path + [eid])
            assert isinstance(result, list)  # check result is a list
            if len(result) == 0:
                # if no paths stored in result, then input match is optimal.
                result = [([eid], 0.0) for eid in match]
            else:
                for pathGainPair in result:
                    p = pathGainPair[0]
                    g = pathGainPair[1]
                    assert g > 0
                    for eid in p:
                        i = p.index(eid)
                        parity = i % 2  # 0th edge is not in match, 1st edge is in match, 2nd edge is not, alternating
                        if not parity:
                            assert eid not in match
                        elif parity:
                            assert eid in match
        result = sorted(result, key=lambda x:x[1])
        return result

    def get_alt_red_paths_with_pos_gain(self, match=[]):
        ''' Wrapper function for get_shortest_improving_paths_wrt_max_match, get_alt_red_paths_with_pos_gain returns
        a list of vertex disjoint alternating paths of red-edges with a positive gain, greedily constructed from a
        sorted list of alternating paths with positive gain.'''
        vertex_disjoint_choices = []
        wtd_paths = self.get_shortest_improving_paths_wrt_max_match(match)
        paths = [x[0] for x in wtd_paths]
        if len(paths) > 0:
            for p in paths:
                d = True
                if len(vertex_disjoint_choices) > 0:
                    for pp in vertex_disjoint_choices:
                        # print(pp, len(pp))
                        d = d and disjoint(p, pp)
                if d:
                    vertex_disjoint_choices += [p]
        return vertex_disjoint_choices

    def get_optimal_red_matching(self, match=[]):
        ''' Compute an optimal/heaviest red matching'''
        # First, take input match and expand to a maximal match.
        assert len(self.left_nodes) > 0
        assert len(self.right_nodes) > 0
        if len(match)==0:
            i = random.choice(list(self.left_nodes.keys()))
            j = random.choice(list(self.right_nodes.keys()))
            match = [(i,j)]

        match = self.get_max_red_matching(match)
        # Second, find a list of vertex disjoint alternating red paths with positive gain.
        alt_paths_to_add = self.get_alt_red_paths_with_pos_gain(match)
        while len(alt_paths_to_add) > 0:
            for p in alt_paths_to_add:
                # Symdif each path
                match = symdif(match, p)
            # Find a new list of vertex disjoint alternating red paths with positive gain
            alt_paths_to_add = self.get_alt_red_paths_with_pos_gain(match)
        return match


def foo(x, y):
    if y:
        print(x)

