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
        ''' Find all augmenting paths with respect to the match by using breadth-first search. '''
        result = None # Receiving None as output means input was not a matching on the nodes.
        q = deque()
        found_shortest_path = False
        len_shortest_path = None

        if self._check_red_match(match):
            # At least pass back an empty list if match is a matching
            result = [] 

            # Seed empty match with a single edge, guaranteeing at least one matched node of either side
            # TODO: Better solution: throw error if input is empty and write scripts to only call with nonempty
            if len(match)==0:
                pairFound = False
                while not pairFound:
                    for j in self.right_nodes.keys():
                        for i in self.left_nodes.keys():
                            if (i,j) in self.red_edge_weights:
                                pairFound = True
                                break
                        if pairFound:
                            break
                assert pairFound
                match = [(i,j)]

            # Assemble a few node lists for convenience
            matched_lefts = [eid[0] for eid in match]
            matched_rights = [eid[1] for eid in match]
            unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
            unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]

            if len(unmatched_rights) == 0 or len(unmatched_lefts) == 0:
                print("UPDATING RESULT")
                old_len = len(result)
                result = match
                new_len = len(result)
                print(result)
                assert new_len - old_len > 0
            else:
                for eid in self.red_edge_weights:
                    if eid[0] in unmatched_lefts:
                        if eid[1] in unmatched_rights:
                            found_shortest_path = True
                            if len_shortest_path is not None:
                                assert len_shortest_path == 1
                            else:
                                len_shortest_path = 1
                            print("UPDATING RESULT")
                            old_len = len(result)
                            result.append([eid])
                            new_len = len(result)
                            print(result)
                            assert new_len - old_len > 0
                        else:
                            q.append([eid])
                if not found_shortest_path:
                    while len(q) > 0:
                        this_path = q.popleft()
                        parity = len(this_path) % 2
                        last_edge = this_path[-1]
                        last_node = last_edge[parity]
                        tn = [eid[0] for eid in this_path] + [eid[1] for eid in this_path]
                        tn = list(dict.fromkeys(tn)) # dedupe

                        for eid in self.red_edge_weights: 
                            if eid[1-parity] not in tn and eid not in this_path and eid[parity] == last_node:
                                next_path = this_path + [eid]
                                if parity and eid in match and not found_shortest_path:
                                    q.append(next_path)
                                elif not parity and eid not in match:
                                    if eid[1] in unmatched_rights:
                                        found_shortest_path = True
                                        if len_shortest_path is not None:
                                            assert len_shortest_path == len(next_path)
                                        else:
                                            len_shortest_path = len(next_path)
                                        print("UPDATING RESULT")
                                        old_len = len(result)
                                        result.append(next_path)
                                        new_len = len(result)   
                                        print(result)                            
                                        assert new_len - old_len > 0
                                    elif not found_shortest_path:
                                        q.append(next_path)
            print("lasting result 1 = ", result)  
        print("lasting result 2 = ", result)
        # Before returning our result, we check each path in the result is alternating
        # with respect to the input match and all have the same length.
        l = len(result[0])
        for p in result:
            assert len(p) == l
            for i in range(l):
                parity = (i+1)%2
                if parity:
                    assert p[i] not in match
                elif not parity:
                    assert p[i] in match
            
        return result

    def get_augmenting_red_paths(self, match=None):
        # Returns list of vertex disjoint augmenting paths of red-edges
        vertex_disjoint_choices = []
        if match is None:
            match = []
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
                    if len(vertex_disjoint_choices) > 0:
                        for pp in vertex_disjoint_choices:
                            # print(pp, len(pp))
                            vertices_in_pp = [eid[0] for eid in pp] + [eid[1] for eid in pp]
                            d = d and disjoint(vertices_in_p, vertices_in_pp)
                    if d:
                        vertex_disjoint_choices += [p]
        return vertex_disjoint_choices

    def get_bigger_red_matching(self, match=None):
        # print("Beginning _get_bigger_red_matching")
        result = None
        if match is None:
            match = []
        if self._check_red_match(match):
            vdc = self.get_augmenting_red_paths(match)
            if len(vdc) > 0:
                for p in vdc:
                    temp = match
                    swap = symdif(temp, p)
                    match = swap
            result = match
        return result

    def get_max_red_matching(self, match=None):
        # line = [e.ident for e in match]
        # print("\n\nBeginning max_matching with match = " + str(line))
        if match is None:
            match = []
        next_match = self.get_bigger_red_matching(match)
        # line = [e.ident for e in next_match]
        # print("First iteration gets us " + str(line))
        while len(next_match) > len(match):
            match = next_match
            next_match = self.get_bigger_red_matching(match)
            # line = [e.ident for e in next_match]
            # print("Next iteration gets us " + str(line))
        return match

    def get_alt_red_paths_with_pos_gain(self, match=None):
        # Returns list of vertex disjoint alternating paths of red-edges with a positive gain
        vertex_disjoint_choices = []
        if match is None:
            match = []
        shortest_paths = self.bfs(match)
        if len(shortest_paths) > 0:
            sorted_paths = None
            if wt:
                weighted_paths = []
                for p in shortest_paths:
                    gain = 0.0
                    for eid in p:
                        if eid not in match:
                            gain = gain + self.red_edge_weights[eid]
                        else:
                            gain = gain - self.red_edge_weights[eid]
                    if gain > 0.0:
                        weighted_paths += [[gain, p]]
                weighted_sorted_paths = sorted(weighted_paths, key=lambda x: x[0], reverse=True)
                sorted_paths = [x[1] for x in weighted_sorted_paths]
            else:
                x = [i for i in range(len(shortest_paths))]
                random.shuffle(x)
                sorted_paths = [shortest_paths[i] for i in x]
            if sorted_paths is not None and len(sorted_paths) > 0:
                for p in sorted_paths:
                    d = True
                    if len(vertex_disjoint_choices) > 0:
                        for pp in vertex_disjoint_choices:
                            # print(pp, len(pp))
                            d = d and disjoint(p, pp)
                    if d:
                        vertex_disjoint_choices += [p]
        return vertex_disjoint_choices

    def get_optimal_red_matching(self, match=None):
        match = self.get_max_red_matching(match)
        alt_paths_to_add = self.get_alt_red_paths_with_pos_gain(match)
        while len(alt_paths_to_add) > 0:
            for p in alt_paths_to_add:
                match = symdif(match, p)
            alt_paths_to_add = self.get_alt_red_paths_with_pos_gain(match)
        return match


def foo(x, y):
    if y:
        print(x)

