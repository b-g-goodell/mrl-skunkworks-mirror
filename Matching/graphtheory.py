from collections import deque
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
        data        : arbitrary.
        count       : integer.
        left_nodes  : dictionary. "nodes side 0." keys & vals are node_idents (integer)
        right_nodes : dictionary. "nodes side 1." keys & vals are node_idents (integer)
        blue_edges  : dictionary. "edges color 0." keys & vals are edge_idents (integer pairs)
        red_edges   : dictionary. "edges color 1." keys are edge_idents (integer pairs), vals are non-positive floats.

    Functions (see below for comments)
        add_node
        add_edge
        del_edge
        del_node
        check_colored_match
        check_colored_maximal_match
        extend_match
        boost_match
        breadth_first_search

    TODO: paths need to be checked to verify they aren't walks! Walks can return to a node or edge, paths can't.
    """

    def __init__(self, par=None):
        if par is None:
            par = dict()
            par['data'] = 'han-tyumi'
            par['count'] = 1
            par['left_nodes'] = dict()
            par['right_nodes'] = dict()
            par['red_edges'] = dict()
            par['blue_edges'] = dict()
        if 'data' not in par:
            par['data'] = 'han-tyumi'
        if 'count' not in par:
            par['count'] = 1
        if 'left_nodes' not in par:
            par['left_nodes'] = {}
        if 'right_nodes' not in par:
            par['right_nodes'] = {}
        if 'red_edges' not in par:
            par['red_edges'] = {}
        if 'blue_edges' not in par:
            par['blue_edges'] = {}
        self.data = par['data']   # str
        if par['count'] <= len(par['left_nodes'])+len(par['right_nodes']):
            self.count = len(par['left_nodes'])+len(par['right_nodes'])+1  # integer
        else:
            self.count = par['count']
        self.left_nodes, self.right_nodes, self.red_edges, self.blue_edges = dict(), dict(), dict(), dict()
        self.left_nodes.update(par['left_nodes'])
        self.right_nodes.update(par['right_nodes'])
        self.red_edges.update(par['red_edges'])
        self.blue_edges.update(par['blue_edges'])

    def add_node(self, b):
        """ Take a bit b indicating side as input. Check that b is a bit and return None if it isn't. Otherwise, create
        a new node on side b, outputting the node_ident.
        """
        assert b in [0, 1]
        result = None
        if b or not b:
            result = self.count
            self.count += 1
            if b:
                # right nodes are side one
                self.right_nodes.update({result: result})
            elif not b:
                # left nodes are side zero
                self.left_nodes.update({result: result})
        return result

    def add_edge(self, b, eid, w):
        """ Take (b, eid, w) as input (a bit b indicating color, edge ident eid, float w indicating weight).
        Fail (output False) if :
          eid[0] not in left_nodes  or
          eid[1] not in right_nodes or
          eid        in blue_edges  or
          eid        in red_edges
        Otherwise create a new edge with color b and output True.
        """
        (x, y) = eid
        result = x in self.left_nodes
        result = result and y in self.right_nodes
        result = result and (x, y) not in self.blue_edges
        result = result and (x, y) not in self.red_edges
        new_dict = {(x, y): w}
        if result and not b:
            self.blue_edges.update(new_dict)
        elif result and b:
            self.red_edges.update(new_dict)
        return result

    def del_edge(self, eid):
        """ Take (x,y) as input (an edge_ident) and remove from all edge dictionaries.
        """
        result = eid in self.blue_edges or eid in self.red_edges
        if eid in self.blue_edges:
            del self.blue_edges[eid]
        if eid in self.red_edges:
            del self.red_edges[eid]
        return result

    def del_node(self, x):
        """  Take x as input (a node ident).
        Fail (output False) if:
          x is not in left_nodes and
          x is not in right_nodes
        Otherwise remove x from all node dictionaries and remove from either edge dictionary all edge_idents of the
          form (x,y) or (y,x) and output True.
        """
        result = x in self.left_nodes or x in self.right_nodes
        if result:
            eids_to_remove = [eid for eid in list(self.blue_edges.keys()) if (eid[0] == x or eid[1] == x)]
            eids_to_remove += [eid for eid in list(self.red_edges.keys()) if (eid[0] == x or eid[1] == x)]
            for eid in eids_to_remove:
                if eid in self.blue_edges:
                    del self.blue_edges[eid]
                if eid in self.red_edges:
                    del self.red_edges[eid]
            if x in self.left_nodes:
                del self.left_nodes[x]
            if x in self.right_nodes:
                del self.right_nodes[x]
        return result

    def check_colored_match(self, b, input_match):
        """ Take input_match as input (a list of edge_idents).
          Fail (output False) if:
            any edge in input_match appear in both colored lists or
            any pair of edges in input_match have two different colors or
            any endpoint of any edge in input_match appears on both sides of the graph or
            any endpoint of any edge in input_match is also incident with any other edge in input_match
          Otherwise output True.
        """
        # TODO: Can this be made more pythonic? Should be a way to state result using set comprehension or something?
        result = None
        if len(input_match) == 0:
            # If the input input_match is empty, then it's a trivial match, return True.
            result = True
        elif len(input_match) > 0:
            for eid in input_match:
                # If any edge appears in both colored lists, or any endpoint occurs in both sides, return False.
                # If any endpoint appears in the wrong side or is not present on the correct side, return False.
                if (eid in self.red_edges and eid in self.blue_edges) or \
                        eid[0] not in self.left_nodes or \
                        eid[0] in self.right_nodes or \
                        eid[1] not in self.right_nodes or \
                        eid[1] in self.left_nodes:
                    result = False
                    break
            if result is not False:
                # Check all edges have the same color as the bit passed in.
                # We already checked each edge only has one color, so we only need to check the presence in the
                # correct dictionary.
                bb = input_match[0] in self.red_edges  # extract color of first edge
                if b != bb:
                    result = False
                if result is not False:
                    for eid in input_match:
                        if (not b and eid not in self.blue_edges) or (b and eid not in self.red_edges):
                            result = False
                    if result is not False:
                        # Check each endpoint is adjacent to only one edge from input_match
                        for i in range(len(input_match)-1):
                            for j in range(i+1, len(input_match)):
                                eid = input_match[i]
                                fid = input_match[j]
                                if eid[0] == fid[0] or eid[1] == fid[1]:
                                    result = False
                        if result is not False:
                            result = True
        return result

    def check_colored_maximal_match(self, b, input_match):
        """  Take input_match as input (a list of edge_idents). Fail (output False) if:
            check_colored_match(input_match) is False or
            any pair of unmatched nodes has an edge with the same color as the edges in input_match
          Otherwise output True.
        """

        if max(len(self.left_nodes), len(self.right_nodes)) > 0 and len(input_match) == 0:
            result = False
        else:
            result = self.check_colored_match(b, input_match)
        if result:
            bb = input_match[0] in self.red_edges  # extract color
            if b != bb:
                result = False
            matched_lefts = [eid[0] for eid in input_match]
            matched_rights = [eid[1] for eid in input_match]
            unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
            unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]
            for x in unmatched_lefts:
                for y in unmatched_rights:
                    if (b and (x, y) in self.red_edges) or (not b and (x, y) in self.blue_edges):
                        # we found an edge that could increase match size!
                        result = False
        return result

    def _parse(self, b, input_match):
        """ parse takes a color b and input_match and either crashes because input_match isn't a match with color b
        or returns the following sets for convenient usage in extend_match and boost_match
            matched_lefts    :  left_nodes incident with an edge in input_match
            matched_rights   :  right_nodes incident with an edge in input_match
            unmatched_lefts  :  left_nodes not incident with any edge in input_match
            unmatched_rights :  right_nodes not incident with any edge in input_mach
            apparent_color   :  the observed color of input_match (which should match the input color b)
            non_match_edges  :  edges with color b that are not in input_match
        """
        assert self.check_colored_match(b, input_match)

        # Node sets
        matched_lefts = [eid[0] for eid in input_match]
        matched_rights = [eid[1] for eid in input_match]
        unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
        unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]

        # color
        if len(input_match) > 0:
            apparent_color = input_match[0] in self.red_edges  # extract color
            assert apparent_color == b
        else:
            apparent_color = b

        # non_match_edges
        if apparent_color:
            non_match_edges = [eid for eid in self.red_edges if eid not in input_match]
        elif not apparent_color:
            non_match_edges = [eid for eid in self.blue_edges if eid not in input_match]
        else:
            assert False

        return (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, apparent_color, non_match_edges)

    def _cleanup(self, b, shortest_paths, non_match_edges, input_match):
        """ For each next_path in shortest_paths, compute :
            (i)   the symmetric difference between next_path and input_match and
            (ii)  the cumulative weight of this symmetric difference, and
            (iii) the difference between the cumulative weight from input_match and the weight from (ii)
          Order the list_of_all_shortest_paths by gain, discarding negative gain paths and discarding any paths with
          length greater than the shortest path with positive gain. Construct a vertex-disjoint list of paths, say
          v_d_list_of_paths, from the list_of_all_shortest_paths by greedy algorithm :
            (i)   For the next_highest_gain_path in list_of_all_shortest_paths, check if next_highest_gain_path is
                  vertex-disjoint from each other_path in v_d_list_of_paths.
            (ii)  If so, include next_highest_gain_path in v_d_list_of_paths.
            (iii) Go back to (i) lol
          Compute the symmetric difference between each path in v_d_list_of_paths input_match.
          Return the result.

          An execution of cleanup takes as input a color, a list of paths, a list of non-match edges.

        Note: cleanup could be written that does not take the non-match edges as input, which is unnecessary
        to pass in, and can be conveniently extracted using the color b as in boost_match and extend_match):

        However, we use this data (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, non_match_edges)
        while executing both extend_match and boost_match, and these are the only functions that call cleanup.

        """

        ordered_shortest_paths = []
        if b:
            edge_set = self.red_edges
        elif not b:
            edge_set = self.blue_edges
        else:
            assert False
        for next_path in shortest_paths:
            for eid in next_path:
                assert eid in edge_set
            gain = sum([edge_set[eid] for eid in next_path if eid in non_match_edges])
            gain = gain - sum([edge_set[eid] for eid in next_path if eid in input_match])
            # Discard negative gains
            if gain > 0.0:
                ordered_shortest_paths.append((next_path, gain))

        # Sort by path length and discard any with non-minimal length
        ordered_shortest_paths = sorted(ordered_shortest_paths, key=lambda z: len(z[0]), reverse=False)
        path_length = len(ordered_shortest_paths[0][0])
        temp = []
        for next_pair in ordered_shortest_paths:
            (next_path, gain) = next_pair
            if len(next_path) <= path_length:
                temp += [next_pair]

        # Sort by gain in decreasing order (greedy)
        ordered_shortest_paths = sorted(temp, key=lambda z: z[1], reverse=True)

        # Construct a list of vertex-disjoint paths from ordered_shortest_paths
        v_d_list = []
        touched_nodes = []
        for next_pair in ordered_shortest_paths:
            (next_path, gain) = next_pair
            if len(v_d_list) == 0:
                v_d_list += [next_path]
                touched_nodes += [eid[0] for eid in next_path] + [eid[1] for eid in next_path]
            else:
                good_path = True
                for eid in next_path:
                    if eid[0] in touched_nodes or eid[1] in touched_nodes:
                        good_path = False
                        break
                if good_path:
                    v_d_list += [next_path]
                    touched_nodes += [eid[0] for eid in next_path] + [eid[1] for eid in next_path]

        # Iteratively take symmetric differences
        result = input_match
        if len(v_d_list) > 0:
            for next_path in v_d_list:
                temp = [eid for eid in result if eid not in next_path]
                temp += [eid for eid in next_path if eid not in result]
                result = temp
        return result

    def extend_match(self, b, input_match):
        """ Take input_match as input (a list of edge_idents). Fail (output False) if
        check_colored_maximal_match(input_match) is True, output input_match. Otherwise:
          Use a breadth-first-search to find all shortest paths* that :
            (i)   consist of edges with the same color as the edges in input_match and
            (ii)  alternate with respect to input_match and
            (iii) begin and end with unmatched nodes
          Call this list of all shortest paths shortest_paths. Then calls cleanup and returns the result.
        """
        result = None
        if self.check_colored_match(b, input_match):
            if self.check_colored_maximal_match(b, input_match):
                result = input_match
            else:
                shortest_paths = []
                found_shortest_path = False
                path_length = None

                (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, bb, non_match_edges) = \
                    self._parse(b, input_match)

                q = deque([[eid] for eid in non_match_edges if eid[0] in unmatched_lefts])
                assert len(q) > 0  # must be true since the match is maximal

                while len(q) > 0:
                    next_path = q.popleft()
                    # Check that if we have found a shortest path, then the path_length has been set.
                    assert not found_shortest_path or path_length is not None
                    # Discard paths longer than the (thus far) minimal path length
                    if path_length is None or (path_length is not None and len(next_path) < path_length):
                        touched_nodes = [eid[0] for eid in next_path] + [eid[1] for eid in next_path]
                        last_edge = next_path[-1]
                        p = len(next_path) % 2  # parity of the path. No offset since the path is augmenting/symmetric
                        last_node = last_edge[p]
                        if p:
                            edge_set_to_search = input_match
                        elif not p:
                            edge_set_to_search = non_match_edges
                        else:
                            assert False

                        for eid in edge_set_to_search:
                            if eid[p] == last_node and eid[1-p] not in touched_nodes:
                                path_to_add = next_path + [eid]
                                if eid[1-p] in unmatched_rights and path_length is None or len(path_to_add) < path_length:
                                    shortest_paths = [path_to_add]
                                    path_length = len(path_to_add)
                                    found_shortest_path = True
                                elif eid[1-p] in unmatched_rights and \
                                        path_length is not None and \
                                        len(path_to_add) == path_length:
                                    shortest_paths += [path_to_add]
                                elif path_length is None or (path_length is not None and len(path_to_add) < path_length):
                                    q.append(path_to_add)
                                else:
                                    assert False

            result = self._cleanup(b, shortest_paths, non_match_edges, input_match)
        return result

    def boost_match(self, b, input_match):
        """ Note : maximality of input_match implies no *augmenting* paths exist, i.e. alternating and beginning and
        ending with unmatched nodes (this is a theorem, that augmenting paths exist IFF the match is maximal).

        However, there may exist half-augmenting paths that alternate with respect to the match, with positive
        gain, that begin with an unmatched node, and that are maximal in the sense that they terminate in a node whose
        only matched neighbors are on the path already. Also, there may exist augmenting alternating cycles that only
        touch matched nodes and alternate with respect to the match.

        Note that this sense of maximality is not maximal in terms of length. Not all half-augmenting paths or cycles
        satisfying these conditions are of the same length, but they are all maximal in the sense that they cannot be
        extended. Of all these paths, we want the shortest maximal half-augmenting paths and augmenting cycles with
        a positive gain.

        We can capture both situations by using a breadth-first search looking for all shortest paths that (i) start
        from any left node with any non-match edge to any matched right node, (ii) alternate with respect to the match,
        and (iii) (maximality) terminate in any node that has no remaining matched neighbors except perhaps on the path.
        """
        result = None
        if self.check_colored_maximal_match(b, input_match):
            shortest_paths = []
            found_shortest_path = False
            path_length = None

            (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, bb, non_match_edges) = \
                self._parse(b, input_match)

            q = deque([[eid] for eid in non_match_edges])
            while len(q) > 0:
                # Check that if we have found a shortest path, then the path_length has been set.
                assert not found_shortest_path or path_length is not None
                # Pop this path for processing
                next_path = q.popleft()
                # Don't bother with paths longer than the (thus far) minimal path length
                if path_length is None or (path_length is not None and len(next_path) < path_length):

                    touched_nodes = [eid[0] for eid in next_path] + [eid[1] for eid in next_path]
                    last_edge = next_path[-1]
                    p = len(next_path) % 2  # parity of the path. No offset since the path is augmenting/symmetric
                    last_node = last_edge[p]
                    if p:
                        edge_set_to_search = input_match
                    elif not p:
                        edge_set_to_search = non_match_edges
                    else:
                        assert False

                    extensible = len(edge_set_to_search) > 0
                    if extensible:
                        for eid in edge_set_to_search:
                            if eid[p] == last_node and eid[1 - p] not in touched_nodes:
                                path_to_add = next_path + [eid]
                                q.append(path_to_add)
                    else:
                        if path_length is None or len(path_to_add) < path_length:
                            shortest_paths = [path_to_add]
                            path_length = len(path_to_add)
                            found_shortest_path = True
                        elif len(path_to_add) == path_length:
                            shortest_paths += [path_to_add]
                            found_shortest_path = True
            result = self._cleanup(b, shortest_paths, non_match_edges, input_match)
        return result
