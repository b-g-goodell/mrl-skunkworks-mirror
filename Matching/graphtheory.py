from collections import deque
import sys

# Check version number
if sys.version_info[0] != 3:
    print("This script requires Python3")
    sys.exit(1)


class BipartiteGraph(object):
    """ Graph object representing a graph

    Attributes:
        data        : arbitrary.
        count       : integer.
        left_nodes  : dictionary. "nodes side 0." keys & vals are node_idents (integer)
        right_nodes : dictionary. "nodes side 1." keys & vals are node_idents (integer)
        blue_edges  : dictionary. "edges color 0." keys & vals are edge_idents (integer pairs)
        red_edges   : dictionary. "edges color 1." keys are edge_idents (integer pairs), vals are non-negative numbers.

    Functions (see below for comments)
        add_node
        add_edge
        del_edge
        del_node
        chk_colored_match
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
        if b or not b:
            result = self.count
            self.count += 1
            if b:
                # right nodes are side one
                self.right_nodes.update({result: result})
                for left_node in self.left_nodes:
                    self.add_edge(0, (left_node, result), 0.0)
                    self.add_edge(1, (left_node, result), 0.0)
            elif not b:
                # left nodes are side zero
                self.left_nodes.update({result: result})
                for right_node in self.right_nodes:
                    self.add_edge(0, (result, right_node), 0.0)
                    self.add_edge(1, (result, right_node), 0.0)
        return result

    def add_edge(self, b, eid, w):
        """ Take (b, eid, w) as input (a bit b indicating color, edge ident eid, float w indicating weight).
        Fail (output False) if :
          eid[0] not in left_nodes  or
          eid[1] not in right_nodes
        Otherwise update edge with color b and edge id eid to have weight w. Note: this OVER-WRITES any previous
        edge weight. Since we always add all red edges with weight 0.0, this allows us to add weights to edges that
        exist in the graph, and assume weight 0.0 for edges that do not.

        Allow non-zero weights only for one color at a time for an edge.
        """
        assert b in [0, 1]
        assert w >= 0.0
        (x, y) = eid  # parse eid to get left node and right node ids
        result = x in self.left_nodes
        result = result and y in self.right_nodes
        if result:
            new_dict = {eid: w}
            zero_dict = {eid: 0.0}
            if b:
                self.red_edges.update(new_dict)
                self.blue_edges.update(zero_dict)
            elif not b:
                self.red_edges.update(zero_dict)
                self.blue_edges.update(new_dict)
        return result

    def del_edge(self, eid):
        """ Take (x,y) as input (an edge_ident). If both of these are legitimate nodes, then we set edge weights in
        both edge dictionaries to 0.0. If either node does not exist, then the edge entry is deleted from the dictionary
        """
        if eid[0] not in self.left_nodes or eid[1] not in self.right_nodes:
            if eid in self.blue_edges:
                del self.blue_edges[eid]
            if eid in self.red_edges:
                del self.red_edges[eid]
            result = True
        else:
            self.blue_edges.update({eid: 0.0})
            self.red_edges.update({eid: 0.0})
            result = True
        return result

    def del_node(self, x):
        """  Take x as input (a node ident).
        Fail (output False) if:
          x is not in left_nodes and
          x is not in right_nodes
        Otherwise remove x from all node dictionaries and remove from both edge dictionaries all edge_idents of the
          form (x,y) or (y,x) and output True.
        """
        result = x in self.left_nodes or x in self.right_nodes
        if result:
            if x in self.left_nodes:
                del self.left_nodes[x]
            if x in self.right_nodes:
                del self.right_nodes[x]
            eids_to_remove = [eid for eid in list(self.blue_edges.keys()) if (eid[0] == x or eid[1] == x)]
            eids_to_remove += [eid for eid in list(self.red_edges.keys()) if (eid[0] == x or eid[1] == x)]
            for eid in eids_to_remove:
                self.del_edge(eid)
        return result

    def chk_colored_match(self, b, input_match):
        if len(input_match) == 0:
            result = True
        else:
            left_vertices = [eid[0] for eid in input_match]
            dedupe_lefts = list(set(left_vertices))
            right_vertices = [eid[1] for eid in input_match]
            dedupe_rights = list(set(right_vertices))
            vtx_disj = (len(dedupe_lefts)==len(dedupe_rights) and len(dedupe_lefts)==len(input_match))
            if vtx_disj:
                blue_wt = sum([self.blue_edges[eid] for eid in input_match])
                red_wt = sum([self.red_edges[eid] for eid in input_match])
                all_one_color = ((blue_wt > 0.0 and red_wt == 0.0) or (blue_wt == 0.0 and red_wt > 0.0))
                correct_color = all_one_color and ((not b and blue_wt > 0.0) or (b and red_wt > 0.0))
            result = vtx_disj and correct_color
        return result

    def check_colored_maximal_match(self, b, input_match):
        """  Take input_match as input (a list of edge_idents). Fail (output False) if:
            chk_colored_match(input_match) is False or
            any pair of unmatched nodes has an edge with the same color as the edges in input_match with nonzero weight
          Otherwise output True.
        """

        if max(len(self.left_nodes), len(self.right_nodes)) > 0 and len(input_match) == 0:
            result = False
        else:
            result = self.chk_colored_match(b, input_match)
        if result:
            matched_lefts = [eid[0] for eid in input_match]
            matched_rights = [eid[1] for eid in input_match]
            unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
            unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]
            for x in unmatched_lefts:
                for y in unmatched_rights:
                    w = None
                    if b and self.red_edges[(x, y)] > 0.0:
                        w = self.red_edges[(x, y)]
                    elif not b and self.blue_edges[(x, y)] > 0.0:
                        w = self.blue_edges[(x, y)]
                    if w is not None and w > 0.0:
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
        assert self.chk_colored_match(b, input_match)

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

        # non-match non-zero edges
        if apparent_color:
            non_match_edges = [eid for eid in self.red_edges if eid not in input_match and self.red_edges[eid] > 0.0]
        elif not apparent_color:
            non_match_edges = [eid for eid in self.blue_edges if eid not in input_match and self.blue_edges[eid] > 0.0]
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
        # TODO: Special case: when shortest_paths is an empty list, the input_match is already optimal.
        ordered_shortest_paths = []
        if b:
            edge_set = self.red_edges
        elif not b:
            edge_set = self.blue_edges
        else:
            assert False

        # Sort by gain in decreasing order (greedy)
        ordered_shortest_paths = sorted(shortest_paths, key=lambda z: z[1], reverse=True)

        # Construct a list of vertex-disjoint paths from ordered_shortest_paths
        v_d_list = []
        touched_nodes = []
        for next_pair in ordered_shortest_paths:
            (next_path, gain) = next_pair
            if len(v_d_list) == 0:
                v_d_list = [next_path]
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

    def extend(self, b, input_match):
        """ Find all shortest paths P satisfying the following constraints and call _cleanup with them.
              (i)   all edges in P share the same color with input_match and
              (ii)  edges in P alternate with respect to input_match and
              (iii) the initial endpoint of P is an unmatched node and
              (iv)  P cannot be extended by any correctly colored edges alternating with respect to input_match without
                    self-intersecting
        """
        result = None
        assert b in [0, 1]
        if b:
            weight_dict = self.red_edges
        else:
            weight_dict = self.blue_edges
        if self.chk_colored_match(b, input_match):
            shortest_paths = [] # solution box
            found = False
            length = None
            if self.check_colored_maximal_match(b, input_match):
                shortest_paths = []
                result = input_match
            else:
                (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, bb, non_match_edges) = \
                    self._parse(b, input_match)
                q = deque([[eid] for eid in non_match_edges if eid[0] in unmatched_lefts or eid[1] in unmatched_rights])
                assert len(q) > 0
                while len(q) > 0:
                    assert not found or (length is not None and len(shortest_paths) > 0)
                    next_path = q.popleft()

                    if length is None or length is not None and len(next_path) <= length:
                        # Paths have distinct edges and distinct vertices. Cycles are like paths, but the first and last
                        # vertices are allowed to match.

                        # Collect some info for convenience
                        first_edge = next_path[0]
                        first_node = first_edge[0]
                        last_edge = next_path[-1]

                        p = len(next_path) % 2  # path parity
                        assert p in [0, 1]
                        last_node = last_edge[p]
                        num_distinct_edges = len(list(set(next_path)))  # count distinct edges

                        # collect sequence of nodes
                        temp = []
                        for i in range(len(next_path)):
                            eid = next_path[i]
                            idx_p = i % 2  # index parity
                            temp += [eid[idx_p]]
                        temp += [eid[1 - idx_p]]

                        distinct_nodes = list(set(temp))
                        num_distinct_nodes = len(distinct_nodes)  # count distinct vertices

                        if len(next_path) == num_distinct_edges and len(temp) == num_distinct_nodes:
                            # Extend the path with allowable alternating edges, placing the extended paths back into the
                            # queue. If the path cannot be extended, compute the gain and if positive place in solution
                            # box
                            if p:
                                edge_set_to_search = input_match
                            else:
                                edge_set_to_search = non_match_edges

                            these_edges = [eid for eid in edge_set_to_search if (eid not in next_path and eid[p] == last_node and eid[1-p] not in distinct_nodes)]
                            if len(these_edges) > 0:
                                for eid in these_edges:
                                    path_to_add = None
                                    path_to_add = next_path + [eid]
                                    q.append(path_to_add)
                                    path_to_add = None
                            else:
                                sd = [weight_dict[eid] for eid in next_path if eid not in input_match]
                                sd += [weight_dict[eid] for eid in input_match if eid not in next_path]
                                gain = sum(sd) - sum([weight_dict[eid] for eid in input_match])
                                if gain > 0.0:
                                    if length is None or len(next_path) < length:
                                        shortest_paths = [(next_path, gain)]
                                        length = len(next_path)
                                        found = True
                                    elif length is not None and len(next_path) == length:
                                        shortest_paths += [(next_path, gain)]
                # Check that found_shortest_path = True if and only if at least one shortest path is in shortest_paths
                assert not found or len(shortest_paths) > 0
                assert len(shortest_paths) == 0 or found
                if len(shortest_paths) > 0:
                    # print(shortest_paths)
                    result = self._cleanup(b, shortest_paths, non_match_edges, input_match)
                else:
                    result = input_match
        return result

    def boost(self, b, input_match):
        """ Should be called after

        Note : maximality of input_match implies no *augmenting* paths exist, i.e. alternating and beginning and
        ending with unmatched nodes (this is a theorem, that augmenting paths exist IFF the match is maximal).

        However, there exist alternating cycles, and the optimal match can be written as the symmetric difference bet-
        ween the input match and a sequence of vertex-disjoint alternating cycles with positive gain.
        """
        result = None
        assert b in [0, 1]
        if b:
            weight_dict = self.red_edges
        else:
            weight_dict = self.blue_edges
        if self.check_colored_maximal_match(b, input_match):
            shortest_cycles_with_pos_gain = [] # solution box
            found = False
            length = None

            (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, bb, non_match_edges) = \
                self._parse(b, input_match)

            q = deque([[eid] for eid in non_match_edges])
            while len(q) > 0:
                assert not found or length is not None

                next_path = q.popleft()  # Get next path

                if length is None or length is not None and len(next_path) <= length:
                    # Paths have distinct edges and distinct vertices. Cycles are like paths, but the first and last
                    # vertices are allowed to match.
                    # Certain paths are degenerate in that they lead into a cycle, looping back on themselves, leading
                    # to a union of a path and a cycle. These are not alternating cycles and are discarded. The em-
                    # bedded cycle will be found earlier in the algorithm anyway

                    # Collect some info for convenience
                    first_edge = next_path[0]
                    first_node = first_edge[0]
                    last_edge = next_path[-1]

                    p = len(next_path) % 2  # path parity
                    assert p in [0, 1]
                    last_node = last_edge[p]
                    num_distinct_edges = len(list(set(next_path)))  # count distinct edges

                    # collect sequence of nodes
                    temp = []
                    for i in range(len(next_path)):
                        eid = next_path[i]
                        idx_p = i % 2  # index parity
                        temp += [eid[idx_p]]
                    temp += [eid[1-idx_p]]


                    num_distinct_nodes = len(list(set(temp)))  # count distinct vertices

                    if len(next_path) == num_distinct_edges:
                        # paths have distinct edges.
                        if len(temp) - num_distinct_nodes in [0, 1]:
                            # paths of N distinct edges have N+1 vertices, except when the path is degenerate or a cycle
                            # in which case they have N vertices.
                            if len(temp) - num_distinct_nodes == 1:
                                # such a path is either a cycle or is degenerate.
                                if first_node == last_node:
                                    # this path is a cycle. If it has positive gain, put it into the solution box.
                                    sd = [weight_dict[eid] for eid in next_path if eid not in input_match]
                                    sd += [weight_dict[eid] for eid in input_match if eid not in next_path]
                                    gain = sum(sd) - sum([weight_dict[eid] for eid in input_match])
                                    if gain > 0.0:
                                        found = True
                                        if length is None or len(next_path) < length:
                                            length = len(next_path)
                                            shortest_cycles_with_pos_gain = []
                                            shortest_cycles_with_pos_gain += [(next_path, gain)]
                                        elif len(next_path) == length:
                                            shortest_cycles_with_pos_gain += [(next_path, gain)]
                                        else:
                                            # In this case, from the first if statement, length is not None (so a
                                            # shortest cycle has been found) and len(next_path) >= length, but, from the
                                            # second if statement, also len(next_path) != length, so
                                            # len(next_path) > length necessarily. This cycle is too long, discard it.
                                            pass
                                else:
                                    # this path is degenerate and is discarded
                                    pass
                            else:
                                # this path is not a cycle and is made of all distinct vertices. Extend the path with
                                # allowable alternating edges, placing the extended paths back into the queue.
                                if p:
                                    edge_set_to_search = input_match
                                else:
                                    edge_set_to_search = non_match_edges

                                for eid in edge_set_to_search:
                                    if eid not in next_path and eid[p] == last_node:
                                        path_to_add = None
                                        path_to_add = next_path + [eid]
                                        q.append(path_to_add)
                                        path_to_add = None
                else:
                    # in this case, length is not None and len(next_path) > length, so we discard next_path
                    pass

            result = self._cleanup(b, shortest_cycles_with_pos_gain, non_match_edges, input_match)
        return result

    def optimize(self, b):
        result = None
        assert b in [0, 1]
        input_match = []
        next_match = self.extend(b, input_match)
        while next_match != input_match:
            input_match = next_match
            next_match = self.extend(b, input_match)
        if b:
            weight_dict = self.red_edges
        else:
            weight_dict = self.blue_edges

        input_match = next_match
        w = sum([weight_dict[eid] for eid in input_match])
        next_match = self.boost(b, input_match)
        v = sum([weight_dict[eid] for eid in next_match])
        while v - w > 0.0:
            w = v
            input_match = next_match
            next_match = self.boost(b, input_match)
            v = sum([weight_dict[eid] for eid in next_match])

        return next_match
