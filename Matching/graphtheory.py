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
        count       : integer, used for naming new nodes
        left_nodes  : dictionary. "nodes side 0." keys & vals are node_idents (integer)
        right_nodes : dictionary. "nodes side 1." keys & vals are node_idents (integer)
        blue_edges  : dictionary. "edges color 0." keys are edge_idents (integer pairs), vals are non-negative numbers.
        red_edges   : dictionary. "edges color 1." keys are edge_idents (integer pairs), vals are non-negative numbers.

    Functions (see below for comments)
        add_node
        add_edge
        del_edge
        del_node
        chk_colored_match
        check_colored_maximal_match
        extend
        boost

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
        """ Take a bit b indicating side as input (b = 0 are left nodes, b = 1 are right nodes). Check that b is a bit
        and return None if it isn't. Otherwise, create a new node on side b, outputting the node_ident.
        """
        result = None
        assert b in [0, 1]
        result = self.count
        assert result not in self.right_nodes and result not in self.left_nodes
        self.count += 1
        if b==1:
            # right nodes are side one
            self.right_nodes.update({result: result})
            for left_node in self.left_nodes:
                self.add_edge(0, (left_node, result), 0.0)
                self.add_edge(1, (left_node, result), 0.0)
        elif b==0:
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
        edge weight of color b with weight w and over-writes the weight of color 1-b with weight 0.0.

        """
        assert b in [0, 1]
        assert w >= 0.0
        (x, y) = eid  # parse eid to get left node and right node ids
        result = x in self.left_nodes
        result = result and y in self.right_nodes
        if result:
            new_dict = {eid: w}
            zero_dict = {eid: 0.0}
            if b==1:
                self.red_edges[eid]=w
                self.blue_edges[eid]=0.0
            else:
                self.red_edges[eid]=0.0
                self.blue_edges[eid]=w
        return result

    def del_edge(self, eid):
        """ Take eid = (x, y) as input (an edge_ident). If both of these are nodes on the proper sides, then we set edge
        weights in both edge dictionaries to 0.0. If either node does not exist or is on the wrong side, then the edge
        entry is deleted from the dictionary. This always succeeds so we return True.
        """
        if eid[0] not in self.left_nodes or eid[1] not in self.right_nodes:
            if eid in self.blue_edges:
                del self.blue_edges[eid]
            if eid in self.red_edges:
                del self.red_edges[eid]
        else:
            self.blue_edges.update({eid: 0.0})
            self.red_edges.update({eid: 0.0})
        return True

    def del_node(self, x):
        """  Take x as input (a node ident).
        Fail (output False) if:
          x is not in left_nodes and
          x is not in right_nodes
        Otherwise remove x from all node dictionaries and delete both color edge identities like (x, y) and (y, x).
        This always succeeds so we return True.
        """
        if x in self.left_nodes or x in self.right_nodes:
            if x in self.left_nodes:
                del self.left_nodes[x]
            if x in self.right_nodes:
                del self.right_nodes[x]

        eids_to_remove = [eid for eid in list(self.blue_edges.keys()) if (eid[0] == x or eid[1] == x)]
        eids_to_remove += [eid for eid in list(self.red_edges.keys()) if (eid[0] == x or eid[1] == x)]
        for eid in eids_to_remove:
            self.del_edge(eid)

        return True

    def chk_colored_match(self, b, input_match):
        """ Take as input a color b and an input_match. If input_match is empty, then it's a trivial match and we return
        True. Otherwise, we check if input_match is a single-colored match with color b, returning True if so and
        False otherwise.
        """
        if len(input_match) == 0:
            result = True
        else:
            # Every left vertex incident with an edge in the purported match must be incident with exactly one edge.
            left_vertices = [eid[0] for eid in input_match]
            # If a left vertex is duplicated in this list, then two edges are incident with the same left node.
            dedupe_lefts = list(set(left_vertices))
            # Every right vertex incident with an edge in the purported match must be incident with exactly one edge.
            right_vertices = [eid[1] for eid in input_match]
            # If a right vertex is duplicated, then two edges are incident with the same right node.
            dedupe_rights = list(set(right_vertices))
            # Therefore this check verifies that the edge set is pairwise vertex disjoint!
            vtx_disj = (len(dedupe_lefts) == len(dedupe_rights) and len(dedupe_lefts) == len(input_match))
            # Now we check that the match is all of one color, and that color is b.
            correct_color = False
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
            there exists an edge with color b and non-zero weight not in input_match with unmatched endpoints
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
                    if (b and self.red_edges[(x, y)] > 0.0) or (not b and self.blue_edges[(x, y)] > 0.0):
                        # we found an edge that could increase match size!
                        result = False
        return result

    def _parse(self, b, input_match):
        """ parse takes a color b and input_match and either crashes because input_match isn't a match with color b
        or returns the following lists for convenient usage in extend and boost.
            matched_lefts    :  left_nodes incident with an edge in input_match
            matched_rights   :  right_nodes incident with an edge in input_match
            unmatched_lefts  :  left_nodes not incident with any edge in input_match
            unmatched_rights :  right_nodes not incident with any edge in input_mach
            apparent_color   :  the observed color of input_match (which should match the input color b)
            non_match_edges  :  edges with color b that are not in input_match, including zero-weight edges
        """
        # Check input_match is a match of color b
        assert self.chk_colored_match(b, input_match)
        assert b in [0, 1]

        # Node sets
        matched_lefts = [eid[0] for eid in input_match]
        matched_rights = [eid[1] for eid in input_match]
        unmatched_lefts = [nid for nid in self.left_nodes if nid not in matched_lefts]
        unmatched_rights = [nid for nid in self.right_nodes if nid not in matched_rights]

        # non-match edges
        if b:
            non_match_edges = [eid for eid in self.red_edges if eid not in input_match]
        else:
            non_match_edges = [eid for eid in self.blue_edges if eid not in input_match]

        result = (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, b, non_match_edges)
        return result

    @staticmethod
    def _clean(b, shortest_paths_with_gains, input_match):
        """ Returns the input_match (when input shortest_paths is empty) or the iterative symdif of input_match """
        assert b in [0, 1]
        result = input_match
        if len(shortest_paths_with_gains) > 0:
            # Order the list
            ordered_shortest_paths = sorted(shortest_paths_with_gains, key=lambda z: z[1], reverse=True)
            # Construct vertex-disjoint list greedily
            vd = []
            touched_nodes = []
            for path_and_gain in ordered_shortest_paths:
                (next_path, gain) = path_and_gain
                if gain > 0.0:
                    # Collect some info for convenience
                    # first_edge = next_path[0]
                    # first_node = first_edge[0]
                    # last_edge = next_path[-1]
                    assert len(next_path) >= 1
                    p = len(next_path) % 2  # path parity
                    assert p in [0, 1]
                    # last_node = last_edge[p]
                    # num_distinct_edges = len(list(set(next_path)))  # count distinct edges
                    # collect sequence of nodes
                    temp = []
                    if len(next_path) >= 1:
                        idx_p = None
                        for i in range(len(next_path)):
                            eid = next_path[i]
                            idx_p = i % 2  # index parity
                            temp += [eid[idx_p]]
                        temp += [next_path[-1][1 - idx_p]]
                    distinct_nodes = list(set(temp))
                    # num_distinct_nodes = len(distinct_nodes)  # count distinct vertices

                    path_is_disj = (len([nid for nid in distinct_nodes if nid in touched_nodes]) == 0)
                    if path_is_disj:
                        vd += [next_path]
                        touched_nodes += distinct_nodes

            # Iteratively take symmetric differences with input_match = result
            if len(vd) > 0:
                temp = input_match
                for next_path in vd:
                    temp = [eid for eid in temp if eid not in next_path] + [eid for eid in next_path if eid not in temp]
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
            shortest_paths = []  # solution box
            found = False
            length = None
            if self.check_colored_maximal_match(b, input_match):
                # shortest_paths = []
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
                        # first_edge = next_path[0]
                        # first_node = first_edge[0]
                        last_edge = next_path[-1]

                        p = len(next_path) % 2  # path parity
                        assert p in [0, 1]
                        last_node = last_edge[p]
                        num_distinct_edges = len(list(set(next_path)))  # count distinct edges

                        # collect sequence of nodes
                        temp = []
                        if len(next_path) >= 1:
                            idx_p = None
                            for i in range(len(next_path)):
                                eid = next_path[i]
                                idx_p = i % 2  # index parity
                                temp += [eid[idx_p]]
                            temp += [next_path[-1][1 - idx_p]]

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

                            these_edges = [eid for eid in edge_set_to_search if
                                           eid not in next_path and eid[p] == last_node and eid[
                                               1 - p] not in distinct_nodes]
                            if len(these_edges) > 0:
                                for eid in these_edges:
                                    path_to_add = None
                                    path_to_add = next_path + [eid]
                                    q.append(path_to_add)
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
                    result = self._clean(b, shortest_paths, input_match)
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
            shortest_cycles_with_pos_gain = []  # solution box
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
                    if len(next_path) >= 1:
                        idx_p = None
                        for i in range(len(next_path)):
                            eid = next_path[i]
                            idx_p = i % 2  # index parity
                            temp += [eid[idx_p]]
                        temp += [next_path[-1][1-idx_p]]

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
                else:
                    # in this case, length is not None and len(next_path) > length, so we discard next_path
                    pass

            result = self._clean(b, shortest_cycles_with_pos_gain, input_match)
        return result

    def optimize(self, b):
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
