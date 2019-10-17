from collections import deque
import itertools
import sys

# Check version number
if sys.version_info[0] != 3:
    #  print("This script requires Python3")
    sys.exit(1)


class BipartiteGraph(object):
    """ Graph object representing a graph

    Attributes:
        data        : arbitrary
        count       : integer index
        left_nodes  : dict, nodes side 0, keys = vals = (int, arbitrary) pairs
        right_nodes : dict, nodes side 1, keys = vals = (int, arbitrary) pairs
        blue_edges  : dict, edges color 0, keys are tuples of the form 
            (node_ident, node_ident, arbitrary)
        red_edges   : dict, edges color 1, keys are tuples of the form 
            (node_ident, node_ident, arbitrary)

    Functions (see below for comments)
        add_node
        add_edge
        del_edge
        del_node
        chhk_colored_match
        chhk_colored_maximal_match
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
            
        self.data = par['data']
        
        if par['count'] <= len(par['left_nodes'])+len(par['right_nodes']):
            self.count = len(par['left_nodes'])+len(par['right_nodes']) + 1
        else:
            self.count = par['count']
            
        self.left_nodes, self.right_nodes, self.red_edges, self.blue_edges = \
            dict(), dict(), dict(), dict()
        self.left_nodes.update(par['left_nodes'])
        self.right_nodes.update(par['right_nodes'])
        self.red_edges.update(par['red_edges'])
        self.blue_edges.update(par['blue_edges'])

    def add_node(self, b, tag=None):
        """ Take a bit b indicating side as input (b = 0 are left nodes, b = 1 
        are right nodes) and a tag. Check that b is a bit and return None if it
        isn't. Otherwise, create a new node on side b, outputting the 
        node_ident = (self.count, tag)
        """
        result = None
        assert b in [0, 1]
        result = (self.count, tag)
        assert result not in self.right_nodes and result not in self.left_nodes
        self.count += 1
        if b:
            # right nodes are side one
            self.right_nodes.update({result: result})
        else:
            # left nodes are side zero
            self.left_nodes.update({result: result})
        return result

    def add_edge(self, b, pair, w, tag=None):
        """ Take (b, pair, w, tag) as input (a bit b indicating color, edge 
        ident pair, float w indicating weight). Fail (output False) if :
          pair[0] not in left_nodes  or
          pair[1] not in right_nodes
        Otherwise update edge with color b and edge id (pair[0], pair[1], tag) 
        to have weight w. Note: OVER-WRITES any previous edge weight of color b
        with weight w and over-writes the weight of color 1-b with weight 0.0.
        """
        # print(pair)
        (x, y) = pair  # parse eid to get left node and right node ids
        result = x in self.left_nodes
        result = result and y in self.right_nodes    
        assert b in [0, 1]
        assert w >= 0.0
        eid = (x, y, tag)
        if result and w == 0.0:
            self.del_edge(eid)
        elif result and w > 0.0:
            result = eid
            new_dict = {result: w}
            if b==1:
                self.red_edges.update(new_dict)
            else:
                self.blue_edges.update(new_dict)
        return result

    def del_edge(self, eid):
        """ Take eid = (x, y, tag) as input (an edge_ident). If both of these 
        are nodes on the proper sides, then we set edge weights in both edge 
        dictionaries to 0.0. If either node does not exist or is on the wrong 
        side, then the edge entry is deleted from the dictionary. This always 
        succeeds so we return True.
        """
        if eid in self.red_edges:
            del self.red_edges[eid]
        if eid in self.blue_edges:
            del self.blue_edges[eid]

    def del_node(self, nid):
        """  Take node identity nid as input (a node ident).
        Fail (output False) if:
          x is not in left_nodes and
          x is not in right_nodes
        Otherwise remove x from all node dictionaries and delete both color 
        edge identities like (x, y) and (y, x). This always succeeds so we 
        return True.
        """
        if nid in self.left_nodes:
            del self.left_nodes[nid]
        if nid in self.right_nodes:
            del self.right_nodes[nid]

        eids_to_remove = [eid for eid in list(self.blue_edges.keys()) \
            if (eid[0] == nid or eid[1] == nid)]
        eids_to_remove += [eid for eid in list(self.red_edges.keys()) \
            if (eid[0] == nid or eid[1] == nid)]
        for eid in eids_to_remove:
            self.del_edge(eid)
            
    def chhk_colored_match(self, b, input_match):
        """ chhk_colored_match takes a color, b, and an alleged match, 
        input_match. Produces as output a boolean.
        """
        result = None
        assert b in [0, 1]
        if len(input_match)==0:
            result = True
        else:
            correct_edge_space = not any([eid not in self.red_edges and \
                eid not in self.blue_edges for eid in input_match])
            # print("Correct edge space = " + str(correct_edge_space))
            exists_red = any([eid in self.red_edges for eid in input_match])
            exists_blue = any([eid in self.blue_edges for eid in input_match])
            exists_both = exists_red and exists_blue
            single_color = (exists_red or exists_blue) and not exists_both
            # print("single color = " + str(single_color))
            if b:
                correct_color = single_color and exists_red
            else:
                correct_color = single_color and exists_blue
            # print("correct color = " + str(correct_color))
            vertex_disjoint = not any([(eid[0]==fid[0] or eid[1] == fid[1]) \
                for eid in input_match for fid in input_match if eid != fid]) 
            # print("vertex disjoint = " + str(vertex_disjoint))
            result = single_color 
            result = result and vertex_disjoint 
            result = result and correct_edge_space 
            result = result and correct_color
            # print("result of check for colored match = " + str(result))
        return result

    def chhk_colored_maximal_match(self, b, input_match):
        assert b in [0, 1]
        result = self.chhk_colored_match(b, input_match)
        if result:
            matched_lefts = list(set([eid[0] for eid in input_match]))
            matched_rights = list(set([eid[1] for eid in input_match]))
            if b:
                wt_dct = self.red_edges
            else:
                wt_dct = self.blue_edges
            unmatched_lefts = list(set([eid[0] for eid in wt_dct \
                if eid[0] not in matched_lefts]))
            unmatched_rights = list(set([eid[1] for eid in wt_dct \
                if eid[1] not in matched_rights]))
            temp = [eid[0] in unmatched_lefts and eid[1] in unmatched_rights \
                for eid in wt_dct]
            result = not any(temp)
        return result

    def _parse(self, b, input_match):
        """ parse takes a color b and input_match and either crashes because 
        input_match isn't a match with color b or returns the following lists 
        for convenient usage in methods extend and boost.
            matched_lefts    :  left_nodes adj with an edge in input_match
            matched_rights   :  right_nodes adj with an edge in input_match
            unmatched_lefts  :  left_nodes not adj with an edge in input_match
            unmatched_rights :  right_nodes not adj with an edge in input_match
            apparent_color   :  the observed color of input_match (which should 
                match the input color b)
            non_match_edges  :  edges with color b that are not in input_match 
                excluding zero-weight edges
        """
        # Check input_match is a match of color b
        try:
            self.chhk_colored_match(b, input_match)
        except AssertionError:
            raise Exception('Input_match is not a match!')
        
        assert b in [0, 1]
        
        if b:
            wt_dct = self.red_edges
        else:
            wt_dct = self.blue_edges

        # Node sets
        matched_lefts = [eid[0] for eid in input_match]
        matched_rights = [eid[1] for eid in input_match]
        unmatched_lefts = [nid for nid in self.left_nodes \
            if nid not in matched_lefts]
        unmatched_rights = [nid for nid in self.right_nodes \
            if nid not in matched_rights]

        # non-match edges
        non_match_edges = [eid for eid in wt_dct \
            if eid not in input_match and wt_dct[eid] > 0.0]
        
        result = (matched_lefts, matched_rights, unmatched_lefts, \
            unmatched_rights, b, non_match_edges)
        return result

    @staticmethod
    def _so_fresh_so_clean(b, shortest_paths_with_gains, input_match):
        ordered_shortest_paths = sorted(shortest_paths_with_gains, \
            key=lambda z:z[1], reverse=True)
        
        tn = []
        paths_to_add = []
        
        # Greedily make shortest_paths_with_gains into a vtx-disjoint list
        for nxt_pth_and_gain in ordered_shortest_paths:
            nxt_pth, gain = nxt_pth_and_gain
             #print("Processing " + str((nxt_pth, gain)))
            touched = False
            for eid in nxt_pth:
                if eid[0] in tn or eid[1] in tn:
                    touched = True
                    break
            if not touched and gain > 0.0:
                # print("Not touched and with positive gain!")
                paths_to_add += [nxt_pth]
                # print("Collected paths = " + str(paths_to_add))
                tn += [eid[0] for eid in nxt_pth]
                tn += [eid[1] for eid in nxt_pth] 
                # print("Touched nodes = " + str(tn))
                
        
        # Next we iteratively XOR these with the input_match.
        # print("Working with paths to add: " + str(paths_to_add))
        result = input_match
        temp = result
        # print("Starting match = " + str(input_match))
        for nxt_pth in paths_to_add:
            # print("XORING with " + str(nxt_pth))
            temp = [eid for eid in result if eid not in nxt_pth]
            temp += [eid for eid in nxt_pth if eid not in result]
            result = temp
            # print("Result = " + str(result))
        return result
        

    @staticmethod
    def _clean(b, shortest_paths_with_gains, input_match):
        """ Returns the input_match (when input shortest_paths is empty) or the
        iterative symdif of input_match """
        assert b in [0, 1]
        result = input_match
        if len(shortest_paths_with_gains) > 0:
            # Order the list
            ordered_shortest_paths = sorted(shortest_paths_with_gains, \
                key=lambda z: z[1], reverse=True)
            # Construct vertex-disjoint list greedily
            vd = []
            touched_nodes = []
            for path_and_gain in ordered_shortest_paths:
                (nxt_pth, gain) = path_and_gain
                if gain > 0.0:
                    # Collect some info for convenience
                    # first_edge = nxt_pth[0]
                    # first_node = first_edge[0]
                    # last_edge = nxt_pth[-1]
                    assert len(nxt_pth) >= 1
                    p = len(nxt_pth) % 2  # path parity
                    assert p in [0, 1]
                    # lst_nd = last_edge[p]
                    # num_distinct_edges = len(list(set(nxt_pth)))  
                    # collect sequence of nodes
                    temp = []
                    if len(nxt_pth) >= 1:
                        idx_p = None
                        for i in range(len(nxt_pth)):
                            eid = nxt_pth[i]
                            idx_p = i % 2  # index parity
                            temp += [eid[idx_p]]
                        temp += [nxt_pth[-1][1 - idx_p]]
                    dst_nds = list(set(temp))

                    path_is_disj = (len([nid for nid in dst_nds if nid \
                        in touched_nodes]) == 0)
                    if path_is_disj:
                        vd += [nxt_pth]
                        touched_nodes += dst_nds
            #  print("NEXT PATH = ", nxt_pth)
            # Iteratively take symmetric differences with input_match = result
            if len(vd) > 0:
                temp = input_match
                for nxt_pth in vd:
                    temp = [eid for eid in temp if eid not in nxt_pth]
                    temp += [eid for eid in nxt_pth if eid not in temp]
                result = temp
        return result

    def xxtend(self, b, input_match=[]):
        """ Find all shortest paths P satisfying the following constraints and 
        call _cleanup with them.
              (i)   all edges in P share the same color with input_match and
              (ii)  edges in P alternate with respect to input_match and
              (iii) the initial endpoint of P is an unmatched node and
              (iv)  P cannot be extended by any correctly colored edges alt-
                    ernating with respect to input_match without self-
                    intersecting.
        """
        result = None
        if not self.chhk_colored_match(b, input_match):
            s = 'Tried to extend an input_match that is not a unicolor match.'
            raise Exception(s)
        else:
            if self.chhk_colored_maximal_match(b, input_match):
                result = input_match
            else:       
                found = False
                lng = 0     
                soln_box = []
                #  print("Setting wt_dct.")
                if b:
                    wt_dct = self.red_edges
                else:
                    wt_dct = self.blue_edges
                #  #  print("Parsing (b, input_match):")
                temp = self._parse(b, input_match)
                (matched_lefts, matched_rights, unmatched_lefts, \
                    unmatched_rights, bb, non_match_edges) = temp

                q = deque([[eid] for eid in non_match_edges \
                    if eid[0] in unmatched_lefts])
                while len(q) > 0:
                    nxt_pth = q.popleft()
                    s = not found and lng == 0
                    s = s or (found and len(nxt_pth) <= lng)
                    if s:
                        nxt_wt = sum([wt_dct[eid] for eid in nxt_pth \
                            if eid not in input_match])
                        nxt_wt += sum([wt_dct[eid] for eid in input_match \
                            if eid not in nxt_pth])
                        gn = nxt_wt - sum([wt_dct[eid] for eid in input_match])
                        endpts = [eid[0] for eid in nxt_pth]
                        endpts += [eid[1] for eid in nxt_pth]
                        tn = set(endpts )
                        p = len(nxt_pth) % 2
                        last_edge = nxt_pth[-1]
                        lst_nd = last_edge[p]
                        if gain > 0.0 and lst_nd in unmatched_rights:
                            if not found:
                                assert lng == 0
                                found = True
                                lng = len(nxt_pth)
                            else:
                                assert 0 < len(nxt_pth) <= lng
                            soln_box += [(nxt_pth, gain)]
                            #  #  print("soln box:")
                            #  #  print(soln_box)
                        elif lst_nd not in unmatched_rights:
                            for eid in wt_dct:
                                if eid[p] == lst_nd and eid[1-p] not in tn:
                                    q.append(nxt_pth + [eid])
                #  #  print("Sorting soln box")
                soln_box = sorted(soln_box, key=lambda x:x[1], reverse=True)
                vd_solns = []
                tn = []
                for path_gain_pair in soln_box:
                    (nxt_pth, gain) = path_gain_pair
                    s = [eid[0] in tn for eid in nxt_pth]
                    s += [eid[1] in tn for eid in nxt_pth]
                    if not any(s):
                        vd_solns += [nxt_pth]
                        for eid in nxt_pth:
                            tn = list(set(tn + [eid[0], eid[1]]))

                result = input_match
                for soln in vd_solns:
                    temp = [eid for eid in soln if eid not in result]
                    temp += [eid for eid in result if eid not in soln]
                    result = temp
        return result
        
    def extend(self, b, input_match=[]):
        # #  #  print("ENTERING EXTEND, HERE IS INPUT MATCH", input_match)
        assert isinstance(input_match, list)
        try:
            assert input_match is not None
        except AssertionError:
            raise Exception('Input_match was not a list!')

        result = None

        assert b in [0, 1]
        if b:
            wt_dct = self.red_edges
        else:
            wt_dct = self.blue_edges

        if len(input_match) > 0:
            for eid in input_match:
                assert eid in wt_dct
        
        if self.chhk_colored_match(b, input_match):
            #  #  print("Found that ", input_match, " is a colored match.")
            shortest_paths = []  # solution box
            found = False
            lng = None
            if self.chhk_colored_maximal_match(b, input_match):
                result = input_match
            else:
                temp = self._parse(b, input_match)
                (matched_lefts, matched_rights, unmatched_lefts, \
                    unmatched_rights, bb, non_match_edges) = temp
                assert bb == b
                q = deque([[eid] for eid in wt_dct if \
                    (eid not in input_match and eid[0] in unmatched_lefts)])
                assert len(q) > 0
                while len(q) > 0:
                    assert not found or \
                        (lng is not None and len(shortest_paths) > 0)
                    nxt_pth = q.popleft()
                    if lng is None or \
                        (lng is not None and len(nxt_pth) <= lng):
                        last_edge = nxt_pth[-1]
                        p = len(nxt_pth) % 2  # path parity
                        assert p in [0, 1]
                        assert p in [0, 1]
                        lst_nd = last_edge[p]
                        num_distinct_edges = len(list(set(nxt_pth)))  
                        temp = []
                        if len(nxt_pth) >= 1:
                            idx_p = None
                            for i in range(len(nxt_pth)):
                                eid = nxt_pth[i]
                                idx_p = i % 2  # index parity
                                temp += [eid[idx_p]]
                            temp += [nxt_pth[-1][1 - idx_p]]

                        dst_nds = list(set(temp))
                        num_dst_nds = len(dst_nds)

                        if len(nxt_pth) == num_distinct_edges and \
                            len(temp) == num_dst_nds:
                            # Extend the path with allowable alternating edges,
                            # placing the extended paths back into the queue. 
                            # If the path cannot be extended, compute the gain 
                            # and if positive place in solution box.
                            if lst_nd in unmatched_rights or \
                                lst_nd in unmatched_lefts:
                                sd = 0.0                                
                                sd = [wt_dct[z] for z in wt_dct if \
                                    (z in input_match and z not in nxt_pth)]
                                sd += [wt_dct[z] for z in wt_dct if \
                                    (z not in input_match and z in nxt_pth)]

                                # I like fitting my code in 80 lines :D
                                gain = sum([wt_dct[x] for x in input_match])
                                gain -= sum(sd)
                                gain = -gain
                                # print("Path has gain ", gain)
                                if gain > 0.0:
                                    if lng is None or len(nxt_pth) < lng:
                                        shortest_paths = [(nxt_pth, gain)]
                                        lng = len(nxt_pth)
                                        found = True
                                    elif lng is not None and \
                                        len(nxt_pth) == lng:
                                        shortest_paths += [(nxt_pth, gain)]
                            else:
                                ed_set = None
                                if p:
                                    ed_set = input_match
                                else:
                                    ed_set = non_match_edges
                                assert ed_set is not None
                                these_edges = [eid for eid in ed_set if \
                                    eid not in nxt_pth and eid[p] == lst_nd \
                                    and eid[1 - p] not in dst_nds]
                                assert len(these_edges) > 0 
                                if len(these_edges) > 0:
                                    for eid in these_edges:
                                        path_to_add = None
                                        path_to_add = nxt_pth + [eid]
                                        q.append(path_to_add)
                # Check that found_shortest_path = True if and only if at 
                # least one shortest path is in shortest_paths
                assert not found or len(shortest_paths) > 0
                assert len(shortest_paths) == 0 or found
                # print("FOUND SHORTEST PATHS = " + str(shortest_paths))
                if len(shortest_paths) > 0:
                    # #  print(shortest_paths)
                    result = self._so_fresh_so_clean(b, shortest_paths, \
                        input_match)
                else:
                    result = input_match
        else:
            s = 'Ooops, check colored match didnt think input_match is a match of a single color.'
            raise Exception(s)
        return result

    def boost(self, b, input_match):
        """ Note : maximality of input_match implies no *augmenting* paths 
        exist, i.e. alternating and beginning and ending with unmatched nodes 
        (this is a theorem, that augmenting paths exist IFF the match is max-
        imal).  However, there exist alternating cycles, and the optimal match 
        can be written as the symmetric difference between the input match and 
        a sequence of vertex-disjoint alternating cycles with positive gain.
        """
        result = None
        assert b in [0, 1]
        assert self.chhk_colored_maximal_match(b, input_match)
        
        if b:
            wt_dct = self.red_edges
        else:
            wt_dct = self.blue_edges
            
        shrt_cyc_pos = []  # solution box
        nds_shrt_cyc = []
        found = False
        lng = None

        (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, \
            bb, non_match_edges) = self._parse(b, input_match)

        q = deque([[eid] for eid in non_match_edges])
        while len(q) > 0:
            assert not found or lng is not None

            nxt_pth = q.popleft()  # Get next path
            
            if lng is None or (lng is not None and len(nxt_pth) <= lng):
                # Paths have distinct edges and distinct vertices. Cycles are 
                # like paths, but the first and last vertices are allowed to 
                # match. Certain paths are degenerate in that they lead into a 
                # cycle, looping back on themselves, leading to a union of a 
                # path and a cycle. These are not alternating cycles and are 
                # discarded. The embedded cycle will be found earlier in the 
                # algorithm anyway.

                # Collect some info for convenience
                first_edge = nxt_pth[0]
                first_node = first_edge[0]
                last_edge = nxt_pth[-1]

                p = len(nxt_pth) % 2  # path parity
                assert p in [0, 1]
                lst_nd = last_edge[p]
                num_distinct_edges = len(list(set(nxt_pth)))

                # collect sequence of nodes
                temp = []
                if len(nxt_pth) >= 1:
                    idx_p = None
                    for i in range(len(nxt_pth)):
                        eid = nxt_pth[i]
                        idx_p = i % 2  # index parity
                        temp += [eid[idx_p]]
                    temp += [nxt_pth[-1][1-idx_p]]

                num_dst_nds = len(list(set(temp)))  # count distinct vertices
                
                assert len(nxt_pth) == num_distinct_edges 
                assert len(temp) - num_dst_nds in [0, 1]
                # Paths have distinct edges. Paths of N distinct edges have N+1
                # vertices, except when the path is degenerate or a cycle in 
                # which case they have N vertices.
                if len(temp) - num_dst_nds == 1:
                    # such a path is either a cycle or is degenerate.
                    if first_node == lst_nd:
                        # this path is a cycle. If it has positive gain, put it
                        # into the solution box and update nds_shrt_cyc
                        sd = [wt_dct[eid] for eid in nxt_pth if \
                            eid not in input_match]
                        sd += [wt_dct[eid] for eid in input_match if \
                            eid not in nxt_pth]
                        gain = sum([wt_dct[eid] for eid in input_match])
                        gain -= sum(sd)
                        gain = -gain
                        if gain > 0.0:
                            found = True
                            if lng is None or len(nxt_pth) < lng:
                                lng = len(nxt_pth)
                                shrt_cyc_pos = []
                                shrt_cyc_pos += [(nxt_pth, gain)]
                                for eid in nxt_pth:
                                    nds_shrt_cyc += [eid[0], eid[1]]
                            elif len(nxt_pth) == lng:
                                shrt_cyc_pos += [(nxt_pth, gain)]
                                for eid in nxt_pth:
                                    nds_shrt_cyc += [eid[0], eid[1]]
                            else:
                                # In this case, from the first if statement, 
                                # lng is not None (so a shortest cycle has been
                                # found) and len(nxt_pth) >= lng, but, from the
                                # second if stment, also len(nxt_pth) != lng, 
                                # so len(nxt_pth) > lng necessarily. This cycle
                                # is too long, discard it.
                                pass
                    else:
                        # this path is degenerate and is discarded
                        pass
                else:
                    # this path is not a cycle and is made of all distinct 
                    # vertices. Extend the path with allowable alternating 
                    # edges, placing the extended paths back into the queue.
                    if p:
                        ed_set = input_match
                    else:
                        ed_set = non_match_edges

                    for eid in ed_set:
                        if eid not in nxt_pth and eid[p] == lst_nd:
                            path_to_add = None
                            path_to_add = nxt_pth + [eid]
                            q.append(path_to_add)
            else:
                # in this case, lng is not None and len(nxt_pth) > lng, so we 
                # discard nxt_pth
                pass
        result = self._so_fresh_so_clean(b, shrt_cyc_pos, input_match)
        return result

    def optimize(self, b):
        ''' Finds a maximal (but not maximum) matching and then optimizes it. 
        These are sub-optimal matchings unless the match is not only maximal 
        but maximum.
        '''
        # TODO: Fix functions leading to this function so as to only result in 
        # *maximum* matchings.
        assert b in [0, 1]
        next_match = self.xxtend(b) # starts with empty match by default
        assert next_match is not None
        temp = None
        while next_match != temp and next_match is not None:
            temp = next_match
            next_match = self.xxtend(b, temp)
        if b:
            wt_dct = self.red_edges
        else:
            wt_dct = self.blue_edges
        assert len(wt_dct) == 0 or \
            (next_match is not None and len(next_match) > 0)

        temp = next_match
        w = sum([wt_dct[eid] for eid in temp])
        next_match = self.boost(b, temp)
        assert next_match is not None
        v = sum([wt_dct[eid] for eid in next_match])
        assert len(temp) == len(next_match)
        while v - w > 0.0:
            temp = next_match
            w = v
            next_match = self.boost(b, input_match)
            v = sum([wt_dct[eid] for eid in next_match])
            assert len(temp) == len(next_match)
            
        return next_match
