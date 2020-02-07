import unittest as ut
import random
from graphtheory import *
from copy import deepcopy


def make_graph(num_left, num_right):
    """ make_graph makes a new graph with num_left left_nodes and num_right right_nodes and no edges. """
    g = BipartiteGraph(None)
    for i in range(num_left):
        g.add_node(0)
    for i in range(num_right):
        g.add_node(1)
    return g


def apply_deterministic_weight(g, n=1, m=5):
    """
    apply_deterministic_weight

    Take as input a graph g, and a pair of integers n, m.

    For left_node i and right_node j, the edge (i, j) is colored
    with color (i*j) % 2 and weighted with 20*(i-n) + 4*(j-m).
    """
    min_weight = 0
    for xid in sorted(list(g.left_nodes.keys())):
        for yid in sorted(list(g.right_nodes.keys())):
            wt = 20.0 * (xid[0] - n) + 4.0 * (yid[0] - m)
            assert wt > 0
            g.add_edge((xid[0] * yid[0]) % 2, (xid, yid), wt)
    return g


def make_deterministic_graph_for_integration_test():
    """
    make_deterministic_graph_for_integration_test

    Produce a BipartiteGraph with 10 left_nodes and 10 right_nodes, then weight with apply_deterministic_weight.
    """
    # This generates a graph deterministically.
    g = make_graph(10, 10)
    return apply_deterministic_weight(g, 1, 9)


def make_r_graph():
    """ make_r_graph produces a random graph. """
    num_left_nodes = random.randint(3, 5)
    num_right_nodes = random.randint(3, 5)
    g = make_graph(num_left_nodes, num_right_nodes)
    # We assign random edge weights
    for xid in g.left_nodes:
        for yid in g.right_nodes:
            g.add_edge(random.getrandbits(1), (xid, yid), random.random())
    return g


def make_rr_graph():
    """ make_r_graph produces a random graph. """
    g = make_graph(3, 3)
    g = apply_deterministic_weight(g, 1, 3)
    return g


def helper():
    """ helper is a helper function that outputs a random graph with make_rr_graph with some additional info. """
    g = make_rr_graph()

    reds_incident_with_one = [eid for eid in g.red_edges if
                              (1, None) in eid and g.red_edges[eid] > 0.0]
    reds_incident_with_two = [eid for eid in g.red_edges if
                              (2, None) in eid and g.red_edges[eid] > 0.0]
    reds_incident_with_three = [eid for eid in g.red_edges if
                                (3, None) in eid and g.red_edges[eid] > 0.0]

    reds_inc = reds_incident_with_one
    reds_inc += reds_incident_with_two
    reds_inc += reds_incident_with_three

    blues_incident_with_one = [eid for eid in g.blue_edges if
                               (1, None) in eid and g.blue_edges[eid] > 0.0]
    blues_incident_with_two = [eid for eid in g.blue_edges if
                               (2, None) in eid and g.blue_edges[eid] > 0.0]
    blues_incident_with_three = [eid for eid in g.blue_edges if
                                 (3, None) in eid and g.blue_edges[
                                     eid] > 0.0]

    blues_inc = blues_incident_with_one
    blues_inc += blues_incident_with_two
    blues_inc += blues_incident_with_three

    result = []
    result += [g]
    result += [reds_inc]
    result += [reds_incident_with_one]
    result += [reds_incident_with_two]
    result += [reds_incident_with_three]
    result += [blues_inc]
    result += [blues_incident_with_one]
    result += [blues_incident_with_two]
    result += [blues_incident_with_three]
    return result


def helper_maximal_matchiness():
    """ helper_maximal_matchiness generates random graphs until the result has edges of both colors. """
    [g, r, riwo, riwt, riwtt, b, biwo, biwt, biwtt] = helper()

    while len(r) == 0 or len(b) == 0:
        [g, r, riwo, riwt, riwtt, b, biwo, biwt, biwtt] = helper()

    return [g, r, riwo, riwt, riwtt, b, biwo, biwt, biwtt]


def make_d_graph():
    """ Make a BipartiteGraph with 5 nodes on each side and with deterministic weighting. """
    # This generates a graph deterministically.
    g = make_graph(5, 5)
    g = apply_deterministic_weight(g, 1, 5)
    
    return g


def make_dd_graph():
    """ Generates a deterministic graph with a slightly different weighting. """
    g = make_d_graph()
    g.red_edges[((1, None), (7, None), None)] -= 1.0
    g.red_edges[((1, None), (9, None), None)] += 1.0
    g.red_edges[((3, None), (7, None), None)] += 1.0
    g.red_edges[((3, None), (9, None), None)] -= 1.0

    return g


def do_a_clean_thing(b, g, input_match, sp):
    """ Tests the clean helper function. """
    assert b in [0, 1]
    # print("sp = " + str(sp))
    # print("input match = " + str(input_match))
    shortest_paths = []
    if b:
        wt_dict = g.red_edges
    else:
        wt_dict = g.blue_edges
    assert len(wt_dict) != 0
    # print("wt dict we are working with = ")
    # for key in list(wt_dict.keys()):
    #     print("\t" + str(key) + "\t\t" + str(wt_dict[key]))
    for eid in input_match:
        # print("eid in input match = " + str(eid))
        assert eid in wt_dict
    for nxt_pth in sp:
        for eid in nxt_pth:
            # print("eid in nxt_pth = " + str(eid))
            assert eid in wt_dict
        sd = [eid for eid in nxt_pth if eid not in input_match]
        sd += [eid for eid in input_match if eid not in nxt_pth]
        gain = sum(wt_dict[eid] for eid in sd)
        gain = gain - sum(wt_dict[eid] for eid in input_match)
        shortest_paths += [(nxt_pth, gain)]

    return g.clean(b, shortest_paths, input_match)


class TestBipartiteGraph(ut.TestCase):
    """ TestBipartiteGraph tests BipartiteGraph objects """
    
    @ut.skip("Skipping test_d_init")
    def test_d_init(self):
        """ test_d_init deterministically tests initialization of a graph. """
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)] , 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)] , 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)] , 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)] , 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)] , 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)] , 96.0)

        self.assertTrue(isinstance(g, BipartiteGraph))
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)
        wt = 4.0
        for xid in sorted(list(g.left_nodes.keys())):
            for yid in sorted(list(g.right_nodes.keys())):
                self.assertTrue((xid, yid, None) in g.red_edges or
                                (xid, yid, None) in g.blue_edges)
                if xid[0]*yid[0] % 2 == 1:
                    self.assertTrue((xid, yid, None) in g.red_edges and
                                    g.red_edges[(xid, yid, None)] == wt)
                elif xid[0]*yid[0] % 2 == 0:
                    self.assertTrue((xid, yid, None) in g.blue_edges and
                                    g.blue_edges[(xid, yid, None)] == wt)
                wt += 4.0

    @ut.skip("Skipping test_d_init_by_hand")
    def test_d_init_by_hand(self):
        """ test_d_init_by_hand deterministically tests init of a graph. """
        g = BipartiteGraph()
        self.assertTrue(isinstance(g, BipartiteGraph))
        self.assertEqual(len(g.left_nodes), 0)
        self.assertEqual(len(g.right_nodes), 0)
        self.assertEqual(len(g.blue_edges), 0)
        self.assertEqual(len(g.red_edges), 0)
        self.assertEqual(g.count, 1)
        self.assertEqual(g.data, 'han-tyumi')

        par = dict()
        par['data'] = 'han-tyumi'
        par['count'] = 12
        par['left_nodes'] = {0: 0, 1: 1, 2: 2}
        par['right_nodes'] = {3: 3, 4: 4, 5: 5}
        par['blue_edges'] = {(0, 3): 100, (1, 4): 101, (1, 5): 99}
        par['blue_edges'].update({(2, 5): 75, (2, 3): 1000})
        par['red_edges'] = {(0, 4): 1, (0, 5): 8, (1, 3): 72, (2, 4): 1}
        g = BipartiteGraph(par)

        self.assertTrue(isinstance(g, BipartiteGraph))

        self.assertEqual(g.data, 'han-tyumi')

        self.assertEqual(g.count, 12)

        self.assertTrue(0 in g.left_nodes and g.left_nodes[0] == 0)
        self.assertTrue(1 in g.left_nodes and g.left_nodes[1] == 1)
        self.assertTrue(2 in g.left_nodes and g.left_nodes[2] == 2)
        self.assertEqual(len(g.left_nodes), 3)

        self.assertTrue(3 in g.right_nodes and g.right_nodes[3] == 3)
        self.assertTrue(4 in g.right_nodes and g.right_nodes[4] == 4)
        self.assertTrue(5 in g.right_nodes and g.right_nodes[5] == 5)
        self.assertEqual(len(g.right_nodes), 3)

        self.assertTrue((0, 3) in g.blue_edges and g.blue_edges[(0, 3)] == 100)
        self.assertTrue((1, 4) in g.blue_edges and g.blue_edges[(1, 4)] == 101)
        self.assertTrue((1, 5) in g.blue_edges and g.blue_edges[(1, 5)] == 99)
        self.assertTrue((2, 5) in g.blue_edges and g.blue_edges[(2, 5)] == 75)
        self.assertTrue((2, 3) in g.blue_edges)
        self.assertTrue(g.blue_edges[(2, 3)] == 1000)
        self.assertEqual(len(g.blue_edges), 5)

        self.assertTrue((0, 4) in g.red_edges and g.red_edges[(0, 4)] == 1)
        self.assertTrue((0, 5) in g.red_edges and g.red_edges[(0, 5)] == 8)
        self.assertTrue((1, 3) in g.red_edges and g.red_edges[(1, 3)] == 72)
        self.assertTrue((2, 4) in g.red_edges and g.red_edges[(2, 4)] == 1)
        self.assertEqual(len(g.red_edges), 4)

    @ut.skip("Skipping test_r_init")
    def test_r_init(self):
        """ test_r_init randomly tests initialization of a graph. """
        g = make_r_graph()

        self.assertTrue(isinstance(g, BipartiteGraph))

        self.assertEqual(g.data, 'han-tyumi')

        n = len(g.right_nodes) + len(g.left_nodes)
        self.assertEqual(g.count-1, n)

        self.assertTrue(3 <= len(g.left_nodes) <= 5)
        for i in range(1, len(g.left_nodes)+1):
            self.assertTrue((i, None) in g.left_nodes)
            self.assertEqual(g.left_nodes[(i, None)], (i, None))

        self.assertTrue(3 <= len(g.right_nodes) <= 5)
        for i in range(len(g.left_nodes)+1, len(g.right_nodes)+1):
            self.assertTrue((i, None) in g.right_nodes)
            self.assertEqual(g.right_nodes[(i, None)], (i, None))

        n = len([x for x in g.right_nodes if x in g.left_nodes])
        self.assertEqual(n, 0)

        n = len(g.right_nodes)*len(g.left_nodes)
        m = len(g.red_edges) + len(g.blue_edges)

        self.assertEqual(n, m)

        n = len([eid for eid in g.red_edges if eid in g.blue_edges])
        self.assertEqual(n, 0)

        n = len([eid for eid in g.blue_edges if eid in g.red_edges])
        self.assertEqual(n, 0)

        for eid in g.red_edges:
            self.assertTrue(g.red_edges[eid] >= 0.0)
        for eid in g.blue_edges:
            self.assertTrue(g.blue_edges[eid] >= 0.0)

        for x in g.left_nodes:
            for y in g.right_nodes:
                s = (x, y, None) in g.red_edges
                t = (x, y, None) in g.blue_edges
                self.assertTrue((s or t) and not (s and t))

    def check_adding_nodes(self, b, g):
        """ Checks that adding nodes works appropriately. """
        self.assertIn(b, [0, 1])
        n = deepcopy(g.count - 1)
        nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
        nnn = deepcopy(g.data)
        j = g.add_node(b)  # create a new node
        if b:
            self.assertTrue(j in g.right_nodes)
            y = g.right_nodes[j] == j
        elif not b:
            self.assertTrue(j in g.left_nodes)
            y = g.left_nodes[j] == j

        self.assertTrue(y)
        m = deepcopy(g.count - 1)
        mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
        mmm = deepcopy(g.data)
        self.assertEqual(m - n, 1)  # we only added one node
        self.assertEqual(mm, nn)  # We added no edges
        self.assertEqual(mmm, nnn)  # arbitrary data preserved
        self.assertEqual(g.count - 1, len(g.left_nodes) + len(g.right_nodes))
        return g

    @ut.skip("Skipping test_d_add_node")
    def test_d_add_node(self):
        """ test_d_add_node deterministically tests adding a node to a graph."""
        g = BipartiteGraph()
        for i in range(100):
            g = self.check_adding_nodes(0, g)
        for i in range(100):
            g = self.check_adding_nodes(1, g)

    @ut.skip("Skipping test_r_add_node")
    def test_r_add_node(self):
        """ test_r_add_node randomly tests adding a node to a graph."""
        g = make_r_graph()
        # n = g.count - 1
        num_to_add = random.randint(50, 200)
        for i in range(num_to_add):
            b = random.getrandbits(1)
            g = self.check_adding_nodes(b, g)

    @ut.skip("Skipping test_d_add_edge")
    def test_d_add_edge(self):
        """ test_d_add_edge deterministically tests adding an edge."""
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        self.assertTrue(len(g.left_nodes), 5)
        self.assertTrue(len(g.right_nodes), 5)
        self.assertTrue(len(g.blue_edges), 18)
        self.assertTrue(len(g.red_edges), 6)

        j = g.add_node(1)  # add a new right node with no new edges
        self.assertTrue(len(g.left_nodes), 5)
        self.assertTrue(len(g.right_nodes), 6)
        self.assertEqual(j, (11, None))
        self.assertTrue(len(g.blue_edges), 18)
        self.assertTrue(len(g.red_edges), 6)

        ii = g.add_node(0)  # add a new left node with no new edges
        self.assertTrue(len(g.left_nodes), 6)
        self.assertTrue(len(g.right_nodes), 6)
        self.assertEqual(ii, (12, None))
        self.assertTrue(len(g.blue_edges), 18)
        self.assertTrue(len(g.red_edges), 6)

        imin = min(list(g.left_nodes.keys()))
        imax = max(list(g.left_nodes.keys()))
        self.assertEqual(imin, (1, None))
        self.assertEqual(imax, (12, None))
        jmin = min(list(g.right_nodes.keys()))
        jmax = max(list(g.right_nodes.keys()))
        self.assertEqual(jmin, (6, None))
        self.assertEqual(jmax, (11, None))

        # weights will start at 0.1 and be incremented by 0.1
        w = 0.1
        # Reset weight of the blue edge (1, 11) to 0.1
        g.add_edge(0, (imin, jmax), w)
        w += 0.1

        self.assertTrue(len(g.left_nodes), 6)
        self.assertTrue(len(g.right_nodes), 6)
        self.assertTrue(len(g.blue_edges), 19)
        self.assertTrue(len(g.red_edges), 6)

        g.add_edge(1, (imax, jmin), w)  # edge (12, 6)
        w += 0.1

        self.assertTrue(len(g.left_nodes), 6)
        self.assertTrue(len(g.right_nodes), 6)
        self.assertTrue(len(g.blue_edges), 19)
        self.assertTrue(len(g.red_edges), 7)

    def check_adding_edges(self, b, g, eid, w):
        """ Checks that adding a single edge does exactly what it's supposed to. """
        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))

        g.add_edge(b, eid, w)  # add a single edge
        nn = deepcopy(len(g.blue_edges))
        mm = deepcopy(len(g.red_edges))
        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        self.assertEqual(nn, n + 1-b)
        self.assertEqual(mm, m + b)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)
        return g

    @ut.skip("Skipping test_r_add_edge")
    def test_r_add_edge(self):
        """ test_r_add_edge randomly tests adding an edge."""
        g = make_r_graph()
        n = len(g.red_edges) + len(g.blue_edges)
        m = len(g.left_nodes)*len(g.right_nodes)
        self.assertEqual(n, m)

        n = g.count - 1  # get the count
        j = g.add_node(1)  # add a node to the right
        self.assertEqual(j, (n+1, None))  # check node identity correct

        ii = g.add_node(0)  # add a node to the left
        self.assertEqual(ii, (n+2, None))  # check node identity correct

        i = min(list(g.left_nodes.keys()))  # get smallest left node key
        self.assertEqual(i, (1, None))  # check it's 1

        jj = min(list(g.right_nodes.keys()))  # get the smallest right node key
        # check jj is where make_r_graph switched to add nodes to the right
        self.assertEqual(jj, (len(g.left_nodes), None))

        w = random.random()
        g = self.check_adding_edges(0, g, (ii, jj), w)
        w += 0.1
        self.check_adding_edges(1, g, (ii, j), w)

    @ut.skip("Skipping test_d_del_edge")
    def test_d_del_edge(self):
        """ test_d_del_edge deterministically tests deletion of edges."""
        # we will delete edge (4, 9)
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        # p = (4*9) % 2
        self.assertTrue(((4, None), (9, None), None) in g.blue_edges)
        self.assertTrue(g.blue_edges[((4, None), (9, None), None)] > 0.0)

        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        n = deepcopy(len(g.red_edges))
        m = deepcopy(len(g.blue_edges))

        g.del_edge(((4, None), (9, None), None))
        self.assertTrue((4, None) in g.left_nodes)
        self.assertTrue((9, None) in g.right_nodes)
        self.assertFalse(((4, None), (9, None), None) in g.red_edges or
                         ((4, None), (9, None), None) in g.blue_edges)

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        nn = deepcopy(len(g.red_edges))
        mm = deepcopy(len(g.blue_edges))

        self.assertEqual(n, nn)
        self.assertEqual(m, mm+1)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)

    @ut.skip("Skipping test_r_del_edge")
    def test_r_del_edge(self):
        """ test_r_del_edge randomly tests deleting an edge. """
        g = make_r_graph()
        while len(g.red_edges) == 0 or len(g.blue_edges) == 0:
            g = make_r_graph()

        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))

        b = random.getrandbits(1)
        self.assertTrue(b in [0, 1])
        if b:
            eid = random.choice(list(g.red_edges.keys()))
            n = deepcopy(len(g.red_edges))
            m = deepcopy(len(g.blue_edges))
        else:
            eid = random.choice(list(g.blue_edges.keys()))
            n = deepcopy(len(g.red_edges))
            m = deepcopy(len(g.blue_edges))

        g.del_edge(eid)
        self.assertTrue(eid not in g.red_edges and eid not in g.blue_edges)
        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        if b:
            nn = deepcopy(len(g.red_edges))
            mm = deepcopy(len(g.blue_edges))
        else:
            nn = deepcopy(len(g.red_edges))
            mm = deepcopy(len(g.blue_edges))

        self.assertEqual(n, nn+b)
        self.assertEqual(m, mm+(1-b))
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)

    @ut.skip("Skipping test_d_del_node")
    def test_d_del_node(self):
        """ test_d_del_node determinsitically tests deletion of nodes."""
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        n = deepcopy(len(g.red_edges))
        m = deepcopy(len(g.blue_edges))

        x = (3, None)  # this is a left node with 5 incident edges
        self.assertTrue(x in g.left_nodes)
        g.del_node(x)
        self.assertFalse(x in g.left_nodes)
        for eid in g.red_edges:
            self.assertFalse(x == eid[0] or x == eid[1])
        for eid in g.blue_edges:
            self.assertFalse(x == eid[0] or x == eid[1])
        # when x = 3 is deleted from the left, the following edges are del'd
        #     ((3, None), (6, None), None),
        #     ((3, None), (7, None), None),
        #     ((3, None), (8, None), None),
        #     ((3, None), (9, None), None), and
        #     ((3, None), (10, None), None)
        # Colors, look:
        #     (3*6) % 2 = 0,
        #     (3*7) % 2 = 1,
        #     (3*8) % 2 = 0,
        #     (3*9) % 2 = 1, and
        #     (3*10) % 2 = 0
        # So blue edges:
        #     ((3, None), (6, None), None),
        #     ((3, None), (8, None), None), and
        #     ((3, None), (10, None), None)
        # and red edges:
        #    ((3, None), (7, None), None),
        #    ((3, None), (9, None), None)

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        nn = deepcopy(len(g.red_edges))
        mm = deepcopy(len(g.blue_edges))

        self.assertEqual(a-aa, 1)
        self.assertEqual(c-cc, 0)
        self.assertEqual(n-nn, 2)
        self.assertEqual(m-mm, 3)

    @ut.skip("Skipping test_r_del_node")
    def test_r_del_node(self):
        """ test_r_del_node randomly tests deletion of nodes. """
        g = make_r_graph()

        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        n = deepcopy(len(g.red_edges))
        m = deepcopy(len(g.blue_edges))

        # pick a random node:
        b = random.getrandbits(1)
        if b:
            x = random.choice(list(g.right_nodes.keys()))
        else:
            x = random.choice(list(g.left_nodes.keys()))
        red_eids_incident_with_x = [eid for eid in g.red_edges if x in eid]
        blue_eids_incident_with_x = [eid for eid in g.blue_edges if x in eid]
        g.del_node(x)

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        nn = deepcopy(len(g.red_edges))
        mm = deepcopy(len(g.blue_edges))

        if b:
            self.assertEqual(a, aa)
            self.assertEqual(c-cc, 1)
        else:
            self.assertEqual(a-aa, 1)
            self.assertEqual(c, cc)
        self.assertEqual(n-nn, len(red_eids_incident_with_x))
        self.assertEqual(m-mm, len(blue_eids_incident_with_x))

    @ut.skip("Skipping test_d_chk_colored_match")
    # noinspection DuplicatedCode
    def test_d_chk_colored_match(self):
        """ test_d_chk_colored_match tests chk_colored_match dtrmnstcly"""
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        # a maximal red matching. there are 9 of these in total.
        b = 1
        match = [((3, None), (7, None), None), ((5, None), (9, None), None)]
        self.assertTrue(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1-b, match))

        # a maximal blue matching. there are 25 of these in total.
        b = 0
        match = [((2, None), (8, None), None), ((4, None), (6, None), None)]
        self.assertTrue(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1-b, match))

        # a non-maximal red matching. there are 9 of these in total.
        b = 1
        match = [((1, None), (7, None), None)]
        self.assertTrue(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1 - b, match))

        # a non-maximal blue matching. there are 25 of these in total.
        b = 0
        match = [((4, None), (10, None), None)]
        self.assertTrue(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1 - b, match))

        # a trivial matching.
        b = 1
        match = []
        self.assertTrue(g.chk_colored_match(b, match))
        self.assertTrue(g.chk_colored_match(1-b, match))

        # a non-vertex-disjoint set of red edges cannot be a match.
        b = 1
        match = [((1, None), (7, None), None), ((3, None), (7, None), None)]
        self.assertFalse(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1 - b, match))

        # a non-vertex-disjoint set of blue edges cannot be a match.
        b = 0
        match = [((4, None), (10, None), None), ((4, None), (8, None), None)]
        self.assertFalse(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1 - b, match))

        # a multi-colored set of edges cannot be a match for either color.
        b = 0
        match = [((4, None), (10, None), None), ((5, None), (7, None), None)]
        self.assertFalse(g.chk_colored_match(b, match))
        self.assertFalse(g.chk_colored_match(1 - b, match))

    @ut.skip("Skipping test_r_chk_colored_match")
    def test_r_chk_colored_match(self):
        """ test_r_chk_colored_match randomly tests chk_colored_match"""
        g = make_r_graph()

        # Pick a random color
        b = random.getrandbits(1)
        if b:
            wt_dict = g.red_edges
            other_wt_dict = g.blue_edges
        else:
            wt_dict = g.blue_edges
            other_wt_dict = g.red_edges

        # Empty matches are matches.
        match = []
        self.assertTrue(g.chk_colored_match(b, match))

        # Pick 2 random edge idents from that color edge dict, add to match
        eid = random.choice(list(wt_dict.keys()))
        wt_dict[eid] = random.random()
        match += [eid]

        eid = random.choice(list(wt_dict.keys()))
        while eid in match:
            eid = random.choice(list(wt_dict.keys()))
        self.assertTrue(eid not in other_wt_dict or other_wt_dict[eid] == 0.0)
        wt_dict[eid] = random.random()
        match += [eid]

        # If these edges are not vtx-disjoint with each other or input_match,
        # then adding these edges violates the match condition. Otherwise, not.

        if any([eidd[0] == eiddd[0] or eidd[1] == eiddd[1]
                for eidd in match
                for eiddd in match if eidd != eiddd]):
            self.assertFalse(g.chk_colored_match(b, match))
            self.assertFalse(g.chk_colored_match(1-b, match))
        else:
            self.assertTrue(g.chk_colored_match(b, match))
            self.assertFalse(g.chk_colored_match(1-b, match))

    @ut.skip("Skipping test_d_chk_colored_maximal_match")
    def test_d_chk_colored_maximal_match(self):
        """ test_d_chk_colored_maximal_match tests chk_colored_maximal_match """
        # print("\n\nEntering test_d_chk_colored_maximal_match\n")
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        b = 1
        # A maximal match of red edges
        # print("\n" + "left nodes = " + str(g.left_nodes))
        # print("\n" + "right nodes = " + str(g.right_nodes))
        # print("\n" + "red edges = " + str(g.red_edges))
        # print("\n" + "blue edges = " + str(g.blue_edges))
        match = [((3, None), (7, None), None), ((5, None), (9, None), None)]
        # print("\n" + "match = " + str(match))
        self.assertTrue(g.chk_colored_maximal_match(b, match))
        self.assertFalse(g.chk_colored_maximal_match(1-b, match))

        b = 0
        # print("\n" + "left nodes = " + str(g.left_nodes))
        # print("\n" + "right nodes = " + str(g.right_nodes))
        # print("\n" + "red edges = " + str(g.red_edges))
        # print("\n" + "blue edges = " + str(g.blue_edges))
        # A maximal but non-maximum match of blue edges.
        match = [((2, None), (8, None), None), ((4, None), (6, None), None),
                 ((1, None), (10, None), None)]
        # Maximal because no edges are available with unmatched endpoints.
        # Not maximum because this is a larger matching:
        #     (2, 7), (4, 9), (1, 6), (3, 8), (5, 10)

        self.assertTrue(g.chk_colored_maximal_match(b, match))
        self.assertFalse(g.chk_colored_maximal_match(1 - b, match))

        # Let's use the above counter-example
        match = [((2, None), (7, None), None), ((4, None), (9, None), None)]
        match += [((1, None), (6, None), None), ((3, None), (8, None), None)]
        match += [((5, None), (10, None), None)]
        self.assertTrue(g.chk_colored_maximal_match(b, match))
        self.assertFalse(g.chk_colored_maximal_match(1 - b, match))

        b = 0
        match = [((2, None), (8, None), None), ((4, None), (6, None), None)]
        # this match isn't maximal (eg (1, 10) makes it bigger)
        self.assertFalse(g.chk_colored_maximal_match(b, match))
        self.assertFalse(g.chk_colored_maximal_match(1 - b, match))

        b = 1
        match = [((3, None), (7, None), None), ((2, None), (8, None), None)]
        # multicolored sets can't be matchings of a single color
        self.assertFalse(g.chk_colored_maximal_match(b, match))
        self.assertFalse(g.chk_colored_maximal_match(1 - b, match))

    @ut.skip("Skipping test_r_chk_colored_maximal_match")
    def test_r_chk_colored_maximal_match_one(self):
        """ test_r_chk_colored_maximal_match_one tests at random the
        chk_colored_maximal_match function """
        # TODO: We should verify this with an independent implementation...
        # Doing this "at random" only has a probability of being correct, so we
        # can run this with a sample size.
        match = []
        stuff = helper_maximal_matchiness()
        g = stuff[0]
        reds_inc = stuff[1]
        blues_inc = stuff[5]

        match.append(random.choice(reds_inc))
        match.append(random.choice(blues_inc))

        # multicolored lists can't be matches
        self.assertFalse(g.chk_colored_maximal_match(0, match))
        self.assertFalse(g.chk_colored_maximal_match(1, match))

    @ut.skip("Skipping test_r_chk_colored_maximal_match_two")
    def test_r_chk_colored_maximal_match_two(self):
        """ test_r_chk_colored_maximal_match_two similar to previous."""
        # TODO: We should verify this with an independent implementation...
        # Doing this "at random" only has a probability of being correct, so we
        # can run this with a sample size.
        sample_size = 100
        for i in range(sample_size):
            match = []
            stuff = helper_maximal_matchiness()
            g = stuff[0]
            reds_incident_with_one = stuff[2]
            reds_incident_with_two = stuff[3]
            reds_incident_with_three = stuff[4]
            blues_incident_with_one = stuff[6]
            blues_incident_with_two = stuff[7]
            blues_incident_with_three = stuff[8]

            temp = dict()
            b = random.getrandbits(1)
            self.assertTrue(b in [0, 1])

            second = False
            if b:
                temp = g.red_edges
                if len(reds_incident_with_one) > 0:
                    match += [random.choice(reds_incident_with_one)]

                if len(reds_incident_with_two) > 0 and any(
                        [eid[0] != match[-1][0] and eid[1] != match[-1][1] for
                         eid in reds_incident_with_two]):
                    next_eid = random.choice(reds_incident_with_two)
                    while next_eid[1] == match[-1][1]:
                        next_eid = random.choice(reds_incident_with_two)
                    match += [next_eid]
                    second = True
                if not second and len(reds_incident_with_three) > 0 and any(
                        [eid[0] != match[-1][0] and eid[1] != match[-1][1] for
                         eid in reds_incident_with_three]):
                    next_eid = random.choice(reds_incident_with_three)
                    while next_eid[1] == match[-1][1]:
                        next_eid = random.choice(reds_incident_with_three)
                    match += [next_eid]
                elif second and len(reds_incident_with_three) > 0 and \
                        any([eid[0] != match[-1][0] and eid[1] != match[-1][
                            1] and
                             eid[0] != match[-2][0] and eid[1] != match[-2][1]
                             for eid in reds_incident_with_three]):
                    next_eid = random.choice(reds_incident_with_three)
                    cnd = (next_eid[1] == match[-1][1]) or (
                            next_eid[1] == match[-2][1])
                    while cnd:
                        next_eid = random.choice(reds_incident_with_three)
                        cnd = (next_eid[1] == match[-1][1]) or (next_eid[1] ==
                                                                match[-2][1])
                    match += [next_eid]
            elif not b:
                temp = g.blue_edges
                
                if len(blues_incident_with_one) > 0:
                    match += [random.choice(blues_incident_with_one)]

                if len(blues_incident_with_two) > 0 and any(
                        [eid[0] != match[-1][0] and eid[1] != match[-1][1] for
                         eid in blues_incident_with_two]):
                    next_eid = random.choice(blues_incident_with_two)
                    while next_eid[1] == match[-1][1]:
                        next_eid = random.choice(blues_incident_with_two)
                    match += [next_eid]
                    second = True
                if not second and len(blues_incident_with_three) > 0 and any(
                        [eid[0] != match[-1][0] and eid[1] != match[-1][1] for
                         eid in blues_incident_with_three]):
                    next_eid = random.choice(blues_incident_with_three)
                    while next_eid[1] == match[-1][1]:
                        next_eid = random.choice(blues_incident_with_three)
                    match += [next_eid]
                elif second and len(blues_incident_with_three) > 0 and \
                        any([eid[0] != match[-1][0] and eid[1] != match[-1][1] and
                             eid[0] != match[-2][0] and eid[1] != match[-2][1]
                             for eid in blues_incident_with_three]):
                    next_eid = random.choice(blues_incident_with_three)
                    cnd = (next_eid[1] == match[-1][1]) or (
                                next_eid[1] == match[-2][1])
                    while cnd:
                        next_eid = random.choice(blues_incident_with_three)
                        cnd = (next_eid[1] == match[-1][1]) or (next_eid[1] ==
                                                                match[-2][1])
                    match += [next_eid]



            self.assertTrue(g.chk_colored_match(b, match))

            lefts = [nid for nid in g.left_nodes if
                     any([eid[0] == nid for eid in temp])]
            unmatched_lefts = [nid for nid in lefts if
                               not any([eid[0] == nid for eid in match])]
            rights = [nid for nid in g.right_nodes if
                      any([eid[1] == nid for eid in temp])]
            unmatched_rights = [nid for nid in rights if
                                not any([eid[1] == nid for eid in match])]

            self.assertEqual(g.chk_colored_maximal_match(b, match), not any(
                [eid[0] in unmatched_lefts and eid[1] in unmatched_rights for
                 eid in temp]))

    @ut.skip("Skipping test_d_parse")
    def test_d_parse(self):
        """ test_d_parse deterministically tests the _parse function. """
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        b = 1
        input_match = [((5, None), (9, None), None)]

        out = g._parse(b, input_match)
        matched_lefts = out[0]
        matched_rights = out[1]
        unmatched_lefts = out[2]
        unmatched_rights = out[3]
        apparent_color = out[4]
        non_match_edges = out[5]

        # (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights,
        # apparent_color, non_match_edges) = out

        self.assertTrue((5, None) in matched_lefts)
        self.assertEqual(len(matched_lefts), 1)

        self.assertTrue((9, None) in matched_rights)
        self.assertEqual(len(matched_rights), 1)

        self.assertTrue((1, None) in unmatched_lefts)
        self.assertTrue((2, None) in unmatched_lefts)
        self.assertTrue((3, None) in unmatched_lefts)
        self.assertTrue((4, None) in unmatched_lefts)
        self.assertEqual(len(unmatched_lefts), 4)

        self.assertTrue((6, None) in unmatched_rights)
        self.assertTrue((7, None) in unmatched_rights)
        self.assertTrue((8, None) in unmatched_rights)
        self.assertTrue((10, None) in unmatched_rights)
        self.assertEqual(len(unmatched_rights), 4)

        self.assertEqual(apparent_color, b)

        self.assertTrue(((1, None), (7, None), None) in non_match_edges)
        self.assertTrue(((1, None), (9, None), None) in non_match_edges)
        self.assertTrue(((3, None), (7, None), None) in non_match_edges)
        self.assertTrue(((3, None), (9, None), None) in non_match_edges)
        self.assertTrue(((5, None), (7, None), None) in non_match_edges)
        self.assertEqual(len(non_match_edges), 6 - len(input_match))

    @ut.skip("Skipping test_d_clean")
    def test_d_clean(self):
        """ test_d_cleanup deterministically tests the cleanup function.
        """
        # Let's run some tests on cleanup using red matches to start.
        b = 1  # color

        # Test one: deterministic graph with edges (i, j) for i = 1, 2, ..., 5
        # and j = 6, 7, ..., 10 and color (i*j) % 2 and wt 20*(i-1) + 4*(j-5)
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        # Using the following:
        input_match = [((3, None), (7, None), None)]
        sp = [[((1, None), (9, None), None)], [((5, None), (9, None), None)]]
        result = do_a_clean_thing(b, g, input_match, sp)
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(((3, None), (7, None), None) in result)
        self.assertTrue(len(result) == 2)

        # reset
        g = None
        input_match = None
        non_match_edges = None
        shortest_paths = None
        result = None

        # Test two: Let's go again with a different first match
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        input_match = [((5, None), (9, None), None)]
        sp = [[((1, None), (7, None), None)], [((3, None), (7, None), None)]]
        result = do_a_clean_thing(b, g, input_match, sp)
        self.assertTrue(((3, None), (7, None), None) in result)
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(len(result) == 2)

        # reset
        g = None
        input_match = None
        non_match_edges = None
        shortest_paths = None
        result = None

        # Test three: Let's go again with a different first match
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        input_match = [((1, None), (7, None), None)]
        sp = [[((3, None), (9, None), None)], [((5, None), (9, None), None)]]
        result = do_a_clean_thing(b, g, input_match, sp)
        self.assertTrue(((1, None), (7, None), None) in result)
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(len(result) == 2)

        # Reset
        g = None
        input_match = None
        sp = None
        result = None

        # Test four: Let's go again with a different first match
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        if b:
            wt_dict = g.red_edges
        else:
            wt_dict = g.blue_edges
        # print("wt dict = " + str(wt_dict))
        input_match = [((3, None), (9, None), None)]
        sp = [[((1, None), (7, None), None)], [((5, None), (7, None), None)]]
        self.assertEqual(b, 1)
        for eid in input_match:
            self.assertIn(eid, wt_dict)
        for nxt_pth in sp:
            for eid in nxt_pth:
                # print("eid = " + str(eid))
                self.assertIn(eid, wt_dict)
        result = do_a_clean_thing(b, g, input_match, sp)

        self.assertIn(((3, None), (9, None), None), result)
        self.assertIn(((5, None), (7, None), None), result)
        self.assertEqual(len(result), 2)

        # Test four: Let's go again with a maximal first match
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        input_match = [((1, None), (7, None), None),
                       ((5, None), (9, None), None)]
        sp = [[((3, None), (7, None), None), ((1, None), (7, None), None),
               ((1, None), (9, None), None), ((5, None), (9, None), None)],
              [((3, None), (9, None), None), ((5, None), (9, None), None),
               ((5, None), (7, None), None), ((1, None), (7, None), None)]]
        result = do_a_clean_thing(b, g, input_match, sp)
        # the result should be the symmetric difference between the input match
        # and the heaviest shortest path so result = [(3, 9), (5, 7)] which is
        # also maximal but heavier!
        self.assertTrue(((3, None), (9, None), None) in result)
        self.assertTrue(((5, None), (7, None), None) in result)
        self.assertTrue(len(result) == 2)

        # Let's run some tests on cleanup using red matches to start.
        b = 0  # color

        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        input_match = [((2, None), (8, None), None)]
        sp = [[((1, None), (6, None), None)], [((1, None), (10, None), None)],
              [((3, None), (6, None), None)], [((3, None), (10, None), None)],
              [((4, None), (6, None), None)], [((4, None), (7, None), None)],
              [((4, None), (9, None), None)], [((4, None), (10, None), None)],
              [((5, None), (6, None), None)], [((5, None), (10, None), None)]]
        result = do_a_clean_thing(b, g, input_match, sp)

        self.assertTrue(((2, None), (8, None), None) in result)
        self.assertTrue(((5, None), (10, None), None) in result)
        self.assertTrue(((4, None), (9, None), None) in result)
        self.assertTrue(((3, None), (6, None), None) in result)

        # This is a matching
        self.assertTrue(g.chk_colored_match(b, result))
        # This is a maximal blue matching: no remaining blue edges have un-
        # matched endpoints.
        self.assertTrue(g.chk_colored_maximal_match(b, result))
        # However, it's not maximum: nodes 1 and 7 could be matched, ie the
        # dumb matching (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)
        self.assertTrue(len(result) == 4)

    def is_truth(self, result, result_test, grnd_trth):
        """ Helper function that asserts that result, result_test, and ground_truth are all consistent. """
        self.assertEqual(len(set(result)), len(result))
        self.assertEqual(len(set(result_test)), len(result_test))
        self.assertEqual(len(set(grnd_trth)), len(grnd_trth))
        for i in result:
            self.assertIn(i, result_test)
            self.assertIn(i, grnd_trth)

    @ut.skip("Skipping test_d_extend")
    def test_d_extend(self):
        """ test_d_extend deterministically tests the extend function. """
        # print("BEGINNING TEST_D_EXTEND")
        # Recall we say a matching is MAXIMAL if no edge can be added to the
        # matching without breaking the matching property... but we say a max-
        # imal matching is OPTIMAL if it has the heaviest weight.
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        self.assertTrue(len(g.red_edges) > 0)

        b = 1

        ##########################################################
        # let's start with a trivial match

        result = g.extend(b)
        self.assertIsNot(result, None)
        # print("Result from calling .extend = result = ", result)

        # Since this does not include an input match is empty, all shortest
        # paths are treated like having gain 1, so they are added to the
        # list greedily. Possible edges: (1, 7), (1, 9), (3, 7), (3, 9),
        # (5, 7), (5, 9) have weights 4, 12, 44, 52, 84, 92. So edge (5, 9)
        # is added first, which is vertex-disjoint with (1, 9) and (3, 9). Next
        # heaviest edge is (3, 7) with weight 44.
        self.assertIn(((5, None), (9, None), None), result)
        self.assertIn(((3, None), (7, None), None), result)
        self.assertEqual(len(result), 2)

        ##########################################################
        # let's use a maximal match (but sub-optimal)

        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (9, None), None)]
        result = g.extend(b, input_match)
        self.assertEqual(input_match, result)

        ##########################################################
        # let's check we raise an exception

        # We start with a pair of multi-colored edges, which can't be a match.
        input_match = [((1, None), (7, None), None)]
        input_match += [((2, None), (6, None), None)]
        self.assertTrue(input_match is not None)
        with self.assertRaises(Exception):
            g.extend(b, input_match)

        # We'll also check a pair of adjacent edges.
        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (7, None), None)]
        self.assertTrue(input_match is not None)
        with self.assertRaises(Exception):
            g.extend(b, input_match)

        # Mixed case
        input_match = [((1, None), (7, None), None)]
        input_match += [((1, None), (6, None), None)]
        self.assertTrue(input_match is not None)
        with self.assertRaises(Exception):
            g.extend(b, input_match)

        ##########################################################
        # let's check that starting with a single edge works.

        input_match = [((1, None), (7, None), None)]
        result = g.extend(b, input_match)
        # Breadth-first search for augmenting paths goes like this:
        # Unmatched lefts are 3 and 5.
        # All augmenting paths must therefore start with (3, 7), (3, 9),
        # (5, 7), or (5, 9).  Of these, (5, 9) and (3, 9) are augmenting paths
        # all by themselves, and (5, 9) is heavier.
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)
        self.assertIn(((1, None), (7, None), None), result)
        self.assertIn(((5, None), (9, None), None), result)

        # this should call _cleanup with the following:
        b_test = 1
        sp = [[((1, None), (7, None), None)], [((5, None), (9, None), None)]]
        result_test = do_a_clean_thing(b_test, g, input_match, sp)

        # both should result in this:
        grnd_trth = [((1, None), (7, None), None)]
        grnd_trth += [((5, None), (9, None), None)]
        self.is_truth(result, result_test, grnd_trth)

        # Now the match is maximal, so if we go again we should get the same
        # result back.
        input_match = result
        result = g.extend(b, input_match)
        self.assertEqual(input_match, result)

        ##########################################################
        # Let's go again with a different match.

        input_match = [((1, None), (9, None), None)]
        result = g.extend(b, input_match)

        # this should call _cleanup with the following:
        b_test = 1
        sp = [[((3, None), (7, None), None)], [((5, None), (7, None), None)]]
        result_test = do_a_clean_thing(b_test, g, input_match, sp)

        # both should result in this:
        grnd_trth = [((1, None), (9, None), None)]
        grnd_trth += [((5, None), (7, None), None)]

        self.is_truth(result, result_test, grnd_trth)

    @ut.skip("Skipping test_dd_extend")
    def test_dd_extend(self):
        """ test_dd_extend deterministically tests extend in a second way."""
        # Recall we say a matching is MAXIMAL if no edge can be added to the
        # matching without breaking the matching property... but we say a max-
        # imal matching is OPTIMAL if it has the heaviest weight.
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        b = 1

        ##########################################################
        # let's start with a trivial match

        input_match = []
        result = g.extend(b, input_match)
        # Since this match is empty, all shortest paths are treated like having
        # gain 1, so they are added to the list greedily in terms of lexico-
        # graphic ordering!
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(((3, None), (7, None), None) in result)
        self.assertEqual(len(result), 2)

        ##########################################################
        # let's use a maximal match

        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (9, None), None)]
        result = g.extend(b, input_match)
        self.assertEqual(input_match, result)

        ##########################################################
        # let's check we get None if we send a non-match in.

        # We start with a pair of multi-colored edges, which can't be a match.
        input_match = [((1, None), (7, None), None)]
        input_match += [((2, None), (6, None), None)]
        with self.assertRaises(Exception):
            result = g.extend(b, input_match)

        # We'll also check a pair of adjacent edges.
        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (7, None), None)]
        with self.assertRaises(Exception):
            result = g.extend(b, input_match)

        # Mixed case
        input_match = [((1, None), (7, None), None)]
        input_match += [((1, None), (6, None), None)]
        with self.assertRaises(Exception):
            result = g.extend(b, input_match)

        ##########################################################
        # let's check that starting with a single edge works.

        input_match = [((1, None), (7, None), None)]
        result = g.extend(b, input_match)

        # this should call _cleanup with the following:
        b_test = 1
        sp = [[((3, None), (9, None), None)], [((5, None), (9, None), None)]]
        result_test = do_a_clean_thing(b_test, g, input_match, sp)

        # both should result in this:
        result_ground_truth = [((1, None), (7, None), None),
                               ((5, None), (9, None), None)]
        self.assertEqual(set(result), set(result_test))
        self.assertEqual(set(result), set(result_ground_truth))
        self.assertEqual(set(result_test), set(result_ground_truth))

        ##########################################################
        # Let's go again with a different match.

        input_match = [((1, None), (9, None), None)]
        result = g.extend(b, input_match)

        # this should call _cleanup with the following:
        b_test = 1
        weight_dict = g.red_edges
        sp = [[((3, None), (7, None), None)], [((5, None), (7, None), None)]]
        for p in sp:
            for eid in p:
                assert eid in weight_dict
        result_test = do_a_clean_thing(b_test, g, input_match, sp)

        # both should result in this:
        grnd_trth = [((1, None), (9, None), None)]
        grnd_trth += [((5, None), (7, None), None)]
        self.assertEqual(len(set(result)), len(result))
        self.assertEqual(len(set(result_test)), len(result_test))
        self.assertEqual(len(set(grnd_trth)), len(grnd_trth))
        self.assertEqual(len(result), len(result_test))
        self.assertEqual(len(result), len(grnd_trth))

        self.assertEqual(set(result), set(result_test))
        self.assertEqual(set(result), set(grnd_trth))

    @ut.skip("Skipping test_r_extend")
    def test_r_extend(self):
        """ test_r_extend randomly tests the extend function."""
        g = make_r_graph()
        b = 1
        input_match = []

        self.assertTrue(g.chk_colored_match(b, input_match))

        result = g.extend(b, input_match)
        self.assertTrue(g.chk_colored_match(b, result))
        s = len(g.red_edges) == 0
        s = s or len(result) > len(input_match)
        self.assertTrue(s)

    @ut.skip("Skipping test_dd_boost")
    def test_dd_boost(self):
        """ test_dd_boost deterministically tests the boost function."""
        g = make_dd_graph()

        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 7.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 17.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 49.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 55.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)

        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)],  24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)],  28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)],  32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)],  36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)],  40.0)

        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)

        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None),  g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)

        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)

        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 7.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 17.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 49.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 55.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)

        # The graph from make_d_graph has at least two matchings with equal
        # weight. We tweaked that graph for this example to ensure that the
        # edge case isn't relevant to our test.
        b = 1
        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (9, None), None)]

        self.assertTrue(g.chk_colored_maximal_match(b, input_match))
        result = g.boost(b, input_match)
        # print("RESULT FROM TEST_DD_BOOST = " + str(result))
        self.assertEqual(len(result), 2)
        self.assertTrue(((1, None), (9, None), None) in result)
        self.assertTrue(((3, None), (7, None), None) in result)

    @ut.skip("Skipping test_d_optimize")
    def test_d_optimize(self):
        """ test_d_optimize deterministically tests the optimize function. """
        g = make_d_graph()
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)

        self.assertIn(((1, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (6, None), None)], 4.0)
        self.assertIn(((1, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (8, None), None)], 12.0)
        self.assertIn(((1, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((1, None), (10, None), None)], 20.0)
        #
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (6, None), None)], 24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (7, None), None)], 28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (8, None), None)], 32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (9, None), None)], 36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((2, None), (10, None), None)], 40.0)
        #
        self.assertIn(((3, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (6, None), None)], 44.0)
        self.assertIn(((3, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (8, None), None)], 52.0)
        self.assertIn(((3, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((3, None), (10, None), None)], 60.0)
        #
        self.assertIn(((4, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (6, None), None)], 64.0)
        self.assertIn(((4, None), (7, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (7, None), None)], 68.0)
        self.assertIn(((4, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (8, None), None)], 72.0)
        self.assertIn(((4, None), (9, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (9, None), None)], 76.0)
        self.assertIn(((4, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((4, None), (10, None), None)], 80.0)
        #
        self.assertIn(((5, None), (6, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (6, None), None)], 84.0)
        self.assertIn(((5, None), (8, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (8, None), None)], 92.0)
        self.assertIn(((5, None), (10, None), None), g.blue_edges)
        self.assertEqual(g.blue_edges[((5, None), (10, None), None)], 100.0)
        #
        self.assertIn(((1, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (7, None), None)], 8.0)
        self.assertIn(((1, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((1, None), (9, None), None)], 16.0)
        self.assertIn(((3, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (7, None), None)], 48.0)
        self.assertIn(((3, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((3, None), (9, None), None)], 56.0)
        self.assertIn(((5, None), (7, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (7, None), None)], 88.0)
        self.assertIn(((5, None), (9, None), None), g.red_edges)
        self.assertEqual(g.red_edges[((5, None), (9, None), None)], 96.0)
        b = 1
        final_answer = g.optimize(b)
        self.assertTrue(((5, None), (9, None), None) in final_answer)
        self.assertTrue(((3, None), (7, None), None) in final_answer)
        self.assertEqual(len(final_answer), 2)

    @ut.skip("Skipping test_d_integration")
    def test_d_integration(self):
        """ test_d_integration deterministically tests everything. """
        g = make_deterministic_graph_for_integration_test()
        
        b = 1

        # Clear grnd_truth and final_answer
        grnd_truth = None
        final_answer = None

        # Here's the ground truth.
        grnd_truth = [((1, None), (11, None), None)]
        grnd_truth += [((3, None), (13, None), None)]
        grnd_truth += [((5, None), (15, None), None)]
        grnd_truth += [((7, None), (17, None), None)]
        grnd_truth += [((9, None), (19, None), None)]

        # Get final answer
        final_answer = g.optimize(b)

        for eid in grnd_truth:
            self.assertIn(eid, final_answer)
        for eid in final_answer:
            self.assertIn(eid, grnd_truth)

        # Clear these
        grnd_truth = None
        final_answer = None

        # Here's the ground truth.
        grnd_truth = [((3, None), (12, None), None)]
        grnd_truth += [((2, None), (13, None), None)]
        grnd_truth += [((5, None), (14, None), None)]
        grnd_truth += [((4, None), (15, None), None)]
        grnd_truth += [((7, None), (16, None), None)]
        grnd_truth += [((6, None), (17, None), None)]
        grnd_truth += [((9, None), (18, None), None)]
        grnd_truth += [((8, None), (19, None), None)]
        grnd_truth += [((10, None), (20, None), None)]

        # Get final answer
        final_answer = g.optimize(1-b)

        for eid in grnd_truth:
            self.assertIn(eid, final_answer)
        assert len(final_answer) == len(grnd_truth)


tests = [TestBipartiteGraph]
# TODO: Better/more formal tests
for test in tests:
    ttr = ut.TextTestRunner(verbosity=2, failfast=True)
    ttr.run(ut.TestLoader().loadTestsFromTestCase(test))
