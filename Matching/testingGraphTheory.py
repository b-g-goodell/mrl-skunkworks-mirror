import unittest
import random
import math
from graphtheory import *
from copy import deepcopy


def make_r_graph():
    g = BipartiteGraph(None)

    # Generate a small random graph for use in random tests below.
    num_left_nodes = random.randint(3, 5)
    for i in range(num_left_nodes):
        g.add_node(0)
    num_right_nodes = random.randint(3, 5)
    for i in range(num_left_nodes):
        g.add_node(1)

    # We will assume the graph is complete and flip a fair coin to determine which edges are red and blue.
    for xid in g.left_nodes:
        for yid in g.right_nodes:
            g.add_edge(random.getrandbits(1), (xid, yid), math.log(random.random()))

    return g


def make_rr_graph():
    g = BipartiteGraph(None)

    # Generate a small random graph for use in random tests below.
    for i in range(3):
        g.add_node(0)
    for i in range(3):
        g.add_node(1)

    # We will assume the graph is complete and flip a fair coin to determine which edges are red and blue.
    for xid in g.left_nodes:
        for yid in g.right_nodes:
            g.add_edge(random.getrandbits(1), (xid, yid), math.log(random.random()))

    return g


def make_d_graph():
    g = BipartiteGraph(None)

    # Generate a small random graph for use in random tests below.
    num_left_nodes = 5
    for i in range(num_left_nodes):
        g.add_node(0)
    num_right_nodes = 5
    for i in range(num_left_nodes):
        g.add_node(1)

    # First, we set the color of (x, y) is (x*y) % 2.
    # Next will set the graph to be complete.
    # Next weight edges by starting with weight 0.0, iterating through all 25 edges up in increments of 4...
    # Formulaically, edge (i, j) with i in [1, 2, 3, 4, 5] and j in [6, 7, 8, 9, 10] has weight:
    #    wt(i, j) = 20*(i-1) + 4*(j-6)
    # Providing the following weights:
    #  (1,  6):   0.0, blue
    #  (1,  7):   4.0, red
    #  (1,  8):   8.0, blue
    #  (1,  9):  12.0, red
    #  (1, 10):  16.0, blue
    #  (2,  6):  20.0, blue
    #  (2,  7):  24.0, blue
    #  (2,  8):  28.0, blue
    #  (2,  9):  32.0, blue
    #  (2, 10):  36.0, blue
    #  (3,  6):  40.0, blue
    #  (3,  7):  44.0, red
    #  (3,  8):  48.0, blue
    #  (3,  9):  52.0, red
    #  (3, 10):  56.0, blue
    #  (4,  6):  60.0, blue
    #  (4,  7):  64.0, blue
    #  (4,  8):  68.0, blue
    #  (4,  9):  72.0, blue
    #  (4, 10):  76.0, blue
    #  (5,  6):  80.0, blue
    #  (5,  7):  84.0, red
    #  (5,  8):  88.0, blue
    #  (5,  9):  92.0, red
    #  (5, 10):  96.0, blue
    # Formulaically: (i, j) has weight 20.0*(i-1) + 4.0*((j

    wt = 0.0
    for xid in g.left_nodes:
        for yid in g.right_nodes:
            g.add_edge((xid*yid) % 2, (xid, yid), wt)
            wt += 4.0

    return g


class TestBipartiteGraph(unittest.TestCase):
    """ TestBipartiteGraph tests BipartiteGraph objects """
    def test_d_init(self):
        g = make_d_graph()

        self.assertTrue(isinstance(g, BipartiteGraph))
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges)+len(g.blue_edges), 25)
        wt = 0.0
        for xid in g.left_nodes:
            for yid in g.right_nodes:
                # Check that if the product of the identities mod 2 is 1 then the edges is red with weight 1.0
                self.assertTrue((xid*yid) % 2 == 0 or ((xid, yid) in g.red_edges and g.red_edges[(xid, yid)] == wt))
                # Check that if the product of the identities mod 2 is 0 then the edge is blue with weight 1.0
                self.assertTrue((xid*yid) % 2 == 1 or ((xid, yid) in g.blue_edges and g.blue_edges[(xid, yid)] == wt))
                wt += 4.0

    def test_d_init_by_hand(self):
        """ test_init tests initialization of a BipartiteGraph """
        g = BipartiteGraph()
        self.assertTrue(isinstance(g, BipartiteGraph))
        self.assertEqual(len(g.left_nodes), 0)
        self.assertEqual(len(g.right_nodes), 0)
        self.assertEqual(len(g.blue_edges), 0)
        self.assertEqual(len(g.red_edges), 0)
        self.assertEqual(g.count, 1)
        self.assertEqual(g.data, 'han-tyumi')

        par = {}
        par.update({'data': 'han-tyumi'})
        par.update({'count': 12})
        par.update({'left_nodes': {0: 0, 1: 1, 2: 2}})
        par.update({'right_nodes': {3: 3, 4: 4, 5: 5}})
        par.update({'blue_edges': {(0, 3): 100, (1, 4): 101, (1, 5): 99, (2, 5): 75, (2, 3):1000}})
        par.update({'red_edges': {(0, 4): 1, (0, 5): 8, (1, 3): 72, (2, 4): 1}})
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
        self.assertTrue((2, 3) in g.blue_edges and g.blue_edges[(2, 3)] == 1000)
        self.assertEqual(len(g.blue_edges), 5)

        self.assertTrue((0, 4) in g.red_edges and g.red_edges[(0, 4)] == 1)
        self.assertTrue((0, 5) in g.red_edges and g.red_edges[(0, 5)] == 8)
        self.assertTrue((1, 3) in g.red_edges and g.red_edges[(1, 3)] == 72)
        self.assertTrue((2, 4) in g.red_edges and g.red_edges[(2, 4)] == 1)
        self.assertEqual(len(g.red_edges), 4)

    def test_r_init(self):
        g = make_r_graph()

        self.assertTrue(isinstance(g, BipartiteGraph))

        self.assertEqual(g.data, 'han-tyumi')

        n = len(g.right_nodes) + len(g.left_nodes)
        self.assertEqual(g.count-1, n)

        self.assertTrue(3 <= len(g.left_nodes) <= 5)
        for i in range(1,len(g.left_nodes)+1):
            self.assertTrue(i in g.left_nodes)
            self.assertEqual(g.left_nodes[i], i)

        self.assertTrue(3 <= len(g.right_nodes) <= 5)
        for i in range(len(g.left_nodes)+1, len(g.right_nodes)+1):
            self.assertTrue(i in g.right_nodes)
            self.assertEqual(g.right_nodes[i] == i)

        self.assertEqual(len([x for x in g.right_nodes if x in g.left_nodes]), 0)

        self.assertEqual(len(g.right_nodes)*len(g.left_nodes), len(g.red_edges) + len(g.blue_edges))
        self.assertEqual(len([eid for eid in g.red_edges if eid in g.blue_edges]), 0)

        for eid in g.red_edges:
            self.assertTrue(g.red_edges[eid] <= 0.0)
        for eid in g.blue_edges:
            self.assertTrue(g.blue_edges[eid] <= 0.0)

    def test_d_add_node(self):
        g = BipartiteGraph()
        b = 0
        for i in range(100):
            n = deepcopy(g.count - 1)
            nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
            nnn = deepcopy(g.data)
            j = g.add_node(b)
            self.assertTrue(not b and j in g.left_nodes and g.left_nodes[j] == j)
            m = deepcopy(g.count - 1)
            mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
            mmm = deepcopy(g.data)
            self.assertEqual(m-n, 1)
            self.assertEqual(mm, nn)
            self.assertEqual(mmm, nnn)
            self.assertEqual(g.count-1, len(g.left_nodes)+len(g.right_nodes))
        b = 1
        for i in range(100):
            n = deepcopy(g.count - 1)
            nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
            nnn = deepcopy(g.data)
            j = g.add_node(b)
            self.assertTrue(b and j in g.right_nodes and g.right_nodes[j] == j)
            m = deepcopy(g.count - 1)
            mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
            mmm = deepcopy(g.data)
            self.assertEqual(m-n, 1)
            self.assertEqual(mm, nn)
            self.assertEqual(mmm, nnn)
            self.assertEqual(g.count-1, len(g.left_nodes)+len(g.right_nodes))

    def test_r_add_node(self):
        g = make_r_graph()
        n = g.count - 1
        num_to_add = random.randint(50, 200)
        for i in range(num_to_add):
            n = deepcopy(g.count - 1)
            nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
            nnn = deepcopy(g.data)
            b = random.getrandbits(1)

            j = g.add_node(b)

            self.assertTrue((b and j in g.right_nodes and g.right_nodes[j]==j) or
                            (not b and j in g.left_nodes and g.left_nodes[j] == j))
            m = deepcopy(g.count - 1)
            mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
            mmm = deepcopy(g.data)
            self.assertEqual(m-n, 1)
            self.assertEqual(mm, nn)
            self.assertEqual(mmm, nnn)
            self.assertEqual(g.count-1, len(g.left_nodes)+len(g.right_nodes))

    def test_d_add_edge(self):
        g = make_d_graph()
        j = g.add_node(1)
        self.assertEqual(j, 11)
        ii = g.add_node(0)
        self.assertEqual(ii, 12)
        i = min(list(g.left_nodes.keys()))
        self.assertEqual(i, 1)
        jj = min(list(g.right_nodes.keys()))
        self.assertEqual(jj, 6)
        # weights will start at 0.0 and be incremented by 0.0001
        w = 0.0

        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        b = deepcopy(len(g.right_nodes))
        g.add_edge(0, (i, j), w)
        w += 0.0001
        nn = deepcopy(len(g.blue_edges))
        mm = deepcopy(len(g.red_edges))
        aa = deepcopy(len(g.left_nodes))
        bb = deepcopy(len(g.right_nodes))
        self.assertEqual(nn-n, 1)
        self.assertEqual(m, mm)
        self.assertEqual(a, aa)
        self.assertEqual(b, bb)

        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        b = deepcopy(len(g.right_nodes))
        g.add_edge(1, (ii, jj), w)
        w += 0.0001
        nn = deepcopy(len(g.blue_edges))
        mm = deepcopy(len(g.red_edges))
        aa = deepcopy(len(g.left_nodes))
        bb = deepcopy(len(g.right_nodes))
        self.assertEqual(mm-m, 1)
        self.assertEqual(n, nn)
        self.assertEqual(a, aa)
        self.assertEqual(b, bb)

    def test_r_add_edge(self):
        g = make_r_graph()
        n = g.count - 1
        j = g.add_node(1)
        self.assertEqual(j, n+1)
        ii = g.add_node(0)
        self.assertEqual(ii, n+2)
        i = min(list(g.left_nodes.keys()))
        self.assertEqual(i, 1)
        jj = min(list(g.right_nodes.keys()))
        self.assertEqual(jj, len(g.left_nodes))

        w = -math.log(random.random())

        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        g.add_edge(0, (i, j), w)
        w += 0.0001
        nn = deepcopy(len(g.blue_edges))
        mm = deepcopy(len(g.red_edges))
        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        self.assertEqual(nn - n, 1)
        self.assertEqual(m, mm)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)

        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        g.add_edge(1, (ii, jj), w)
        w += 0.0001
        nn = deepcopy(len(g.blue_edges))
        mm = deepcopy(len(g.red_edges))
        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        self.assertEqual(mm - m, 1)
        self.assertEqual(n, nn)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)

    def test_d_del_edge(self):
        g = make_d_graph()
        p = (4*9) % 2
        self.assertTrue((not p and (4, 9) in g.blue_edges) or (p and (4,9) in g.red_edges))
        self.assertEqual(p, 0)
        self.assertTrue((4, 9) in g.blue_edges)

        b = None
        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        if (4, 9) in g.blue_edges:
            b = 0
            n = deepcopy(len(g.blue_edges))
            m = deepcopy(len(g.red_edges))
        else:
            b = 1
            n = deepcopy(len(g.red_edges))
            m = deepcopy(len(g.blue_edges))

        g.del_edge((4,9))

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        if b:
            nn = deepcopy(len(g.red_edges))
            mm = deepcopy(len(g.blue_edges))
        elif not b:
            nn = deepcopy(len(g.blue_edges))
            mm = deepcopy(len(g.red_edges))
        else:
            self.assertTrue(False)

        self.assertEqual(n-nn, 1)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)
        self.assertEqual(m, mm)


    def test_r_del_edge(self):
        g = make_r_graph()

        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))

        b = random.getrandbits(1)
        if b:
            eid = random.choice(list(g.red_edges.keys()))
            n = deepcopy(len(g.red_edges))
            m = deepcopy(len(g.blue_edges))
        else:
            eid = random.choice(list(g.blue_edges.keys()))
            n = deepcopy(len(g.blue_edges))
            m = deepcopy(len(g.red_edges))

        g.del_edge(eid)

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        if b:
            nn = deepcopy(len(g.red_edges))
            mm = deepcopy(len(g.blue_edges))
        elif not b:
            nn = deepcopy(len(g.blue_edges))
            mm = deepcopy(len(g.red_edges))
        else:
            self.assertTrue(False)

        self.assertTrue((n-nn == 1 and m == mm) or (n == nn and m - mm == 1))
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)


    def test_d_del_node(self):
        g = make_d_graph()

        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        n = deepcopy(len(g.red_edges))
        m = deepcopy(len(g.blue_edges))

        x = 3  # this is a left node with 5 incident edges
        g.del_node(x)
        # when x = 3 is deleted from the left, the edges (3, 6), (3, 7), (3, 8), (3, 9), and (3, 10) are deleted.
        # (3*6) % 2 = 0, (3*7) % 2 = 1, (3*8) % 2 = 0, (3*9) % 2 = 1, and (3*10) % 2 = 0
        # so blues: (3, 6), (3, 8), (3, 10)
        # and reds: (3, 7), (3, 9)

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        nn = deepcopy(len(g.red_edges))
        mm = deepcopy(len(g.blue_edges))

        self.assertEqual(a-aa, 1)
        self.assertEqual(c, cc)
        self.assertEqual(n-nn, 2)
        self.assertEqual(m-mm, 3)

    def test_r_del_node(self):
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

    def test_d_check_colored_match(self):
        g = make_d_graph()

        # a maximal red matching. there are 9 of these in total.
        b = 1
        match = [(3, 7), (5, 9)]
        self.assertTrue(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1-b, match))

        # a maximal blue matching. there are 25 of these in total.
        b = 0
        match = [(2, 8), (4, 6)]
        self.assertTrue(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1-b, match))

        # a non-maximal red matching. there are 9 of these in total.
        b = 1
        match = [(1, 7)]
        self.assertTrue(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1 - b, match))

        # a non-maximal blue matching. there are 25 of these in total.
        b = 0
        match = [(4, 10)]
        self.assertTrue(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1 - b, match))

        # a trivial matching.
        b = 1
        match = []
        self.assertTrue(g.check_colored_match(b, match))
        self.assertTrue(g.check_colored_match(1-b, match))

        # a non-vertex-disjoint set of red edges cannot be a match.
        b = 1
        match = [(1, 7), (3, 7)]
        self.assertFalse(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1 - b, match))

        # a non-vertex-disjoint set of blue edges cannot be a match.
        b = 0
        match = [(4, 10), (4, 8)]
        self.assertFalse(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1 - b, match))

        # a multi-colored set of edges cannot be a match for either color.
        b = 0
        match = [(4, 10), (5, 7)]
        self.assertFalse(g.check_colored_match(b, match))
        self.assertFalse(g.check_colored_match(1 - b, match))

    def test_r_check_colored_match(self):
        g = make_r_graph()

        match = []

        b = random.getrandbits(1)
        if b:
            eid = random.choice(list(g.red_edges.keys()))
        else:
            eid = random.choice(list(g.blue_edges.keys()))
        match += [eid]
        old = eid

        self.assertTrue(g.check_colored_match(b, match))

        bb = random.getrandbits(1)
        if bb:
            eid = random.choice(list(g.red_edges.keys()))
        else:
            eid = random.choice(list(g.blue_edges.keys()))
        match += [eid]

        if old[0] == eid[0] or old[1] == eid[1] or b != bb:
            self.assertFalse(g.check_colored_match(b, match))
            self.assertFalse(g.check_colored_match(1-b, match))
        else:
            self.assertTrue(g.check_colored_match(b, match))
            self.assertFalse(g.check_colored_match(1-b, match))

    def test_d_check_colored_maximal_match(self):
        g = make_d_graph()

        b = 1
        match = [(3,7), (5, 9)]
        self.assertTrue(g.check_colored_maximal_match(b, match))
        self.assertFalse(g.check_colored_maximal_match(1-b, match))

        b = 0
        match = [(2, 8), (4, 6), (1, 10)]
        self.assertTrue(g.check_colored_maximal_match(b, match))
        self.assertFalse(g.check_colored_maximal_match(1 - b, match))

        b = 0
        match = [(2, 8), (4, 6)]
        self.assertFalse(g.check_colored_maximal_match(b, match))
        self.assertFalse(g.check_colored_maximal_match(1 - b, match))

        b = 1
        match = [(3, 7), (2, 8)]  # multicolored sets can't be matchings of a single color
        self.assertFalse(g.check_colored_maximal_match(b, match))
        self.assertFalse(g.check_colored_maximal_match(1 - b, match))

    def test_r_check_colored_maximal_match(self):
        # TODO: We should verify this with an independent implementation.
        g = make_rr_graph()
        match = []

        reds_incident_with_one = [eid for eid in g.red_edges if 1 in eid]
        reds_incident_with_two = [eid for eid in g.red_edges if 2 in eid]
        reds_incident_with_three = [eid for eid in g.red_edges if 3 in eid]

        reds_incident_with_four = [eid for eid in g.red_edges if 4 in eid]
        reds_incident_with_five = [eid for eid in g.red_edges if 5 in eid]
        reds_incident_with_six = [eid for eid in g.red_edges if 6 in eid]

        blues_incident_with_one = [eid for eid in g.blue_edges if 1 in eid]
        blues_incident_with_two = [eid for eid in g.blue_edges if 2 in eid]
        blues_incident_with_three = [eid for eid in g.blue_edges if 3 in eid]

        blues_incident_with_four = [eid for eid in g.blue_edges if 4 in eid]
        blues_incident_with_five = [eid for eid in g.blue_edges if 5 in eid]
        blues_incident_with_six = [eid for eid in g.blue_edges if 6 in eid]

        match.append(random.choice(reds_incident_with_one + reds_incident_with_two + reds_incident_with_three))
        match.append(random.choice(blues_incident_with_one + blues_incident_with_two + blues_incident_with_three))
        self.assertFalse(g.check_colored_maximal_match(0, match))  # multicolored lists can't be matches
        self.assertFalse(g.check_colored_maximal_match(1, match))  # multicolored lists can't be matches

        sample_size = 1 # Tested up to 1000000 without failure... so far.
        for i in range(sample_size):
            g = make_rr_graph()
            match = []

            reds_incident_with_one = [eid for eid in g.red_edges if 1 in eid]
            reds_incident_with_two = [eid for eid in g.red_edges if 2 in eid]
            reds_incident_with_three = [eid for eid in g.red_edges if 3 in eid]

            reds_incident_with_four = [eid for eid in g.red_edges if 4 in eid]
            reds_incident_with_five = [eid for eid in g.red_edges if 5 in eid]
            reds_incident_with_six = [eid for eid in g.red_edges if 6 in eid]

            blues_incident_with_one = [eid for eid in g.blue_edges if 1 in eid]
            blues_incident_with_two = [eid for eid in g.blue_edges if 2 in eid]
            blues_incident_with_three = [eid for eid in g.blue_edges if 3 in eid]

            blues_incident_with_four = [eid for eid in g.blue_edges if 4 in eid]
            blues_incident_with_five = [eid for eid in g.blue_edges if 5 in eid]
            blues_incident_with_six = [eid for eid in g.blue_edges if 6 in eid]

            b = random.getrandbits(1)
            if b:
                if len(reds_incident_with_one) > 0:
                    match.append(random.choice(reds_incident_with_one))
                if len(reds_incident_with_two) > 0:
                    match.append(random.choice(reds_incident_with_two))
                if len(reds_incident_with_three) > 0:
                    match.append(random.choice(reds_incident_with_three))
            else:
                if len(blues_incident_with_one) > 0:
                    match.append(random.choice(blues_incident_with_one))
                if len(blues_incident_with_two) > 0:
                    match.append(random.choice(blues_incident_with_two))
                if len(blues_incident_with_three) > 0:
                    match.append(random.choice(blues_incident_with_three))

            touched_lefts = [eid[0] for eid in match]
            touched_rights = [eid[1] for eid in match]
            if len(touched_lefts) == len(set(touched_lefts)) and len(touched_rights) == len(set(touched_rights)) and len(match) > 0:
                self.assertTrue(g.check_colored_maximal_match(b, match))
            else:
                self.assertFalse(g.check_colored_maximal_match(b, match))

    def test_d_parse(self):
        g = make_d_graph()

        b = 1
        input_match = [(5, 9)]

        out = g._parse(b, input_match)
        (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, apparent_color, non_match_edges) = out

        self.assertTrue(5 in matched_lefts)
        self.assertEqual(len(matched_lefts), 1)
        self.assertTrue(9 in matched_rights)
        self.assertEqual(len(matched_rights), 1)
        self.assertTrue(1 in unmatched_lefts)
        self.assertTrue(2 in unmatched_lefts)
        self.assertTrue(3 in unmatched_lefts)
        self.assertTrue(4 in unmatched_lefts)
        self.assertEqual(len(unmatched_lefts), 4)
        self.assertTrue(6 in unmatched_rights)
        self.assertTrue(7 in unmatched_rights)
        self.assertTrue(8 in unmatched_rights)
        self.assertTrue(10 in unmatched_rights)
        self.assertEqual(len(unmatched_rights), 4)
        self.assertEqual(apparent_color, b)
        self.assertTrue((1, 7) in non_match_edges)
        self.assertTrue((1, 9) in non_match_edges)
        self.assertTrue((3, 7) in non_match_edges)
        self.assertTrue((3, 9) in non_match_edges)
        self.assertTrue((5, 7) in non_match_edges)
        self.assertEqual(len(non_match_edges), 5)

    def test_d_cleanup(self):
        """ test_d_cleanup deterministically tests the cleanup function.
        """
        # Let's run some tests on cleanup using red matches to start.
        b = 1  # color

        # Test one: deterministic graph with all edges (i, j) with i = 1, 2, .., 5, and j = 6, 7, .., 10
        # where (i, j) has color (i*j) % 2 and weight 20*(i-1) + 4*(j-6)
        g = make_d_graph()

        # Using the following:
        input_match = [(3,7)]
        non_match_edges = [(1,7), (1,9), (3,9), (5,7), (5, 9)]
        shortest_paths = [[(1,9)], [(5, 9)]]
        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        print(result)
        self.assertTrue((5, 9) in result)
        self.assertTrue((3, 7) in result)
        self.assertTrue(len(result) == 2)

        # reset
        g = None
        input_match = None
        non_match_edges = None
        shortest_paths = None
        result = None

        # Test two: Let's go again with a different first match
        g = make_d_graph()
        input_match = [(5, 9)]
        non_match_edges = [(1, 7), (1, 9), (3, 7), (3, 9), (5, 7)]
        shortest_paths = [[(1, 7)], [(3, 7)]]
        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        self.assertTrue((3, 7) in result)
        self.assertTrue((5, 9) in result)
        self.assertTrue(len(result) == 2)

        # reset
        g = None
        input_match = None
        non_match_edges = None
        shortest_paths = None
        result = None

        # Test three: Let's go again with a different first match
        g = make_d_graph()
        input_match = [(1, 7)]
        non_match_edges = [(1, 9), (3, 7), (3, 9), (5, 7), (5, 9)]
        shortest_paths = [[(3, 9)], [(5, 9)]] # (3, 9) has wt 52, (5, 9) has weight 92 so we will end up with (5, 9)
        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        self.assertTrue((1, 7) in result)
        self.assertTrue((5, 9) in result)
        self.assertTrue(len(result) == 2)

        # Test four: Let's go again with a different first match
        g = make_d_graph()
        input_match = [(1, 7)]
        non_match_edges = [(1, 9), (3, 7), (3, 9), (5, 7), (5, 9)]
        shortest_paths = [[(3, 9)], [(5, 9)]] # (3, 9) has wt 52, (5, 9) has weight 92 so we will end up with (5, 9)
        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        self.assertTrue((1, 7) in result)
        self.assertTrue((5, 9) in result)
        self.assertTrue(len(result) == 2)

        # Test four: Let's go again with a different first match
        g = make_d_graph()
        input_match = [(1, 7), (5, 9)] # note this is already a maximal match!
        non_match_edges = [(1, 9), (3, 7), (3, 9), (5, 7)]
        shortest_paths = [[(3, 7), (1, 7), (1, 9), (5, 9)], [(3, 9), (5, 9), (5, 7), (1, 7)]]
        # gain of first shortest path is
        #     20*(3-1) + 4*(7-6) - 20*(1-1) - 4*(7-6) + 20*(1-1) + 4*(9-6) - 20*(5-1) - 4*(9-6) = -40
        # gain of second shortest path is
        #     20*(3-1) + 4*(9-6) - 20*(5-1) - 4*(9-6) + 20*(5-1) + 4*(7-6) - 20*(1-1) + 4*(7-6) = +40
        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        # the result should be the symmetric difference between the input match and the heaviest shortest path
        # so result = [(3, 9), (5, 7)] which is also maximal but heavier!
        self.assertTrue((3, 9) in result)
        self.assertTrue((5, 7) in result)
        self.assertTrue(len(result) == 2)

        # Let's run some tests on cleanup using red matches to start.
        b = 0  # color

        g = make_d_graph()
        input_match = [(2, 8)]
        non_match_edges = [(1,6), (1, 8), (1, 10), (2, 6), (2, 7), (2, 9), (2, 10), (3, 6), (3, 8), (3, 10),
                           (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 6), (5, 8), (5, 10)]
        shortest_paths = [[(1, 6)], [(1, 10)], [(3, 6)], [(3, 10)], [(4, 6)], [(4, 7)], [(4, 9)], [(4, 10)], [(5, 6)],
                          [(5, 10)]]
        # gains of these paths:
        #  (1,  6): 20*(1-1) + 4*(6-6)  = 0.0
        #  (1, 10): 20*(1-1) + 4*(10-6) = 16.0
        #  (3,  6): 20*(3-1) + 4*(6-6)  = 40.0
        #  (3, 10): 20*(3-1) + 4*(10-6) = 56.0
        #  (4,  6): 20*(4-1) + 4*(6-6)  = 60.0
        #  (4,  7): 20*(4-1) + 4*(7-6)  = 64.0
        #  (4,  9): 20*(4-1) + 4*(9-6)  = 72.0
        #  (4, 10): 20*(4-1) + 4*(10-6) = 76.0
        #  (5,  6): 20*(5-1) + 4*(6-6)  = 80.0
        #  (5, 10): 20*(5-1) + 4*(10-6) = 96.0

        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        # greedy selection of vertex-disjoint sets: first we pick (5, 10), then (4, 9), then (3, 6).
        # the symmetric difference between this set of edges and the input_match (2,8) is the union.
        self.assertTrue((2, 8) in result)
        self.assertTrue((5, 10) in result)
        self.assertTrue((4, 9) in result)
        self.assertTrue((3, 6) in result)
        self.assertTrue(len(result) == 4)
        # This is now a maximal blue matching.
        self.assertTrue(g.check_colored_maximal_match(b, result))

        # Now let's use the previous result as a new input match and see what happens!
        input_match = result
        # The shortest paths

        # reset
        g = None
        input_match = None
        non_match_edges = None
        shortest_paths = None
        result = None

        # Test two: Let's go again with a different first match
        g = make_d_graph()
        input_match = [(2, 6)]
        non_match_edges = [(1, 6), (1, 8), (1, 10), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 6), (3, 8), (3, 10)]
        non_match_edges += [(4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 6), (5, 8), (5, 10)]
        for eid in non_match_edges:
            self.assertTrue(eid in g.blue_edges)
        shortest_paths = [[(1, 8)], [(1, 10)], [(3, 8)], [(3, 10)], [(4, 7)], [(4, 8)], [(4, 9)], [(4, 10)], [(5, 8)],
                          [(5, 10)]]
        # Weights:
        # [(1,  8)]:  20*(1-1) + 4*(8-6) =  8.0
        # [(1, 10)]: 20*(1-1) + 4*(10-6) = 16.0
        # [(3,  8)]: 20*(3-1) + 4*(8-6)  = 48.0
        # [(3, 10)]: 20*(3-1) + 4*(10-6) = 56.0
        # [(4,  7)]: 20*(4-1) + 4*(7-6)  = 64.0
        # [(4,  8)]: 20*(4-1) + 4*(8-6)  = 68.0
        # [(4,  9)]: 20*(4-1) + 4*(9-6)  = 72.0
        # [(4, 10)]: 20*(4-1) + 4*(10-6) = 76.0
        # [(5,  8)]: 20*(5-1) + 4*(8-6)  = 88.0
        # [(5, 10)]: 20*(5-1) + 4*(10-6) = 96.0
        # Vertex disjoint result should be: (5, 10), (4, 9), (3, 8)
        # symdif with input_match of (2, 6) is the union
        result = g._cleanup(b, shortest_paths, non_match_edges, input_match)
        self.assertTrue((5, 10) in result)
        self.assertTrue((4, 9) in result)
        self.assertTrue((3, 8) in result)
        self.assertTrue((2, 6) in result)
        self.assertTrue(len(result) == 4)

    def test_d_extend_match(self):
        pass

    def test_r_extend_match(self):
        g = make_r_graph()

    def test_d_boost_match(self):
        pass

    def test_r_boost_match(self):
        g = make_r_graph()


tests = [TestBipartiteGraph]
for test in tests:
    unittest.TextTestRunner(verbosity=2, failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
