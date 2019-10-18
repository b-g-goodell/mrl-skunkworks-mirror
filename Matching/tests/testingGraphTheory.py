import unittest as ut
import random
from graphtheory import *
from copy import deepcopy


def make_r_graph():
    g = BipartiteGraph(None)

    # Generate a small random graph for use in random tests below.
    num_left_nodes = random.randint(3, 5)
    for i in range(num_left_nodes):
        g.add_node(0)
    num_right_nodes = random.randint(3, 5)
    for i in range(num_right_nodes):
        g.add_node(1)

    # We assign random edge weights
    for xid in g.left_nodes:
        for yid in g.right_nodes:
            g.add_edge(random.getrandbits(1), (xid, yid), random.random())

    return g


def make_rr_graph():
    g = BipartiteGraph(None)

    # Generate a small random graph for use in random tests below.
    for i in range(3):
        g.add_node(0)
    for i in range(3):
        g.add_node(1)

    # We will assume the graph is complete and flip a fair coin to determine 
    # which edges are red and blue.
    for xid in g.left_nodes:
        for yid in g.right_nodes:
            g.add_edge(random.getrandbits(1), (xid, yid), random.random())

    return g


def make_d_graph():
    # This generates a graph deterministically.
    g = BipartiteGraph(None)

    # Generate a small random graph for use in random tests below.
    num_left_nodes = 5
    for i in range(num_left_nodes):
        g.add_node(0)
    num_right_nodes = 5
    for i in range(num_right_nodes):
        g.add_node(1)

    # Weight edges by starting with weight 0.0, iterating through all 25 edges 
    # up in increments of 4.0. Formulaically, we use edge ident 
    #         eid := ((i, t), (j, s), r) 
    # and color 
    #       color := (i*j)%2
    # and weight
    #    wt(i, j) := 20*(i-1) + 4*(j-5)
    
    for xid in sorted(list(g.left_nodes.keys())):
        for yid in sorted(list(g.right_nodes.keys())):
            wt = 20.0*(xid[0]-1) + 4.0*(yid[0]-5)
            g.add_edge((xid[0]*yid[0]) % 2, (xid, yid), wt)
            
    assert len(g.left_nodes)==5
    assert len(g.right_nodes)==5
    assert len(g.red_edges)==6
    assert len(g.blue_edges)==19
    
    assert ((1, None), (6, None), None) in g.blue_edges
    assert g.blue_edges[((1, None), (6, None), None)] == 4.0 
    assert ((1, None), (8, None), None) in g.blue_edges
    assert g.blue_edges[((1, None), (8, None), None)] == 12.0
    assert ((1, None), (10, None), None) in g.blue_edges
    assert g.blue_edges[((1, None), (10, None), None)] == 20.0
    
    assert ((2, None), (6, None), None) in g.blue_edges
    assert g.blue_edges[((2, None), (6, None), None)] == 24.0
    assert ((2, None), (7, None), None) in g.blue_edges
    assert g.blue_edges[((2, None), (7, None), None)] == 28.0
    assert ((2, None), (8, None), None) in g.blue_edges
    assert g.blue_edges[((2, None), (8, None), None)] == 32.0
    assert ((2, None), (9, None), None) in g.blue_edges
    assert g.blue_edges[((2, None), (9, None), None)] == 36.0
    assert ((2, None), (10, None), None) in g.blue_edges
    assert g.blue_edges[((2, None), (10, None), None)] == 40.0
    
    assert ((3, None), (6, None), None) in g.blue_edges
    assert g.blue_edges[((3, None), (6, None), None)] == 44.0
    assert ((3, None), (8, None), None) in g.blue_edges
    assert g.blue_edges[((3, None), (8, None), None)] == 52.0
    assert ((3, None), (10, None), None) in g.blue_edges
    assert g.blue_edges[((3, None), (10, None), None)] == 60.0
    
    assert ((4, None), (6, None), None) in g.blue_edges
    assert g.blue_edges[((4, None), (6, None), None)] == 64.0
    assert ((4, None), (7, None), None) in g.blue_edges
    assert g.blue_edges[((4, None), (7, None), None)] == 68.0
    assert ((4, None), (8, None), None) in g.blue_edges
    assert g.blue_edges[((4, None), (8, None), None)] == 72.0
    assert ((4, None), (9, None), None) in g.blue_edges
    assert g.blue_edges[((4, None), (9, None), None)] == 76.0
    assert ((4, None), (10, None), None) in g.blue_edges
    assert g.blue_edges[((4, None), (10, None), None)] == 80.0
    
    assert ((5, None), (6, None), None) in g.blue_edges
    assert g.blue_edges[((5, None), (6, None), None)] == 84.0
    assert ((5, None), (8, None), None) in g.blue_edges
    assert g.blue_edges[((5, None), (8, None), None)] == 92.0
    assert ((5, None), (10, None), None) in g.blue_edges
    assert g.blue_edges[((5, None), (10, None), None)] == 100.0
    
    assert ((1, None), (7, None), None) in g.red_edges
    assert g.red_edges[((1, None), (7, None), None)] == 8.0
    assert ((1, None), (9, None), None) in g.red_edges
    assert g.red_edges[((1, None), (9, None), None)] == 16.0
    assert ((3, None), (7, None), None) in g.red_edges
    assert g.red_edges[((3, None), (7, None), None)] == 48.0
    assert ((3, None), (9, None), None) in g.red_edges
    assert g.red_edges[((3, None), (9, None), None)] == 56.0
    assert ((5, None), (7, None), None) in g.red_edges
    assert g.red_edges[((5, None), (7, None), None)] == 88.0
    assert ((5, None), (9, None), None) in g.red_edges
    assert g.red_edges[((5, None), (9, None), None)] == 96.0
    
    return g


def make_dd_graph():
    g = make_d_graph()
    g.red_edges[((1, None), (7, None), None)] -= 1.0
    g.red_edges[((1, None), (9, None), None)] += 1.0
    g.red_edges[((3, None), (7, None), None)] += 1.0
    g.red_edges[((3, None), (9, None), None)] -= 1.0
    return g


class TestBipartiteGraph(ut.TestCase):
    """ TestBipartiteGraph tests BipartiteGraph objects """
    
    #  @ut.skip("Skipping test_d_init")
    def test_d_init(self):
        g = make_d_graph()

        self.assertTrue(isinstance(g, BipartiteGraph))
        self.assertEqual(len(g.left_nodes), 5)
        self.assertEqual(len(g.right_nodes), 5)
        self.assertEqual(len(g.red_edges), 6)
        self.assertEqual(len(g.blue_edges), 19)
        wt = 4.0
        for xid in sorted(list(g.left_nodes.keys())):
            for yid in sorted(list(g.right_nodes.keys())):
                self.assertTrue((xid, yid, None) in g.red_edges or \
                    (xid, yid, None) in g.blue_edges)
                if xid[0]*yid[0] % 2:
                    self.assertTrue((xid, yid, None) in g.red_edges and \
                        g.red_edges[(xid, yid, None)] == wt)
                else:
                    self.assertTrue((xid, yid, None) in g.blue_edges and \
                        g.blue_edges[(xid, yid, None)] == wt)
                wt += 4.0


    #  @ut.skip("Skipping test_d_init_by_hand")
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

    # @ut.skip("Skipping test_r_init")
    def test_r_init(self):
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

    #  #  @ut.skip("Skipping test_d_add_node")
    def test_d_add_node(self):
        g = BipartiteGraph()
        b = 0
        for i in range(100):
            n = deepcopy(g.count - 1)
            nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
            nnn = deepcopy(g.data)
            j = g.add_node(b)  # create a new left node
            x = j in g.left_nodes
            y = g.left_nodes[j] == j
            self.assertTrue(not b and x and y)
            m = deepcopy(g.count - 1)
            mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
            mmm = deepcopy(g.data)
            self.assertEqual(m-n, 1)  # we only added one node
            self.assertEqual(mm, nn)  
            self.assertEqual(mm, 0) # still haven't added any edges
            self.assertEqual(mmm, nnn)  # arbitrary data preserved
            self.assertEqual(g.count-1, len(g.left_nodes)+len(g.right_nodes))
        b = 1
        for i in range(100):
            n = deepcopy(g.count - 1)
            nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
            nnn = deepcopy(g.data)
            j = g.add_node(b)  # create a new right node 
            x = j in g.right_nodes
            y = g.right_nodes[j] == j
            self.assertTrue(b and x and y)
            m = deepcopy(g.count - 1)
            mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
            mmm = deepcopy(g.data)
            self.assertEqual(m-n, 1)  # we only added one node
            self.assertEqual(mm, nn)  
            self.assertEqual(mm, 0) # still haven't added any edges
            self.assertEqual(mmm, nnn)  # arbitrary data preserved
            self.assertEqual(g.count-1, len(g.left_nodes)+len(g.right_nodes))


    #  @ut.skip("Skipping test_r_add_node")
    def test_r_add_node(self):
        g = make_r_graph()  
        # n = g.count - 1
        num_to_add = random.randint(50, 200)
        for i in range(num_to_add):
            n = deepcopy(g.count - 1)
            nn = deepcopy(len(g.red_edges) + len(g.blue_edges))
            nnn = deepcopy(g.data)
            b = random.getrandbits(1)

            j = g.add_node(b)
            x = j in g.right_nodes
            y = g.right_nodes[j] == j
            
            xx = j in g.left_nodes
            yy = g.left_nodes[j] == j
            self.assertTrue((b and x and y) or (not b and xx and yy))
            
            m = deepcopy(g.count - 1)
            mm = deepcopy(len(g.red_edges) + len(g.blue_edges))
            mmm = deepcopy(g.data)
            self.assertEqual(m-n, 1)
            self.assertEqual(mm, nn)
            self.assertEqual(mmm, nnn)
            self.assertEqual(g.count-1, len(g.left_nodes)+len(g.right_nodes))


    #  @ut.skip("Skipping test_d_add_edge")
    def test_d_add_edge(self):
        g = make_d_graph()  
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

        ii = g.add_node(0) # add a new left node with no new edges
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


    #  @ut.skip("Skipping test_r_add_edge")
    def test_r_add_edge(self):
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

        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        g.add_edge(0, (ii, jj), w) # add a blue edge from ii to jj
        w += 0.1
        nn = deepcopy(len(g.blue_edges))
        mm = deepcopy(len(g.red_edges))
        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        self.assertEqual(nn, n+1)
        self.assertEqual(m, mm)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)

        g.add_edge(1, (ii, j), w) # add a red edge from ii to j
        w += 0.1
        n = deepcopy(len(g.blue_edges))
        m = deepcopy(len(g.red_edges))
        a = deepcopy(len(g.left_nodes))
        c = deepcopy(len(g.right_nodes))
        self.assertEqual(nn, n)
        self.assertEqual(m, mm+1)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)


    #  @ut.skip("Skipping test_d_del_edge")
    def test_d_del_edge(self):
        # we will delete edge (4, 9)
        g = make_d_graph()

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
        self.assertFalse(((4, None), (9, None), None) in g.red_edges or \
            ((4, None), (9, None), None) in g.blue_edges)

        aa = deepcopy(len(g.left_nodes))
        cc = deepcopy(len(g.right_nodes))
        nn = deepcopy(len(g.red_edges))
        mm = deepcopy(len(g.blue_edges))

        self.assertEqual(n, nn)
        self.assertEqual(m, mm+1)
        self.assertEqual(a, aa)
        self.assertEqual(c, cc)


    #  @ut.skip("Skipping test_r_del_edge")
    def test_r_del_edge(self):
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


    #  @ut.skip("Skipping test_d_del_node")
    def test_d_del_node(self):
        g = make_d_graph()

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


    #  @ut.skip("Skipping test_r_del_node")
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


    #  @ut.skip("Skipping test_d_chhk_colored_match")
    def test_d_chhk_colored_match(self):
        g = make_d_graph()

        # a maximal red matching. there are 9 of these in total.
        b = 1
        match = [((3, None), (7, None), None), ((5, None), (9, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1-b, match))

        # a maximal blue matching. there are 25 of these in total.
        b = 0
        match = [((2, None), (8, None), None), ((4, None), (6, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1-b, match))

        # a non-maximal red matching. there are 9 of these in total.
        b = 1
        match = [((1, None), (7, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a non-maximal blue matching. there are 25 of these in total.
        b = 0
        match = [((4, None), (10, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a trivial matching.
        b = 1
        match = []
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertTrue(g.chhk_colored_match(1-b, match))

        # a non-vertex-disjoint set of red edges cannot be a match.
        b = 1
        match = [((1, None), (7, None), None), ((3, None), (7, None), None)]
        self.assertFalse(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a non-vertex-disjoint set of blue edges cannot be a match.
        b = 0
        match = [((4, None), (10, None), None), ((4, None), (8, None), None)]
        self.assertFalse(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a multi-colored set of edges cannot be a match for either color.
        b = 0
        match = [((4, None), (10, None), None), ((5, None), (7, None), None)]
        self.assertFalse(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

    #  @ut.skip("Skipping test_d_chhk_colored_match")
    def test_d_chhk_colored_match(self):
        g = make_d_graph()

        # a maximal red matching. there are 9 of these in total.
        b = 1
        match = [((3, None), (7, None), None), ((5, None), (9, None), None)]
        self.assertIn(match[0], g.red_edges)
        self.assertIn(match[1], g.red_edges)
        # print(match)
        # print(list(g.red_edges.keys()))
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1-b, match))

        # a maximal blue matching. there are 25 of these in total.
        b = 0
        match = [((2, None), (8, None), None), ((4, None), (6, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1-b, match))

        # a non-maximal red matching. there are 9 of these in total.
        b = 1
        match = [((1, None), (7, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a non-maximal blue matching. there are 25 of these in total.
        b = 0
        match = [((4, None), (10, None), None)]
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a trivial matching.
        b = 1
        match = []
        self.assertTrue(g.chhk_colored_match(b, match))
        self.assertTrue(g.chhk_colored_match(1-b, match))

        # a non-vertex-disjoint set of red edges cannot be a match.
        b = 1
        match = [((1, None), (7, None), None), ((3, None), (7, None), None)]
        self.assertFalse(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a non-vertex-disjoint set of blue edges cannot be a match.
        b = 0
        match = [((4, None), (10, None), None), ((4, None), (8, None), None)]
        self.assertFalse(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

        # a multi-colored set of edges cannot be a match for either color.
        b = 0
        match = [((4, None), (10, None), None), ((5, None), (7, None), None)]
        self.assertFalse(g.chhk_colored_match(b, match))
        self.assertFalse(g.chhk_colored_match(1 - b, match))

    #  @ut.skip("Skipping test_r_chhk_colored_match")
    def test_r_chhk_colored_match(self):
        ''' test_r_chhk_colored_match tests using a random graph '''
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
        self.assertTrue(g.chhk_colored_match(b, match))

            
        # Pick 2 random edge idents from that color edge dict, add to match
        eid = random.choice(list(wt_dict.keys()))  
        wt_dict[eid] = random.random()
        match += [eid]
        old = eid

        eid = random.choice(list(wt_dict.keys()))
        while eid in match:
            eid = random.choice(list(wt_dict.keys()))
        self.assertTrue(eid not in other_wt_dict or other_wt_dict[eid] == 0.0)
        wt_dict[eid] = random.random()
        match += [eid]
        
        # If these edges are not vtx-disjoint with each other or input_match,
        # then adding these edges violates the match condition. Otherwise, not.
        
        if any([eidd[0] == eiddd[0] or eidd[1] == eiddd[1] for eidd in match \
            for eiddd in match if eidd != eiddd]):
            self.assertFalse(g.chhk_colored_match(b, match))
            self.assertFalse(g.chhk_colored_match(1-b, match)) 
        else:
            self.assertTrue(g.chhk_colored_match(b, match))
            self.assertFalse(g.chhk_colored_match(1-b, match))

    #  @ut.skip("Skipping test_d_chhk_colored_maximal_match")
    def test_d_chhk_colored_maximal_match(self):
        # print("\n\nEntering test_d_chhk_colored_maximal_match\n")
        g = make_d_graph()

        b = 1
        # A maximal match of red edges
        # print("\nleft nodes = " + str(g.left_nodes))
        # print("\nright nodes = " + str(g.right_nodes))
        # print("\nred edges = " + str(g.red_edges))
        # print("\nblue edges = " + str(g.blue_edges))
        match = [((3, None), (7, None), None), ((5, None), (9, None), None)]
        # print("\nmatch = " + str(match))
        self.assertTrue(g.chhk_colored_maximal_match(b, match))
        self.assertFalse(g.chhk_colored_maximal_match(1-b, match))

        b = 0
        # print("\nleft nodes = " + str(g.left_nodes))
        # print("\nright nodes = " + str(g.right_nodes))
        # print("\nred edges = " + str(g.red_edges))
        # print("\nblue edges = " + str(g.blue_edges))
        # A maximal but non-maximum match of blue edges.
        match = [((2, None), (8, None), None), ((4, None), (6, None), None), \
            ((1, None), (10, None), None)]
        # Maximal because no edges are available with unmatched endpoints.
        # Not maximum because this is a larger matching: 
        #     (2, 7), (4, 9), (1, 6), (3, 8), (5, 10)
        
        self.assertTrue(g.chhk_colored_maximal_match(b, match))
        self.assertFalse(g.chhk_colored_maximal_match(1 - b, match))
        
        # Let's use the above counter-example
        match = [((2, None), (7, None), None), ((4, None), (9, None), None)]
        match += [((1, None), (6, None), None), ((3, None), (8, None), None)]
        match += [((5, None), (10, None), None)]
        self.assertTrue(g.chhk_colored_maximal_match(b, match))
        self.assertFalse(g.chhk_colored_maximal_match(1 - b, match))
        

        b = 0
        match = [((2, None), (8, None), None), ((4, None), (6, None), None)]
        # this match isn't maximal (eg (1, 10) makes it bigger)
        self.assertFalse(g.chhk_colored_maximal_match(b, match)) 
        self.assertFalse(g.chhk_colored_maximal_match(1 - b, match))

        b = 1
        match = [((3, None), (7, None), None), ((2, None), (8, None), None)]  
        # multicolored sets can't be matchings of a single color
        self.assertFalse(g.chhk_colored_maximal_match(b, match))
        self.assertFalse(g.chhk_colored_maximal_match(1 - b, match))

    #  @ut.skip("Skipping test_r_chhk_colored_maximal_match")
    def test_r_chhk_colored_maximal_match_one(self):
        # TODO: We should verify this with an independent implementation...
        # Doing this "at random" only has a probability of being correct, so we
        # can run this with a sample size.
        g = make_rr_graph()  
        match = []

        reds_incident_with_one = [eid for eid in g.red_edges if \
            (1, None) in eid and g.red_edges[eid] > 0.0]
        reds_incident_with_two = [eid for eid in g.red_edges if \
            (2, None) in eid and g.red_edges[eid] > 0.0]
        reds_incident_with_three = [eid for eid in g.red_edges if \
            (3, None) in eid and g.red_edges[eid] > 0.0]
            
        reds_inc = reds_incident_with_one
        reds_inc += reds_incident_with_two
        reds_inc += reds_incident_with_three

        blues_incident_with_one = [eid for eid in g.blue_edges if \
            (1, None) in eid and g.blue_edges[eid] > 0.0]
        blues_incident_with_two = [eid for eid in g.blue_edges if \
            (2, None) in eid and g.blue_edges[eid] > 0.0]
        blues_incident_with_three = [eid for eid in g.blue_edges if \
            (3, None) in eid and g.blue_edges[eid] > 0.0]
            
        blues_inc = blues_incident_with_one
        blues_inc += blues_incident_with_two
        blues_inc += blues_incident_with_three
        
        
        while len(reds_inc) == 0 or len(blues_inc) == 0:
            g = make_rr_graph()  

            reds_incident_with_one = [eid for eid in g.red_edges if \
                (1, None) in eid and g.red_edges[eid] > 0.0]
            reds_incident_with_two = [eid for eid in g.red_edges if \
                (2, None) in eid and g.red_edges[eid] > 0.0]
            reds_incident_with_three = [eid for eid in g.red_edges if \
                (3, None) in eid and g.red_edges[eid] > 0.0]

            reds_inc = reds_incident_with_one
            reds_inc += reds_incident_with_two
            reds_inc += reds_incident_with_three
            
            blues_incident_with_one = [eid for eid in g.blue_edges if \
                (1, None) in eid and g.blue_edges[eid] > 0.0]
            blues_incident_with_two = [eid for eid in g.blue_edges if \
                (2, None) in eid and g.blue_edges[eid] > 0.0]
            blues_incident_with_three = [eid for eid in g.blue_edges if \
                (3, None) in eid and g.blue_edges[eid] > 0.0]
                    
            blues_inc = blues_incident_with_one
            blues_inc += blues_incident_with_two
            blues_inc += blues_incident_with_three

        match.append(random.choice(reds_inc))
        match.append(random.choice(blues_inc))
        
        # multicolored lists can't be matches
        self.assertFalse(g.chhk_colored_maximal_match(0, match))  
        self.assertFalse(g.chhk_colored_maximal_match(1, match)) 

    #  @ut.skip("Skipping test_r_chhk_colored_maximal_match_two")
    def test_r_chhk_colored_maximal_match_two(self):
        # TODO: We should verify this with an independent implementation...
        # Doing this "at random" only has a probability of being correct, so we 
        # can run this with a sample size.
        sample_size = 100  
        for i in range(sample_size):
            g = make_rr_graph() 
            match = []

            reds_incident_with_one = [eid for eid in g.red_edges if \
                (1, None) in eid and g.red_edges[eid] > 0.0]
            reds_incident_with_two = [eid for eid in g.red_edges if \
                (2, None) in eid and g.red_edges[eid] > 0.0]
            reds_incident_with_three = [eid for eid in g.red_edges if \
                (3, None) in eid and g.red_edges[eid] > 0.0]
                    
            reds_inc = reds_incident_with_one
            reds_inc += reds_incident_with_two
            reds_inc += reds_incident_with_three

            blues_incident_with_one = [eid for eid in g.blue_edges if \
                (1, None) in eid and g.blue_edges[eid] > 0.0]
            blues_incident_with_two = [eid for eid in g.blue_edges if \
                (2, None) in eid and g.blue_edges[eid] > 0.0]
            blues_incident_with_three = [eid for eid in g.blue_edges if \
                (3, None) in eid and g.blue_edges[eid] > 0.0]
                
            blues_inc = blues_incident_with_one
            blues_inc += blues_incident_with_two
            blues_inc += blues_incident_with_three
            
            while len(reds_inc) == 0 or len(blues_inc) == 0:
                g = make_rr_graph()  

                reds_incident_with_one = [eid for eid in g.red_edges if \
                    (1, None) in eid and g.red_edges[eid] > 0.0]
                reds_incident_with_two = [eid for eid in g.red_edges if \
                    (2, None) in eid and g.red_edges[eid] > 0.0]
                reds_incident_with_three = [eid for eid in g.red_edges if \
                    (3, None) in eid and g.red_edges[eid] > 0.0]

                reds_inc = reds_incident_with_one
                reds_inc += reds_incident_with_two
                reds_inc += reds_incident_with_three
                
                blues_incident_with_one = [eid for eid in g.blue_edges if \
                    (1, None) in eid and g.blue_edges[eid] > 0.0]
                blues_incident_with_two = [eid for eid in g.blue_edges if \
                    (2, None) in eid and g.blue_edges[eid] > 0.0]
                blues_incident_with_three = [eid for eid in g.blue_edges if \
                    (3, None) in eid and g.blue_edges[eid] > 0.0]
                        
                blues_inc = blues_incident_with_one
                blues_inc += blues_incident_with_two
                blues_inc += blues_incident_with_three
                
            b = random.getrandbits(1)
            if b:
                if len(reds_incident_with_one) > 0:
                    match += [random.choice(reds_incident_with_one)]
                if len(reds_incident_with_two) > 0:
                    match += [random.choice(reds_incident_with_two)]
                if len(reds_incident_with_three) > 0:
                    match += [random.choice(reds_incident_with_three)]
                temp = g.red_edges
            else:
                if len(blues_incident_with_one) > 0:
                    match += [random.choice(blues_incident_with_one)]
                if len(blues_incident_with_two) > 0:
                    match += [random.choice(blues_incident_with_two)]
                if len(blues_incident_with_three) > 0:
                    match += [random.choice(blues_incident_with_three)]
                temp = g.blue_edges
                
            self.assertTrue(g.chhk_colored_match(b, match))
                
            lefts = [nid for nid in g.left_nodes if \
                any([eid[0]==nid for eid in temp])]
            unmatched_lefts = [nid for nid in lefts if \
                not any([eid[0]==nid for eid in match])]
            rights = [nid for nid in g.right_nodes if \
                any([eid[1]==nid for eid in temp])]
            unmatched_rights = [nid for nid in rights if \
                not any([eid[1]==nid for eid in match])]
            
            self.assertEqual(g.chhk_colored_maximal_match(b, match), \
                not any([eid[0] in unmatched_lefts and eid[1] in unmatched_rights for eid in temp]))

    #  @ut.skip("Skipping test_d_parse")
    def test_d_parse(self):
        g = make_d_graph() 

        b = 1
        input_match = [((5, None), (9, None), None)]

        out = g._parse(b, input_match)
        (matched_lefts, matched_rights, unmatched_lefts, unmatched_rights, \
            apparent_color, non_match_edges) = out

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

    #  @ut.skip("Skipping test_d_clean")
    def test_d_clean(self):
        """ test_d_cleanup deterministically tests the cleanup function.
        """
        # Let's run some tests on cleanup using red matches to start.
        b = 1  # color

        # Test one: deterministic graph with edges (i, j) for i = 1, 2, ..., 5
        # and j = 6, 7, ..., 10 and color (i*j) % 2 and wt 20*(i-1) + 4*(j-5)
        g = make_d_graph()

        # Using the following:
        input_match = [((3, None), (7, None), None)]
        sp = [[((1, None), (9, None), None)], [((5, None), (9, None), None)]]
        shortest_paths = []
        for next_path in sp:
            sd = [eid for eid in next_path if eid not in input_match]
            sd += [eid for eid in input_match if eid not in sd]
            gain = sum(g.red_edges[eid] for eid in sd) 
            gain = gain - sum(g.red_edges[eid] for eid in input_match)
            shortest_paths += [(next_path, gain)]

        result = g._clean(b, shortest_paths, input_match)
        # print(result)
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
        input_match = [((5, None), (9, None), None)]
        sp = [[((1, None), (7, None), None)], [((3, None), (7, None), None)]]
        shortest_paths = []
        for next_path in sp:
            sd = [eid for eid in next_path if eid not in input_match]
            sd += [eid for eid in input_match if eid not in sd]
            gain = sum(g.red_edges[eid] for eid in sd) 
            gain = gain - sum(g.red_edges[eid] for eid in input_match)
            shortest_paths += [(next_path, gain)]
        result = g._clean(b, shortest_paths, input_match)
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
        input_match = [((1, None), (7, None), None)]
        sp = [[((3, None), (9, None), None)], [((5, None), (9, None), None)]]  
        shortest_paths = []
        for next_path in sp:
            sd = [eid for eid in next_path if eid not in input_match]
            sd += [eid for eid in input_match if eid not in sd]
            gain = sum(g.red_edges[eid] for eid in sd) 
            gain = gain - sum(g.red_edges[eid] for eid in input_match)
            shortest_paths += [(next_path, gain)]
        result = g._clean(b, shortest_paths, input_match)
        self.assertTrue(((1, None), (7, None), None) in result)
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(len(result) == 2)

        # Test four: Let's go again with a different first match
        g = make_d_graph()
        input_match = [((1, None), (7, None), None)]
        sp = [[((3, None), (9, None), None)], [((5, None), (9, None), None)]]  
        shortest_paths = []
        for next_path in sp:
            sd = [eid for eid in next_path if eid not in input_match]
            sd += [eid for eid in input_match if eid not in sd]
            gain = sum(g.red_edges[eid] for eid in sd) 
            gain = gain - sum(g.red_edges[eid] for eid in input_match)
            shortest_paths += [(next_path, gain)]
        result = g._clean(b, shortest_paths, input_match)
        self.assertTrue(((1, None), (7, None), None) in result)
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(len(result) == 2)

        # Test four: Let's go again with a maximal first match
        g = make_d_graph()
        input_match = [((1, None), (7, None), None), \
            ((5, None), (9, None), None)]  
        sp = [[((3, None), (7, None), None), ((1, None), (7, None), None), \
            ((1, None), (9, None), None), ((5, None), (9, None), None)], \
            [((3, None), (9, None), None), ((5, None), (9, None), None), \
            ((5, None), (7, None), None), ((1, None), (7, None), None)]]
        shortest_paths = []
        for next_path in sp:
            sd = [eid for eid in next_path if eid not in input_match]
            sd += [eid for eid in input_match if eid not in sd]
            gain = sum(g.red_edges[eid] for eid in sd) 
            gain = gain - sum(g.red_edges[eid] for eid in input_match)
            shortest_paths += [(next_path, gain)]
        # gain of first shortest path is
        #     20*(3-1) + 4*(7-6) - 20*(1-1) - 4*(7-6) + 20*(1-1) + 4*(9-6) - \
        #         20*(5-1) - 4*(9-6) = -40
        # gain of second shortest path is
        #     20*(3-1) + 4*(9-6) - 20*(5-1) - 4*(9-6) + 20*(5-1) + 4*(7-6) - \
        #         20*(1-1) + 4*(7-6) = +40
        
        result = g._clean(b, shortest_paths, input_match)
        # the result should be the symmetric difference between the input match
        # and the heaviest shortest path so result = [(3, 9), (5, 7)] which is 
        # also maximal but heavier!
        self.assertTrue(((3, None), (9, None), None) in result)
        self.assertTrue(((5, None), (7, None), None) in result)
        self.assertTrue(len(result) == 2)

        # Let's run some tests on cleanup using red matches to start.
        b = 0  # color

        g = make_d_graph()
        input_match = [((2, None), (8, None), None)]
        sp = [[((1, None), (6, None), None)], [((1, None), (10, None), None)],\
            [((3, None), (6, None), None)], [((3, None), (10, None), None)], \
            [((4, None), (6, None), None)], [((4, None), (7, None), None)], \
            [((4, None), (9, None), None)], [((4, None), (10, None), None)], \
            [((5, None), (6, None), None)], [((5, None), (10, None), None)]]
        shortest_paths = []
        for next_path in sp:
            sd = [eid for eid in next_path if eid not in input_match]
            sd += [eid for eid in input_match if eid not in sd]
            gain = sum(g.blue_edges[eid] for eid in sd) 
            gain = gain - sum(g.blue_edges[eid] for eid in input_match)
            shortest_paths += [(next_path, gain)]
        
        result = g._clean(b, shortest_paths, input_match)
        
        self.assertTrue(((2, None), (8, None), None) in result)
        self.assertTrue(((5, None), (10, None), None) in result)
        self.assertTrue(((4, None), (9, None), None) in result)
        self.assertTrue(((3, None), (6, None), None) in result)

        # This is a matching
        self.assertTrue(g.chhk_colored_match(b, result))
        # This is a maximal blue matching: no remaining blue edges have un-
        # matched endpoints.
        self.assertTrue(g.chhk_colored_maximal_match(b, result))
        # However, it's not maximum: nodes 1 and 7 could be matched, ie the 
        # dumb matching (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)
        self.assertTrue(len(result) == 4)

    # #  @ut.skip("Skipping test_d_extend")
    def test_d_extend(self):
        # print("BEGINNING TEST_D_EXTEND")
        # Recall we say a matching is MAXIMAL if no edge can be added to the 
        # matching without breaking the matching property... but we say a max-
        # imal matching is OPTIMAL if it has the heaviest weight.
        g = make_d_graph()
        self.assertTrue(len(g.red_edges) > 0)
        
        b = 1  
        
        ##########################################################
        # let's start with a trivial match

        result = g.xxtend(b)
        self.assertIsNot(result, None)
        # print("Result from calling .xxtend = result = ", result)
        
        # Since this does not include an input match is empty, all shortest 
        # paths are treated like having gain 1, so they are added to the
        # list greedily. Possible edges: (1, 7), (1, 9), (3, 7), (3, 9), 
        # (5, 7), (5, 9) have weightes 4, 12, 44, 52, 84, 92. So edge (5, 9)
        # is added first, which is vertex-disjoint with (1, 9) and (3, 9). Next
        # heaviest edge is (3, 7) with weight 44.
        self.assertIn(((5, None), (9, None), None), result)
        self.assertIn(((3, None), (7, None), None), result)
        self.assertEqual(len(result), 2)

        ##########################################################
        # let's use a maximal match (but sub-optimal)

        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (9, None), None)]  
        result = g.xxtend(b, input_match)
        self.assertEqual(input_match, result)

        ##########################################################
        # let's check we raise an exception

        # We start with a pair of multi-colored edges, which can't be a match.
        input_match = [((1, None), (7, None), None)]
        input_match += [((2, None), (6, None), None)]
        self.assertTrue(input_match is not None)
        with self.assertRaises(Exception):
            result = g.xxtend(b, input_match)

        # We'll also check a pair of adjacent edges.
        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (7, None), None)]
        self.assertTrue(input_match is not None)
        with self.assertRaises(Exception):
            result = g.xxtend(b, input_match)

        # Mixed case
        input_match = [((1, None), (7, None), None)]
        input_match += [((1, None), (6, None), None)]
        self.assertTrue(input_match is not None)
        with self.assertRaises(Exception):
            result = g.xxtend(b, input_match)

        ##########################################################
        # let's check that starting with a single edge works.

        input_match = [((1, None), (7, None), None)]
        result = g.xxtend(b, input_match)
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
        weight_dict = g.red_edges
        # b_test = 0
        # weight_dict = g.blue_edges
        sp = [[((1, None), (7, None), None)], [((5, None), (9, None), None)]]
        shortest_paths_test = []
        for next_path in sp:
            sd = [weight_dict[eid] for eid in next_path if \
                eid not in input_match]
            sd += [weight_dict[eid] for eid in input_match if \
                eid not in next_path]
            gain = sum(sd) - sum([weight_dict[eid] for eid in input_match])
            shortest_paths_test += [(next_path, gain)]
        # non_match_edges_test = [((1, None), (9, None), None), ((3, None), 
        # (7, None), None), ((3, None), (9, None), None), 
        # ((5, None), (7, None), None), ((5, None), (9, None), None)]
    
        result_test = g._clean(b_test, shortest_paths_test, input_match)
        # print("Cleaning results = ", result_test)

        # both should result in this:
        grnd_trth = [((1, None), (7, None), None)]
        grnd_trth += [((5, None), (9, None), None)]
        self.assertEqual(len(set(result)), len(result))
        self.assertEqual(len(set(result_test)), len(result_test))
        self.assertEqual(len(set(grnd_trth)), len(grnd_trth))
        for i in result:
            self.assertIn(i, result_test)
            self.assertIn(i, result_ground_truth)
            
        # Now the match is maximal, so if we go again we should get the same 
        # result back.
        input_match = result
        result = g.xxtend(b, input_match)
        self.assertEqual(input_match, result)
        
        ##########################################################
        # Let's go again with a different match.

        input_match = [((1, None), (9, None), None)]
        result = g.xxtend(b, input_match)

        # this should call _cleanup with the following:
        b_test = 1
        weight_dict = g.red_edges
        # b_test = 0
        # weight_dict = g.blue_edges
        sp = [[((3, None), (7, None), None)], [((5, None), (7, None), None)]]
        shortest_paths_test = []
        for next_path in sp:
            sd = [weight_dict[eid] for eid in next_path if \
                eid not in input_match]
            sd += [weight_dict[eid] for eid in input_match if \
                eid not in next_path]
            gain = sum(sd) - sum([weight_dict[eid] for eid in input_match])
            shortest_paths_test += [(next_path, gain)]
        # non_match_edges_test = [(1, 7), (3, 7), (3, 9), (5, 7), (5, 9)]
        result_test = g._clean(b_test, shortest_paths_test, input_match)

        # both should result in this:
        grnd_trth = [((1, None), (9, None), None)]
        grnd_trth += [((5, None), (7, None), None)]
        self.assertEqual(len(set(result)), len(result))
        self.assertEqual(len(set(result_test)), len(result_test))
        self.assertEqual(len(set(grnd_trth)), len(grnd_trth))
        for i in result:
            self.assertIn(i, result_test)
            self.assertIn(i, result_ground_truth)
            

    #  @ut.skip("Skipping test_dd_extend")
    def test_dd_extend(self):
        # Recall we say a matching is MAXIMAL if no edge can be added to the 
        # matching without breaking the matching property... but we say a max-
        # imal matching is OPTIMAL if it has the heaviest weight.
        g = make_d_graph()

        b = 1

        ##########################################################
        # let's start with a trivial match

        input_match = []
        result = g.xxtend(b, input_match)
        # Since this match is empty, all shortest paths are treated like having
        # gain 1, so they are added to the list greedily in terms of lexi-
        # cographic ordering!
        self.assertTrue(((5, None), (9, None), None) in result)
        self.assertTrue(((3, None), (7, None), None) in result)
        self.assertEqual(len(result), 2)

        ##########################################################
        # let's use a maximal match

        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (9, None), None)]  
        result = g.xxtend(b, input_match)
        self.assertEqual(input_match, result)

        ##########################################################
        # let's check we get None if we send a non-match in.

        # We start with a pair of multi-colored edges, which can't be a match.
        input_match = [((1, None), (7, None), None)]
        input_match += [((2, None), (6, None), None)]
        with self.assertRaises(Exception):
            result = g.xxtend(b, input_match)

        # We'll also check a pair of adjacent edges.
        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (7, None), None)]
        with self.assertRaises(Exception):
            result = g.xxtend(b, input_match)

        # Mixed case
        input_match = [((1, None), (7, None), None)]
        input_match += [((1, None), (6, None), None)]
        with self.assertRaises(Exception):
            result = g.xxtend(b, input_match)

        ##########################################################
        # let's check that starting with a single edge works.

        input_match = [((1, None), (7, None), None)]
        result = g.xxtend(b, input_match)

        # this should call _cleanup with the following:
        b_test = 1
        weight_dict = g.red_edges
        # b_test = 0
        # weight_dict = g.blue_edges
        sp = [[((3, None), (9, None), None)], [((5, None), (9, None), None)]]
        shortest_paths_test = []
        for next_path in sp:
            sd = [weight_dict[eid] for eid in next_path if \
                eid not in input_match]
            sd += [weight_dict[eid] for eid in input_match if \
                eid not in next_path]
            gain = sum(sd) - sum([weight_dict[eid] for eid in input_match])
            shortest_paths_test += [(next_path, gain)]
        # non_match_edges_test = [((1, None), (9, None), None),
        # ((3, None), (7, None), None), ((3, None), (9, None), None), 
        # ((5, None), (7, None), None), ((5, None), (9, None), None)]
        result_test = g._clean(b_test, shortest_paths_test, input_match)

        # both should result in this:
        result_ground_truth = [((1, None), (7, None), None), \
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
        # b_test = 0
        # weight_dict = g.blue_edges
        sp = [[((3, None), (7, None), None)], [((5, None), (7, None), None)]]
        for p in sp:
            for eid in p:
                assert eid in weight_dict
        shortest_paths_test = []
        for next_path in sp:
            sd = [weight_dict[eid] for eid in next_path if \
                eid not in input_match]
            sd += [weight_dict[eid] for eid in input_match if \
                eid not in next_path]
            gain = sum(sd) - sum([weight_dict[eid] for eid in input_match])
            shortest_paths_test += [(next_path, gain)]
        # non_match_edges_test = [((1, None), (7, None), None), \
        # ((3, None), (7, None), None), ((3, None), (9, None), None), \
        # ((5, None), (7, None), None), ((5, None), (9, None), None)]
        result_test = g._clean(b_test, shortest_paths_test, input_match)

        # both should result in this:
        result_ground_truth = [((1, None), (9, None), None)]
        result_ground_truth += [((5, None), (7, None), None)]
        self.assertEqual(result, result_test)
        self.assertEqual(result, result_ground_truth)
        self.assertEqual(result_test, result_ground_truth)

    #  @ut.skip("Skipping test_r_extend")
    def test_r_extend(self):
        g = make_r_graph()
        b = 1
        input_match = []

        self.assertTrue(g.chhk_colored_match(b, input_match))

        result = g.extend(b, input_match)
        self.assertTrue(g.chhk_colored_match(b, result))
        s = len(g.red_edges) == 0
        s = s or len(result) > len(input_match)
        self.assertTrue(s)

    #  @ut.skip("Skipping test_dd_boost")
    def test_dd_boost(self):
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
        
        self.assertIn(((2, None), (6, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((2, None), (6, None), None)],  24.0)
        self.assertIn(((2, None), (7, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((2, None), (7, None), None)],  28.0)
        self.assertIn(((2, None), (8, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((2, None), (8, None), None)],  32.0)
        self.assertIn(((2, None), (9, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((2, None), (9, None), None)],  36.0)
        self.assertIn(((2, None), (10, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((2, None), (10, None), None)],  40.0)
        
        self.assertIn( ((3, None), (6, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((3, None), (6, None), None)] , 44.0)
        self.assertIn( ((3, None), (8, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((3, None), (8, None), None)] , 52.0)
        self.assertIn( ((3, None), (10, None), None), g.blue_edges)
        self.assertEqual( g.blue_edges[((3, None), (10, None), None)] , 60.0)
        
        self.assertIn( ((4, None), (6, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((4, None), (6, None), None)] , 64.0)
        self.assertIn( ((4, None), (7, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((4, None), (7, None), None)] , 68.0)
        self.assertIn( ((4, None), (8, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((4, None), (8, None), None)] , 72.0)
        self.assertIn( ((4, None), (9, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((4, None), (9, None), None)] , 76.0)
        self.assertIn( ((4, None), (10, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((4, None), (10, None), None)] , 80.0)
        
        self.assertIn( ((5, None), (6, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((5, None), (6, None), None)] , 84.0)
        self.assertIn( ((5, None), (8, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((5, None), (8, None), None)] , 92.0)
        self.assertIn( ((5, None), (10, None), None) , g.blue_edges)
        self.assertEqual( g.blue_edges[((5, None), (10, None), None)] , 100.0)
        
        self.assertIn( ((1, None), (7, None), None) , g.red_edges)
        self.assertEqual( g.red_edges[((1, None), (7, None), None)] , 7.0)
        self.assertIn( ((1, None), (9, None), None) , g.red_edges)
        self.assertEqual( g.red_edges[((1, None), (9, None), None)] , 17.0)
        self.assertIn( ((3, None), (7, None), None) , g.red_edges)
        self.assertEqual( g.red_edges[((3, None), (7, None), None)] , 49.0)
        self.assertIn( ((3, None), (9, None), None) , g.red_edges)
        self.assertEqual( g.red_edges[((3, None), (9, None), None)] , 55.0)
        self.assertIn( ((5, None), (7, None), None) , g.red_edges)
        self.assertEqual( g.red_edges[((5, None), (7, None), None)] , 88.0)
        self.assertIn( ((5, None), (9, None), None) , g.red_edges)
        self.assertEqual( g.red_edges[((5, None), (9, None), None)] , 96.0)
        
        # The graph from make_d_graph has at least two matchings with equal 
        # weight. We tweaked that graph for this example to ensure that the 
        # edge case isn't relevant to our test.
        b = 1
        input_match = [((1, None), (7, None), None)]
        input_match += [((3, None), (9, None), None)]

        self.assertTrue(g.chhk_colored_maximal_match(b, input_match))
        result = g.boost(b, input_match)
        # print("RESULT FROM TEST_DD_BOOST = " + str(result))
        self.assertEqual(len(result), 2)
        self.assertTrue(((1, None), (9, None), None) in result)
        self.assertTrue(((3, None), (7, None), None) in result)

    #  @ut.skip("Skipping test_d_optimize")
    def test_d_optimize(self):
        ''' This is an integration test, combines the functionalities. '''
        g = make_d_graph()
        b = 1
        final_answer = g.optimize(b)
        self.assertTrue(((5, None), (9, None), None) in final_answer)
        self.assertTrue(((3, None), (7, None), None) in final_answer)
        self.assertEqual(len(final_answer), 2)


tests = [TestBipartiteGraph]
for test in tests:
    ttr = ut.TextTestRunner(verbosity=2, failfast=True)
    ttr.run(ut.TestLoader().loadTestsFromTestCase(test))
