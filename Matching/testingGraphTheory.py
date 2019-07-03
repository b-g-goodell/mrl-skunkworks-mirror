import unittest
import random
from graphtheory import disjoint
from graphtheory import symdif
from graphtheory import *

class TestSymDif(unittest.TestCase):
    """ TestSymDif tests both disjoint() and symdif() """
    def test_disjoint(self):
        """ testing disjointness function disjoint() """
        x = [1,2,3]
        y = [5,6,7]
        self.assertTrue(disjoint(x,y))
        x = [1,2,3]
        y = [3,4,5]
        self.assertFalse(disjoint(x,y))

    def test(self):
        """ testing symdif """
        x = [1, 2, 3]
        y = [5, 6, 7]
        self.assertTrue(x + y == symdif(x, y))

        y.append(3)
        y.append(2)
        x.append(5)
        x.append(6)
        self.assertTrue(1 in symdif(x, y))
        self.assertTrue(7 in symdif(x, y))
        self.assertTrue(len(symdif(x, y)) == 2)

        x.append(0)
        self.assertTrue(0 in symdif(x, y))
        self.assertTrue(1 in symdif(x, y))
        self.assertTrue(7 in symdif(x, y))
        self.assertTrue(len(symdif(x, y)) == 3)

        x = random.sample(range(100), 60)
        y = random.sample(range(100), 60)  # Pigeonhole principle implies at least 20 collisions
        z = symdif(x, y)
        self.assertTrue(len(z) >= 20)
        zp = [w for w in x if w not in y] + [w for w in y if w not in x]
        self.assertEqual(len(z), len(zp))
        for i in z:
            self.assertTrue(i in zp)
            self.assertTrue((i in x and i not in y) or (i in y and i not in x))

class TestBipartiteGraph(unittest.TestCase):
    """ TestBipartiteGraph tests BipartiteGraph objects """
    def test_init(self):
        """ test_init tests initialization of a BipartiteGraph """
        # self.data = par['data']   # str
        # self.ident = par['ident']  # str
        # self.left_nodes = {}.update(par['left'])
        # self.right_nodes = {}.update(par['right'])
        # self.red_edge_weights = {}.update(par['red'])
        # self.blue_edge_weights = {}.update(par['blue'])
        par = {'data': None, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        
        n = 15  # nodeset size
        k = 5  # neighbor size

        ct = 0
        while len(g.left_nodes) < n:
            # print("Adding new leftnode: ", len(g.left_nodes))
            while ct in g.left_nodes or ct in g.right_nodes:
                ct += 1
            numleft = len(g.left_nodes)
            rejected = g.add_left_node(ct)
            newnumleft = len(g.left_nodes)
            self.assertTrue(not rejected and newnumleft - numleft == 1)

        while len(g.right_nodes) < n:
            # print("Adding new rightnode: ", len(G.right))
            while ct in g.right_nodes or ct in g.left_nodes:
                ct += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(ct)
            newnumright = len(g.right_nodes)
            self.assertTrue(not rejected and newnumright - numright == 1)

        numRedEdges = len(g.red_edge_weights)
        leftnodekeys = list(g.left_nodes.keys())
        rightnodekeys = list(g.right_nodes.keys())
        for i in range(n,2*n):
            idxs = random.sample(range(n), k)
            assert len(idxs) == k
            for j in idxs:
                eid = (j,i)
                rejected = g.add_red_edge(eid, 1.0)
                assert not rejected

        self.assertTrue(len(g.red_edge_weights) - numRedEdges == k*n)
        self.assertTrue(len(g.left_nodes) + len(g.right_nodes) == 2*n)
        self.assertTrue(len(g.red_edge_weights) == k*n)

    def test_add_left_node(self):
        """ test_add_left_node tests adding a left-node."""
        # Pick a random graph identity
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 25
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)

    def test_add_right_node(self):
        """ test_add_right_node tests adding a right-node. """
        # Pick a random graph identity
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 25
        nodect = 0
        while len(g.right_nodes) < n:
            while nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)

    def test_add_red_edge(self):
        """ test_add_red_edge tests adding a red edge."""
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 25
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)

        # print(g.left_nodes.keys())
        # print(g.right_nodes.keys())
        # print(g.red_edge_weights.keys())
        # print(g.blue_edge_weights.keys())

        i = random.randint(0,24)
        j = random.randint(25,49)
        self.assertTrue((i,j) not in g.red_edge_weights)
        self.assertTrue((i,j) not in g.blue_edge_weights)
        self.assertEqual(len(g.red_edge_weights),0)
        rejected = g.add_red_edge((i,j), -1.0)
        self.assertFalse(rejected)
        self.assertEqual(len(g.red_edge_weights),1)
        self.assertTrue((i,j) in g.red_edge_weights)
        self.assertEqual(g.red_edge_weights[(i,j)],-1.0)

    def test_add_blue_edge(self):
        """ test_add_blue_edge tests adding a blue edge. """
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 25
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)

        i = random.randint(0,24)
        j = random.randint(25,49)
        self.assertTrue((i,j) not in g.red_edge_weights)
        self.assertTrue((i,j) not in g.blue_edge_weights)
        self.assertEqual(len(g.blue_edge_weights),0)
        rejected = g.add_blue_edge((i,j), -1.0)
        self.assertFalse(rejected)
        self.assertEqual(len(g.blue_edge_weights),1)
        self.assertTrue((i,j) in g.blue_edge_weights)
        self.assertEqual(g.blue_edge_weights[(i,j)],-1.0)

    def test_del_edge(self):
        """ test_del_edge tests deletion of an edge. """
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 2
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)
        for i in range(n):
            for j in range(n,2*n):
                g.add_red_edge((i,j),0.0)

        # print("LEFT NODES = ", list(g.left_nodes.keys()))
        # print("RIGHT NODES = ", list(g.right_nodes.keys()))
        # print("RED EDGES = ", list(g.red_edge_weights.keys()))
        # print("BLUE EDGES = ", list(g.blue_edge_weights.keys()))
        self.assertEqual(len(g.red_edge_weights),4)
        i = random.randint(0,n-1)
        j = random.randint(n,2*n-1)
        # print("(i,j) = ", (i,j))
        self.assertTrue((i,j) in g.red_edge_weights)
        rejected = g.del_edge((i,j))
        self.assertFalse(rejected)
        self.assertEqual(len(g.red_edge_weights),3)

    def test_check_red_match(self):
        """ test_check_red_match tests the function that checks whether a given set of edges is a match of red edges. """
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 2
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)
        for i in range(n):
            for j in range(n,2*n):
                g.add_red_edge((i,j),0.0)

        self.assertTrue(g._check_red_match([]))

        self.assertTrue(g._check_red_match([(0,2)]))
        self.assertTrue(g._check_red_match([(0,3)]))
        self.assertTrue(g._check_red_match([(1,2)]))
        self.assertTrue(g._check_red_match([(1,3)]))

        self.assertFalse(g._check_red_match([(0,2),(0,3)]))
        self.assertTrue(g._check_red_match([(0,2),(1,3)]))
        self.assertFalse(g._check_red_match([(0,2),(1,2)]))
        self.assertFalse(g._check_red_match([(0,3),(1,3)]))
        self.assertTrue(g._check_red_match([(0,3),(1,2)]))
        self.assertFalse(g._check_red_match([(1,2),(1,3)]))

        self.assertFalse(g._check_red_match([(0,2),(0,3),(1,3)]))
        self.assertFalse(g._check_red_match([(0,2),(0,3),(1,2)]))
        self.assertFalse(g._check_red_match([(0,2),(1,3),(1,2)]))
        self.assertFalse(g._check_red_match([(0,3),(1,3),(1,2)]))

        self.assertFalse(g._check_red_match([(0,2),(0,3),(1,2),(1,3)]))

    def test_check_blue_match(self):
        """ test_check_blue_match tests the function that checks whether a set of edges is a match of blue edges. This is an unnecessary test because we do not use matchings of blue edges at all. """
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 2
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)
        for i in range(n):
            for j in range(n,2*n):
                g.add_blue_edge((i,j),0.0)

        self.assertTrue(g._check_blue_match([]))

        self.assertTrue(g._check_blue_match([(0,2)]))
        self.assertTrue(g._check_blue_match([(0,3)]))
        self.assertTrue(g._check_blue_match([(1,2)]))
        self.assertTrue(g._check_blue_match([(1,3)]))

        self.assertTrue(g._check_blue_match([(0,2),(1,3)]))
        self.assertTrue(g._check_blue_match([(0,3),(1,2)]))
        self.assertFalse(g._check_blue_match([(0,2),(0,3)]))
        self.assertFalse(g._check_blue_match([(0,2),(1,2)]))
        self.assertFalse(g._check_blue_match([(0,3),(1,3)]))
        self.assertFalse(g._check_blue_match([(1,2),(1,3)]))

        self.assertFalse(g._check_blue_match([(0,2),(0,3),(1,3)]))
        self.assertFalse(g._check_blue_match([(0,2),(0,3),(1,2)]))
        self.assertFalse(g._check_blue_match([(0,2),(1,3),(1,2)]))
        self.assertFalse(g._check_blue_match([(0,3),(1,3),(1,2)]))

        self.assertFalse(g._check_blue_match([(0,2),(0,3),(1,2),(1,3)]))

    def test_add_delete_together(self):
        """ test_add_delete_together tests addition and deletion of nodes and edges."""

        # Create graph
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 25 # Add n nodes to each side
        k = 5 # make a k-regular graph; k = number of red edges adjacent to right nodes, i.e. ring size
        
        self.assertEqual(len(g.red_edge_weights), 0)
        self.assertEqual(len(g.blue_edge_weights), 0)
        self.assertEqual(len(g.left_nodes), 0)
        self.assertEqual(len(g.right_nodes), 0)

        # Add left nodes
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)

        self.assertEqual(len(g.red_edge_weights), 0)
        self.assertEqual(len(g.blue_edge_weights), 0)
        self.assertEqual(len(g.left_nodes), n)
        self.assertEqual(len(g.right_nodes), 0)

        # Add right nodes
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)

        self.assertEqual(len(g.red_edge_weights), 0)
        self.assertEqual(len(g.blue_edge_weights), 0)
        self.assertEqual(len(g.left_nodes), n)
        self.assertEqual(len(g.right_nodes), n)
        
        # Set first signature (right node with id n) to be linked by red edges with first k keys
        # (left nodes with ids 0, ..., k-1).
        rejected = None
        for i in range(k):
            g.add_red_edge((i, n), 0.0)
            self.assertTrue((i, n) in g.red_edge_weights)
            self.assertEqual(g.red_edge_weights[(i,n)], 0.0)

        self.assertEqual(len(g.red_edge_weights), k)
        self.assertEqual(len(g.blue_edge_weights), 0)
        self.assertEqual(len(g.left_nodes), n)
        self.assertEqual(len(g.right_nodes), n)
        
        # Pick a random signature (right-node) with no adjacent edges (any with ids n+1, n+2, ..., 2n-1 are fair), pick
        # k random keys (left-nodes), add red edges between them. For simple test, we do this uniformly.
        leftnodekeys = list(g.left_nodes.keys())
        rightnodekeys = list(g.right_nodes.keys())
        r = random.choice(range(n+1, 2*n))
        l = random.sample(leftnodekeys, k)

        # Assemble identities of the edges to be added for this signature
        self.assertTrue(len(l) == k)
        edge_idents = [(lefty, r) for lefty in l]
        self.assertTrue(len(edge_idents) == k)
        
        for i in range(k):
            # For each ring member, create an edge, add it, and check that the edge was not rejected.
            self.assertTrue(edge_idents[i] not in g.red_edge_weights)
            self.assertTrue(edge_idents[i] not in g.blue_edge_weights)
            rejected = g.add_red_edge(edge_idents[i],0.0)
            self.assertTrue(edge_idents[i] in g.red_edge_weights)
            self.assertFalse(rejected)

        self.assertEqual(len(g.red_edge_weights), 2*k)
        self.assertEqual(len(g.blue_edge_weights), 0)
        self.assertEqual(len(g.left_nodes), n)
        self.assertEqual(len(g.right_nodes), n)

        # Pick a random signature and key that are not connected with a red-edge
        r = random.choice(rightnodekeys)
        l = random.choice(leftnodekeys) 
        while (l, r) in g.red_edge_weights:
            r = random.choice(rightnodekeys)
            l = random.choice(leftnodekeys)
            
        # Create a blue-edge for this pair
        eid = (l, r)
        # Include the blue edge in the graph, verify not rejected, verify there is only one blue-edge.
        rejected = g.add_blue_edge(eid,0.0)
        self.assertFalse(rejected)

        self.assertEqual(len(g.red_edge_weights), 2*k)
        self.assertEqual(len(g.blue_edge_weights), 1)
        self.assertEqual(len(g.left_nodes), n)
        self.assertEqual(len(g.right_nodes), n)

        # Find the next node identity that is not yet in the graph
        while nodect in g.left_nodes or nodect in g.right_nodes:
            nodect += 1
        # Create that left node, add it to the left
        left_ident = nodect
        rejected = g.add_left_node(left_ident)

        self.assertFalse(rejected)
        self.assertEqual(len(g.red_edge_weights), 2*k)
        self.assertEqual(len(g.blue_edge_weights), 1)
        self.assertEqual(len(g.left_nodes), n+1)
        self.assertEqual(len(g.right_nodes), n)

        # Pick the next node identity that doesn't yet exist.
        while nodect in g.right_nodes or nodect in g.left_nodes:
            nodect += 1
        # Create that node and add it to the right.
        right_ident = nodect
        rejected = g.add_right_node(right_ident)

        self.assertFalse(rejected)
        self.assertEqual(len(g.red_edge_weights), 2*k)
        self.assertEqual(len(g.blue_edge_weights), 1)
        self.assertEqual(len(g.left_nodes), n+1)
        self.assertEqual(len(g.right_nodes), n+1)

        # Now create either a red-edge or an blue-edge connecting l and r with 50% probability of each
        b = random.choice([0, 1])  # Pick a random bit
        # Include edge as red-edge if b=0, blue-edge if b=1, and mark the edge as rejected if attempt fails
        rejected = None
        if b == 0:
            rejected = g.add_red_edge((left_ident, right_ident), 0.0)
        elif b == 1:
            rejected = g.add_blue_edge((left_ident, right_ident), 0.0)

        self.assertTrue(rejected is not None)  # This should always be true since we've covered all possibilities.
        self.assertFalse(rejected)  # Check that the new edge was not rejected
        # Make sure that we have the right number of total edges and verify that the edge landed in the correct place
        self.assertEqual(len(g.red_edge_weights), 2*k+(1-b))
        self.assertEqual(len(g.blue_edge_weights), 1+b)
        self.assertEqual(len(g.left_nodes), n+1)
        self.assertEqual(len(g.right_nodes), n+1)
        if b == 0:
            self.assertEqual(len(g.red_edge_weights), 2*k+1)
            self.assertEqual(len(g.blue_edge_weights), 1)
        elif b == 1:
            self.assertEqual(len(g.red_edge_weights), 2*k)
            self.assertEqual(len(g.blue_edge_weights), 2)
        else:
            self.assertTrue(rejected)

        # Pick a random red-edge
        ek = random.choice(list(g.red_edge_weights.keys()))
        self.assertTrue(ek in g.red_edge_weights)
        # Attempt to delete it.
        rejected = g.del_edge(ek)
        # Check that deletion worked
        self.assertFalse(rejected)
        self.assertEqual(len(g.red_edge_weights) + len(g.blue_edge_weights), 2*k+1)

        # Pick a random leftnode
        mk = random.choice(list(g.left_nodes.keys()))
        # Collect the edge identities of edges that will be delted if this node is deleted
        edges_lost = len([eid for eid in g.red_edge_weights if eid[0]==mk])
        # Attempt to delete the node
        rejected = g.del_node(mk)
        self.assertFalse(rejected)
        self.assertTrue(len(g.left_nodes) + len(g.right_nodes) == 2*n+1)
        self.assertEqual(len(g.red_edge_weights) + len(g.blue_edge_weights), 2*k+1 - edges_lost)

    def test_bfs_simple(self):
        """ test_bfs_simple is a simple test of redd_bfs. """
        # 0 ---- 2
        #    /  
        # 1 /--- 3
        # Edges (0,4), (1,4), (2,4), (2,5), (2,6), (3,7)
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': {}, 'right': {}, 'red_edges': {}, 'blue_edges': {}}
        g = BipartiteGraph(par)
        n = 2 # Add n nodes to each side
        nodect = 0
        while len(g.left_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numleft = len(g.left_nodes)
            # print("\n",nodect, g.left_nodes)
            rejected = g.add_left_node(nodect)
            self.assertFalse(rejected)
            newnumleft = len(g.left_nodes)
            self.assertTrue(newnumleft - numleft == 1)
        while len(g.right_nodes) < n:
            while nodect in g.left_nodes or nodect in g.right_nodes:
                nodect += 1
            numright = len(g.right_nodes)
            rejected = g.add_right_node(nodect)
            self.assertFalse(rejected)
            newnumright = len(g.right_nodes)
            self.assertTrue(newnumright - numright == 1)

        self.assertTrue(0 in g.left_nodes)
        self.assertTrue(1 in g.left_nodes)
        self.assertEqual(len(g.left_nodes), 2)

        self.assertTrue(2 in g.right_nodes)
        self.assertTrue(3 in g.right_nodes)
        self.assertEqual(len(g.right_nodes), 2)

        g.add_red_edge((0,2), 0.0)
        g.add_red_edge((1,2), 0.0)
        g.add_red_edge((1,3), 0.0)

        self.assertTrue((0,2) in g.red_edge_weights)
        self.assertTrue((1,2) in g.red_edge_weights)
        self.assertTrue((1,3) in g.red_edge_weights)
        self.assertEqual(len(g.red_edge_weights), 3)

        match = list()
        match.append((1,2))

        # print("\n\n======\n\n")
        # print("Red edge weights = ", g.red_edge_weights)
        # print("Match = ", match)
        results = g.redd_bfs(match)
        # print("Results = ", results)
        self.assertEqual(results, [[(0,2),(1,2),(1,3)]])

    def test_bfs_full(self):
        """ test_bfs_full is a less-simple test of bfs_red. """
        # 0 ---- 4
        #    / /
        # 1 / // 5
        #    //
        # 2 /--- 6
        # 3 ---- 7
        # Edges (0,4), (1,4), (2,4), (2,5), (2,6), (3,7)

        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
        g = BipartiteGraph(par)

        # 4 nodes on each side
        n = 4
        ct = 0
        # Add left nodes
        while len(g.left_nodes) < n:
            while ct in g.left_nodes:
                ct += 1
            g.add_left_node(ct)
            ct += 1
        # Add right nodes
        while len(g.right_nodes) < n:
            while ct in g.right_nodes:
                ct += 1
            g.add_right_node(ct)
            ct += 1

        # Check left keys have 0, 1, 2, 3.
        leftkeys = list(g.left_nodes.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        # Check that right keys have 4, 5, 6, 7.
        rightkeys = list(g.right_nodes.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        # Edges to be added
        edge_idents = [(0, 4), (1, 4), (2, 4), (2, 5), (2, 6), (3, 7)]
          
        for ident in edge_idents:
            g.add_red_edge(ident, 0.0)

        match = [(2,4)]
        results = g.redd_bfs(match)
        # print("\n\n====\n\n results = ", results, type(results), "\n\n====\n\n")
        self.assertEqual(results, [[(3,7)]])

    def test_bfs_fourshortest(self):
        """ test_bfs_full is a less-simple test of bfs_red. """
        # 0 ---- 3
        #    / /
        # 1 / // 4
        #    //
        # 2 /--- 5
        # Edges (0,3), (1,3), (2,3), (2,4), (2,5)
        # Initial match: [(2,3)]

        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
        g = BipartiteGraph(par)

        # 4 nodes on each side
        n = 3
        ct = 0
        # Add left nodes
        while len(g.left_nodes) < n:
            while ct in g.left_nodes:
                ct += 1
            g.add_left_node(ct)
            ct += 1
        # Add right nodes
        while len(g.right_nodes) < n:
            while ct in g.right_nodes:
                ct += 1
            g.add_right_node(ct)
            ct += 1

        # Check left keys have 0, 1, 2
        leftkeys = list(g.left_nodes.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        # Check that right keys have 3, 4, 5.
        rightkeys = list(g.right_nodes.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        # Edges to be added
        edge_idents = [(0,3), (1,3), (2,3), (2,4), (2,5)]
          
        for ident in edge_idents:
            g.add_red_edge(ident, 0.0)
        self.assertEqual(len(g.red_edge_weights),5)

        match = [(2,3)]
        results = g.redd_bfs(match)
        # print("\n\n====\n\n results = ", results, type(results), "\n\n====\n\n")
        self.assertTrue([(0,3), (2,3), (2,4)] in results)
        self.assertTrue([(0,3), (2,3), (2,5)] in results)
        self.assertTrue([(1,3), (2,3), (2,4)] in results)
        self.assertTrue([(1,3), (2,3), (2,5)] in results)

    def test_get_augmenting_path(self):
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left_nodes) < n:
            while ct in g.left_nodes:
                ct += 1
            g.add_left_node(ct)
            ct += 1
        while len(g.right_nodes) < n:
            while ct in g.right_nodes:
                ct += 1
            g.add_right_node(ct)
            ct += 1

        leftkeys = list(g.left_nodes.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right_nodes.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents = [(0, 3), (1, 3), (2, 3), (2, 4), (2, 5)]
        for ident in edge_idents:
            g.add_red_edge(ident, 0.0)

        match = [(2,3)]
        results = g.get_augmenting_red_paths(match)
        # line = [[e.ident for e in p] for p in results]
        # print("Results from _get_augmenting_paths = " + str(line))
        self.assertTrue(len(results) == 1) # one resulting path
        self.assertTrue(len(results[0]) == 3) # with length 3
        self.assertTrue(results[0][0][0] == 0 or results[0][0][0] == 1)
        self.assertEqual(results[0][0][1], 3)
        self.assertEqual(results[0][1][0], 2)
        self.assertEqual(results[0][1][1], 3)
        self.assertTrue(results[0][2][0], 2)
        self.assertTrue(results[0][2][1] == 4 or results[0][2][1] == 5)

    def test_max_matching_weightless(self):
        # print("Beginning test_max_matching)")
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left_nodes) < n:
            while ct in g.left_nodes:
                ct += 1
            g.add_left_node(ct)
            ct += 1
        while len(g.right_nodes) < n:
            while ct in g.right_nodes:
                ct += 1
            g.add_right_node(ct)
            ct += 1

        leftkeys = list(g.left_nodes.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right_nodes.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents = [(0, 3), (1, 3), (2, 3), (2, 4), (2, 5)]
        for eid in edge_idents:
            g.add_red_edge(eid, 0.0)

        match = [(2,3)]
        results = g.get_max_red_matching(match)
        self.assertTrue((0, 3) in results or (1, 3) in results)
        self.assertTrue((2, 4) in results or (2, 5) in results)
        self.assertFalse((2, 3) in results)
        self.assertEqual(len(results), 2)

    def test_get_optimal_red_matching(self):
        # print("Beginning test_max_matching)")
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left_nodes) < n:
            while ct in g.left_nodes:
                ct += 1
            g.add_left_node(ct)
            ct += 1
        while len(g.right_nodes) < n:
            while ct in g.right_nodes:
                ct += 1
            g.add_right_node(ct)
            ct += 1

        leftkeys = list(g.left_nodes.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right_nodes.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents_and_weights = [[(0, 3), 1], [(1, 3), 2], [(2, 3), 3], [(2, 4), 4], [(2, 5), 5]]
        for ident in edge_idents_and_weights:
            eid = ident[0]
            w = ident[1]
            g.add_red_edge(eid, w)

        match = [(2,3)]
        results = g.get_optimal_red_matching(match)
        self.assertEqual(len(results), 2)
        self.assertTrue((1,3) in results)
        self.assertTrue((2,5) in results)

        # line = [e.ident for e in results]
        # print("Maximal matching is " + str(line))

    def test_get_alt_red_paths_with_pos_gain(self):
        ''' test_get_alt_red_paths_with_pos_gain tests whether, given a maximal match on a weighted graph,
        whether we correctly extract another maximal match with higher weight'''
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
        g = BipartiteGraph(par) # want four left nodes and two right nodes.
        n = 4
        k = 3
        ct = 0
        while len(g.left_nodes) < 2*n:
            while ct in g.left_nodes or ct in g.right_nodes:
                ct += 1
            g.add_left_node(ct)
            ct += 1
        while len(g.right_nodes) < n:
            while ct in g.right_nodes or ct in g.left_nodes:
                ct += 1
            g.add_right_node(ct)
            ct += 1

        leftkeys = list(g.left_nodes.keys())
        for i in range(2*n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), 2*n)

        rightkeys = list(g.right_nodes.keys())
        for i in range(2*n, 3*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        weids = []
        weids.append([(0, 8), 7])
        weids.append([(1, 8), 6])
        weids.append([(2, 8), 5])
        weids.append([(2, 9), 6])
        weids.append([(3, 9), 7])
        weids.append([(4, 9), 8])
        weids.append([(3, 10), 5])
        weids.append([(4, 10), 6])
        weids.append([(5, 10), 7])
        weids.append([(5, 11), 1])
        weids.append([(6, 11), 2])
        weids.append([(7, 11), 3])

        eids = []
        for ident in weids:
            rejected = g.add_red_edge(ident[0], ident[1])
            assert not rejected
            eids.append(ident[0])

        bad_match = [(1,8), (2, 9), (3, 10), (5, 11)]
        self.assertTrue(g._check_red_match(bad_match))

        matched_rights = [eid[1] for eid in bad_match]
        self.assertEqual(len(matched_rights), len(g.right_nodes))

        results = g.get_shortest_improving_paths_wrt_max_match(bad_match)
        results = sorted(results, key=lambda x:x[1], reverse=True)
        print(results)
        self.assertTrue(([(0, 8), (1, 8)], 1) in results)
        self.assertTrue(([(4, 9), (2, 9)], 2) in results)
        self.assertTrue(([(4, 10), (3, 10)], 3) in results)
        self.assertTrue(([(6, 11), (5, 11)], 1) in results)
        self.assertTrue(([(7, 11), (5, 11)], 2) in results)

    def test_get_opt_matching(self, sample_size=10**3, verbose=False):
        ''' test_get_opt_matching tests finding an optimal matching for a COMPLETE n=4 bipartite graph with random
        edge weights. '''
        for tomato in range(sample_size):
            # print("Beginning test_max_matching)")
            s = random.random()
            x = str(hash(str(1.0) + str(s)))
            par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'red_edges': [], 'blue_edges': []}
            g = BipartiteGraph(par)
            n = 4
            ct = 0
            while len(g.left_nodes) < n:
                while ct in g.left_nodes:
                    ct += 1
                g.add_left_node(ct)
                ct += 1
            while len(g.right_nodes) < n:
                while ct in g.right_nodes:
                    ct += 1
                g.add_right_node(ct)
                ct += 1

            leftkeys = list(g.left_nodes.keys())
            for i in range(n):
                self.assertTrue(i in leftkeys)
            self.assertEqual(len(leftkeys), n)

            rightkeys = list(g.right_nodes.keys())
            for i in range(n, 2*n):
                self.assertTrue(i in rightkeys)
            self.assertEqual(len(rightkeys), n)

            # print(g.left_nodes)
            # print(g.right_nodes)
            for i in range(n):
                for j in range(n,2*n):
                    g.add_red_edge((i,j), random.random())
            # print(g.in_edges)

            results = g.get_optimal_red_matching()
            readable_results = []
            for eid in results:
                assert eid in g.in_edges or eid in g.out_edges and not (eid in g.in_edges and eid in g.out_edges)
                if eid in g.in_edges:
                    readable_results.append((g.in_edges[eid].weight, eid))
                elif eid in g.out_edges:
                    readable_results.append((g.out_edges[eid].weight, eid))
            net_weight = sum(x[0] for x in readable_results)
            if verbose:
                print("optmatch results = " + str(readable_results))

            # There are 24 possible matchings, we are going to compute the cumulative weight of each.
            all_m = list()
            all_m.append([(0, 4), (1, 5), (2, 6), (3, 7)])
            all_m.append([(0, 4), (1, 5), (2, 7), (3, 6)])
            all_m.append([(0, 4), (1, 6), (2, 5), (3, 7)])
            all_m.append([(0, 4), (1, 6), (2, 7), (3, 5)])
            all_m.append([(0, 4), (1, 7), (2, 5), (3, 6)])
            all_m.append([(0, 4), (1, 7), (2, 6), (3, 5)])
            all_m.append([(0, 5), (1, 4), (2, 6), (3, 7)])
            all_m.append([(0, 5), (1, 4), (2, 7), (3, 6)])
            all_m.append([(0, 5), (1, 6), (2, 4), (3, 7)])
            all_m.append([(0, 5), (1, 6), (2, 7), (3, 4)])
            all_m.append([(0, 5), (1, 7), (2, 4), (3, 6)])
            all_m.append([(0, 5), (1, 7), (2, 6), (3, 4)])
            all_m.append([(0, 6), (1, 4), (2, 5), (3, 7)])
            all_m.append([(0, 6), (1, 4), (2, 7), (3, 5)])
            all_m.append([(0, 6), (1, 5), (2, 4), (3, 7)])
            all_m.append([(0, 6), (1, 5), (2, 7), (3, 4)])
            all_m.append([(0, 6), (1, 7), (2, 4), (3, 5)])
            all_m.append([(0, 6), (1, 7), (2, 5), (3, 4)])
            all_m.append([(0, 7), (1, 4), (2, 5), (3, 6)])
            all_m.append([(0, 7), (1, 4), (2, 6), (3, 5)])
            all_m.append([(0, 7), (1, 5), (2, 4), (3, 6)])
            all_m.append([(0, 7), (1, 5), (2, 6), (3, 4)])
            all_m.append([(0, 7), (1, 6), (2, 4), (3, 5)])
            all_m.append([(0, 7), (1, 6), (2, 5), (3, 4)])

            weighted_matches = []
            for m in all_m:
                this_weight = 0.0
                for eid in m:
                    this_weight += g.in_edges[eid].weight
                weighted_matches.append((this_weight, m))
            if verbose:
                print("==Weight==\t\t\t==Match==")
                for wm in weighted_matches:
                    print(str(wm[0]) + "\t" + str(wm[1]))
                    self.assertTrue(wm[0] <= net_weight)


tests = [TestSymDif, TestBipartiteGraph]
for test in tests:
    unittest.TextTestRunner(verbosity=2, failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
