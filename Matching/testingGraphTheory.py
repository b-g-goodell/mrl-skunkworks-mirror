import unittest
from graphtheory import *


class TestSymDif(unittest.TestCase):
    def test(self):
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


class TestNode(unittest.TestCase):
    def test_init(self):
        par = {'data': None, 'ident': None}
        nelly = Node(par)
        self.assertTrue(nelly.data is None)
        self.assertTrue(nelly.ident is None)
        self.assertTrue(nelly.in_edges == [])
        self.assertTrue(nelly.out_edges == [])

        par = {'data': 1.0, 'ident': 0}
        nelly = Node(par)
        self.assertFalse(nelly.data is None)
        self.assertEqual(nelly.data, 1.0)
        self.assertEqual(nelly.ident, 0)

    def test_add_edge(self):
        par = {'data': 1.0, 'ident': 0}
        n = Node(par)

        par = {'data': 2.0, 'ident': 1}
        m = Node(par)

        par = {'data': 3.0, 'ident': (0, 1), 'left': n, 'right': m, 'weight': 2.3}
        e = Edge(par)

        n.add_in_edge(e)
        m.add_in_edge(e)

        self.assertTrue(len(n.in_edges) == 1)
        self.assertTrue(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)

    def test_del_edge(self):
        x = 0
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        x = 1
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        x = (0, 1)
        par = {'data': 3.0, 'ident': x, 'left': n, 'right': m, 'weight': 2.3}
        e = Edge(par)
        n.add_in_edge(e)
        m.add_in_edge(e)

        self.assertTrue(len(n.in_edges) == 1)
        self.assertTrue(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)

        rejected = n.del_edge(e)
        self.assertFalse(rejected)
        self.assertTrue(len(n.in_edges) == 0)
        self.assertFalse(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)

        rejected = m.del_edge(e)
        self.assertFalse(rejected)
        self.assertTrue(len(n.in_edges) == 0)
        self.assertFalse(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 0)
        self.assertFalse(e in m.in_edges)


class TestEdge(unittest.TestCase):
    def test_init(self):
        x = 0
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        x = 1
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        x = (0, 1)
        par = {'data': 3.0, 'ident': x, 'left': n, 'right': m, 'weight': 2.3}
        e = Edge(par)
        n.add_in_edge(e)
        m.add_in_edge(e)

        self.assertTrue(len(n.in_edges) == 1)
        self.assertTrue(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)
        self.assertTrue(e.data == 3.0)
        self.assertFalse(e.data is None)
        self.assertTrue(e.ident == x)
        self.assertFalse(e.ident is None)
        self.assertTrue(e.ident == (0, 1))
        self.assertFalse(e.ident is None)
        self.assertTrue(e.left == n)
        self.assertFalse(e.weight < 0)
        self.assertTrue(e.weight == 2.3)
        self.assertFalse(e.weight == 2.4)


class TestBipartiteGraph(unittest.TestCase):
    def test_init(self):
        print("Testing init")
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 15  # nodeset size
        k = 5  # neighbor size

        ct = 0
        while len(g.left) < n:
            # print("Adding new leftnode: ", len(g.left))
            while ct in g.left or ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            numleft = len(g.left)
            rejected = g.add_left(nod)
            newnumleft = len(g.left)
            self.assertTrue(not rejected and newnumleft - numleft == 1)

        while len(g.right) < n:
            # print("Adding new rightnode: ", len(G.right))
            while ct in g.right or ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            numright = len(g.right)
            rejected = g.add_right(nod)
            newnumright = len(g.right)
            self.assertTrue(not rejected and newnumright - numright == 1)

        leftnodekeys = list(g.left.keys())
        rightnodekeys = list(g.right.keys())
        for i in range(n):
            sigidx = rightnodekeys[i]
            rightnode = g.right[sigidx]
            idxs = random.sample(range(n), k)
            assert len(idxs) == k
            for j in idxs:
                otkidx = leftnodekeys[j]  # one-time key (otk) index (idx)
                leftnode = g.left[otkidx]
                x = (leftnode.ident, rightnode.ident)
                par = {'data': 1.0, 'ident': x, 'left': leftnode, 'right': rightnode, 'weight': 0.0}
                e = Edge(par)
                g.add_in_edge(e)

        self.assertTrue(len(g.left) + len(g.right) == 2*n)
        self.assertTrue(len(g.in_edges) == k*n)

    def test_add_delete(self):
        # Pick a random graph identity
        par = {'data': 1.0, 'ident': 'han-tyumi', 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 25  # nodeset size... G actually has 2*N nodes in total
        k = 5  # neighbor size = ring size

        # First: Throw all the 2N nodes into G
        # For each new node, check that the node set size has increased by one
        nodect = 0
        while len(g.left) < n:
            while nodect in g.left or nodect in g.right:
                nodect += 1
            par = {'data': 1.0, 'ident': nodect}
            nod = Node(par)
            numleft = len(g.left)
            rejected = g.add_left(nod)
            self.assertFalse(rejected)
            newnumleft = len(g.left)
            self.assertTrue(newnumleft - numleft == 1)            

        while len(g.right) < n:
            while nodect in g.right or nodect in g.left:
                nodect += 1
            par = {'data': 1.0, 'ident': nodect}
            nod = Node(par)
            numright = len(g.right)
            rejected = g.add_right(nod)
            self.assertFalse(rejected)
            newnumright = len(g.right)
            self.assertTrue(newnumright - numright == 1)

        # Check that there are 2N nodes in the node sets and no edges
        self.assertTrue(len(g.left) + len(g.right) == 2*n)
        self.assertTrue(len(g.in_edges) == 0)
        self.assertTrue(len(g.out_edges) == 0)

        leftnodekeys = list(g.left.keys())
        rightnodekeys = list(g.right.keys())
        r = int(random.choice(rightnodekeys))  # Pick a random signature
        l = []
        while len(l) < k:  # K = number of neighbors left neighbors adjacent to in-edges, i.e. ring size
            nextl = random.choice(leftnodekeys)  # For now we will just pick neighbors uniformly at random with repl
            if nextl not in l:
                l.append(nextl)

        # Assemble identities of the edges to be added for this signature
        self.assertTrue(len(l) == k)
        edge_idents = [(lefty, r) for lefty in l]
        self.assertTrue(len(edge_idents) == k)
        
        for i in range(k):
            # For each ring member, create an edge, add it, and check that the edge was not rejected.
            par = {'data': 1.0, 'ident': edge_idents[i], 'left': g.left[l[i]], 'right': g.right[r], 'weight': 0.0}
            e = Edge(par)
            self.assertTrue(e.ident not in g.in_edges)
            self.assertTrue(e.ident not in g.out_edges)
            rejected = g.add_in_edge(e)
            self.assertTrue(e.ident in g.in_edges)
            self.assertFalse(rejected)
        # Check that enough edges were added.
        self.assertTrue(len(g.in_edges) == k)

        # Pick a random signature and key that are not connected with an in-edge
        r = random.choice(rightnodekeys)
        l = random.choice(leftnodekeys) 
        while (l, r) in g.in_edges:
            r = random.choice(rightnodekeys)
            l = random.choice(leftnodekeys)
            
        # Create an out-edge for this pair
        eid = (l, r)
        par = {'data': 1.0, 'ident': eid, 'left': g.left[l], 'right': g.right[r], 'weight': 0.0}
        e = Edge(par)
        # Include the edge in the graph, verify not rejected, verify there is only one out-edge.
        rejected = g.add_out_edge(e)
        self.assertFalse(rejected)
        self.assertTrue(len(g.out_edges) == 1)

        # Find the next node identity that is not yet in the graph
        while nodect in g.left or nodect in g.right:
            nodect += 1
        # Create that left node, add it to the left
        l = nodect
        par = {'data': 1.0, 'ident': l}
        nod = Node(par)
        rejected = g.add_left(nod)
        # Verify node not rejected, that there are now N+1 left nodes, still N right nodes, K in edges and 1 out edge.
        self.assertFalse(rejected)
        self.assertEqual(len(g.left), n+1)
        self.assertEqual(len(g.right), n)
        self.assertTrue(len(g.in_edges) == k)
        self.assertTrue(len(g.out_edges) == 1)

        # Pick the next node identity that doesn't yet exist.
        while nodect in g.right or nodect in g.left:
            nodect += 1
        # Create that node and add it to the right.
        r = nodect
        par = {'data': 2.0, 'ident': r}
        nod = Node(par)
        rejected = g.add_right(nod)
        # Verify inclusion of point was not rejected, both sides have N+1 nodes, still only K in edges and 1 out edge
        self.assertFalse(rejected)
        self.assertEqual(len(g.left), n+1)
        self.assertEqual(len(g.right), n+1)
        self.assertTrue(len(g.in_edges) == k)
        self.assertTrue(len(g.out_edges) == 1)

        # Now create either an in-edge or an out-edge connecting l and r with 50% probability of each
        b = random.choice([0, 1])  # Pick a random bit
        # Construct the edge
        par = {'data': 1.0, 'ident': (l, r), 'left': g.left[l], 'right': g.right[r], 'weight': 0.0}
        e = Edge(par)
        rejected = None
        # Include edge as in-edge if b=0, out-edge if b=1, and mark the edge as rejected if neither work.
        if b == 0:
            rejected = g.add_in_edge(e)
        elif b == 1:
            rejected = g.add_out_edge(e)

        self.assertTrue(rejected is not None)  # This should always be true since we've covered all possibilities.
        self.assertFalse(rejected)  # Check that the new edge was not rejected
        # Make sure that we have the right number of total edges and verify that the edge landed in the correct place
        self.assertEqual(len(g.in_edges) + len(g.out_edges), k+2)
        if b == 0:
            self.assertEqual(len(g.in_edges), k+1)
            self.assertEqual(len(g.out_edges), 1)
        elif b == 1:
            self.assertEqual(len(g.in_edges), k)
            self.assertEqual(len(g.out_edges), 2)
        else:
            self.assertTrue(rejected)

        # Pick a random in-edge
        ek = random.choice(list(g.in_edges.keys()))
        e = g.in_edges[ek]
        self.assertEqual(e.ident, ek)
        self.assertTrue(ek in g.in_edges)
        # Attempt to delete it.
        rejected = g.del_edge(e)
        # Check that deletion worked
        self.assertFalse(rejected)
        self.assertEqual(len(g.in_edges) + len(g.out_edges), k+1)

        # Pick a random leftnode
        mk = random.choice(list(g.left.keys()))
        nod = g.left[mk]
        # Collect the edge identities of edges that will be delted if this node is deleted
        edges_lost = len(nod.in_edges)
        # Attempt to delete the node
        rejected = g.del_node(nod)
        self.assertFalse(rejected)
        self.assertTrue(len(g.left) + len(g.right) == 2*n+1)
        self.assertEqual(len(g.in_edges) + len(g.out_edges), k+1 - edges_lost)

    def test_make_graph(self):
        for i in range(2, 20):
            for r in range(2, i):
                g = make_graph(i, r)
                self.assertTrue(len(g.left)+len(g.right) == 2*i)
                self.assertTrue(len(g.in_edges) == r*i)
                for j in range(i):
                    right_node_key = list(g.right.keys())[j]
                    right_node = g.right[right_node_key]
                    self.assertEqual(len(right_node.in_edges), r)

    def test_bfs_one(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 2
        ct = 0
        while len(g.left) < n:
            while ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        self.assertTrue(0 in leftkeys)
        self.assertTrue(1 in leftkeys)
        self.assertEqual(len(leftkeys), 2)

        rightkeys = list(g.right.keys())
        self.assertTrue(2 in rightkeys)
        self.assertTrue(3 in rightkeys)
        self.assertEqual(len(rightkeys), 2)

        l = g.left[0]
        r = g.right[2]
        x = (l.ident, r.ident)
        par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0.0}
        e = Edge(par)
        g.add_in_edge(e)

        l = g.left[1]
        r = g.right[2]
        x = (l.ident, r.ident)
        par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0.0}
        e = Edge(par)
        g.add_in_edge(e)

        l = g.left[1]
        r = g.right[3]
        x = (l.ident, r.ident)
        par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0.0}
        e = Edge(par)
        g.add_in_edge(e)

        eid1 = (0, 2)
        eid2 = (1, 2)
        eid3 = (1, 3)

        self.assertTrue(eid1 in g.in_edges)
        self.assertTrue(eid2 in g.in_edges)
        self.assertTrue(eid3 in g.in_edges)

        match = list()
        match.append(g.in_edges[eid2])

        results = g.bfs(match)
        self.assertTrue(len(results) == 1)
        self.assertTrue(len(results[0]) == 3)
        self.assertTrue(results[0][0].left.ident == 0)
        self.assertTrue(results[0][0].right.ident == 2)
        self.assertTrue(results[0][1].left.ident == 1)
        self.assertTrue(results[0][1].right.ident == 2)
        self.assertTrue(results[0][2].left.ident == 1)
        self.assertTrue(results[0][2].right.ident == 3)
        # print("results from bfs_one= " + str(results))

    def test_bfs_disconnected(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 4
        ct = 0
        while len(g.left) < n:
            while ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents = [[0, 4], [1, 4], [2, 4], [2, 5], [2, 6], [3, 7]]
        eid = []
        for ident in edge_idents:
            l = g.left[ident[0]]
            r = g.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0.0}
            e = Edge(par)
            g.add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2, 4)
        match.append(g.in_edges[match_edge])

        results = g.bfs(match)
        # print(results, type(results))
        self.assertTrue(len(results) == 1)
        self.assertTrue(len(results[0]) == 1)
        self.assertTrue(results[0][0].left.ident == 3)
        self.assertTrue(results[0][0].right.ident == 7)

    def test_bfs_fourshortest(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left) < n:
            while ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents = [[0, 3], [1, 3], [2, 3], [2, 4], [2, 5]]
        eid = []
        for ident in edge_idents:
            l = g.left[ident[0]]
            r = g.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0.0}
            e = Edge(par)
            g.add_in_edge(e)
            eid.append(x)

        # for e in G.in_edges:
        #    print(e, type(e), str(e))
        match = []
        match_edge = (2, 3)
        match.append(g.in_edges[match_edge])

        results = g.bfs(match)
        self.assertTrue(len(results) == 4)
        # print("RESULTS FROM TEST BFS FOURSHORTEST")
        for entry in results:
            self.assertTrue(len(entry) == 3)
            # print([e.ident for e in entry])

        self.assertTrue(len(results) == 4)

        self.assertTrue(len(results[0]) == 3)

        self.assertTrue(results[0][0].left.ident in [0, 1])
        self.assertTrue(results[0][0].right.ident == 3)
        self.assertTrue(results[0][1].left.ident == 2 and results[0][1].right.ident == 3)
        self.assertTrue(results[0][2].left.ident == 2 and results[0][2].right.ident in [4, 5])

    def test_get_augmenting_path(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left) < n:
            while ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents = [[0, 3], [1, 3], [2, 3], [2, 4], [2, 5]]
        eid = []
        for ident in edge_idents:
            l = g.left[ident[0]]
            r = g.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0}
            e = Edge(par)
            g.add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2, 3)
        match.append(g.in_edges[match_edge])

        results = g.get_augmenting_paths(match, wt=False)
        # line = [[e.ident for e in p] for p in results]
        # print("Results from _get_augmenting_paths = " + str(line))
        self.assertTrue(len(results) == 1)
        self.assertTrue(len(results[0]) == 3)
        self.assertTrue(results[0][0].left.ident == 0 or results[0][0].left.ident == 1)
        self.assertTrue(results[0][0].right.ident == 3)
        self.assertTrue(results[0][1].left.ident == 2)
        self.assertTrue(results[0][1].right.ident == 3)
        self.assertTrue(results[0][2].left.ident == 2)
        self.assertTrue(results[0][2].right.ident == 4 or results[0][2].right.ident == 5)

    def test_max_matching_weightless(self):
        # print("Beginning test_max_matching)")
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left) < n:
            while ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents = [[0, 3], [1, 3], [2, 3], [2, 4], [2, 5]]
        eid = []
        for ident in edge_idents:
            l = g.left[ident[0]]
            r = g.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0}
            e = Edge(par)
            g.add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2, 3)
        match.append(g.in_edges[match_edge])

        results = g.max_matching(match)
        eids = [e.ident for e in results]
        self.assertTrue((0, 3) in eids or (1, 3) in eids)
        self.assertTrue((2, 4) in eids or (2, 5) in eids)
        self.assertFalse((2, 3) in eids)
        self.assertEqual(len(eids), 2)

    def test_max_matching_weighted(self):
        # print("Beginning test_max_matching)")
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 3
        ct = 0
        while len(g.left) < n:
            while ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        edge_idents_and_weights = [[0, 3, 1], [1, 3, 2], [2, 3, 3], [2, 4, 4], [2, 5, 5]]
        eid = []
        for ident in edge_idents_and_weights:
            l = g.left[ident[0]]
            r = g.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': ident[2]}
            e = Edge(par)
            g.add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2, 3)
        match.append(g.in_edges[match_edge])

        results = g.max_matching(match, True)
        edge_idents = [e.ident for e in results]
        assert (1, 3) in edge_idents
        assert (2, 5) in edge_idents

        # line = [e.ident for e in results]
        # print("Maximal matching is " + str(line))

    def test_get_improving_cycles(self):
        # print("Beginning test_max_matching)")
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        g = BipartiteGraph(par)
        n = 4
        ct = 0
        while len(g.left) < n:
            while ct in g.left or ct in g.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_left(nod)
            ct += 1
        while len(g.right) < n:
            while ct in g.right or ct in g.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            nod = Node(par)
            g.add_right(nod)
            ct += 1

        leftkeys = list(g.left.keys())
        for i in range(n):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), n)

        rightkeys = list(g.right.keys())
        for i in range(n, 2*n):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), n)

        weids = [[0, 4, 0], [0, 5, 1], [1, 4, 2], [1, 5, 3], [1, 6, 4], [2, 5, 5], [2, 7, 6], [3, 6, 7], [3, 7, 8]]
        eids = []
        for ident in weids:
            l = g.left[ident[0]]
            r = g.right[ident[1]]
            x = (ident[0], ident[1])
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': ident[2]}
            e = Edge(par)
            g.add_in_edge(e)
            eids.append(x)

        bad_ids = [(0, 4), (1, 6), (2, 5), (3, 7)]
        bad_match = [g.in_edges[i] for i in bad_ids]
        for e in bad_match:
            self.assertTrue(isinstance(e, Edge))
        results = g.get_improving_cycles(bad_match)
        # print("Readable results:")
        # readable_results = [[e.ident for e in c] for c in results]
        # for r in readable_results:
        #     print("\n"+str(r))
        self.assertEqual(len(results), 0)

        g.in_edges[(3, 6)].weight = 100
        results = g.get_improving_cycles(bad_match)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 6)
        resulting_identities = [e.ident for e in results[0]]
        self.assertTrue((3, 6) in resulting_identities)
        self.assertTrue((1, 6) in resulting_identities)
        self.assertTrue((1, 5) in resulting_identities)
        self.assertTrue((2, 5) in resulting_identities)
        self.assertTrue((2, 7) in resulting_identities)
        self.assertTrue((3, 7) in resulting_identities)

    def test_get_opt_matching(self, sample_size=10**3, verbose=False):
        for tomato in range(sample_size):
            # print("Beginning test_max_matching)")
            s = random.random()
            x = str(hash(str(1.0) + str(s)))
            par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
            g = BipartiteGraph(par)
            n = 4
            ct = 0
            while len(g.left) < n:
                while ct in g.left:
                    ct += 1
                par = {'data': 1.0, 'ident': ct}
                nod = Node(par)
                g.add_left(nod)
                ct += 1
            while len(g.right) < n:
                while ct in g.right:
                    ct += 1
                par = {'data': 1.0, 'ident': ct}
                nod = Node(par)
                g.add_right(nod)
                ct += 1

            leftkeys = list(g.left.keys())
            for i in range(n):
                self.assertTrue(i in leftkeys)
            self.assertEqual(len(leftkeys), n)

            rightkeys = list(g.right.keys())
            for i in range(n, 2*n):
                self.assertTrue(i in rightkeys)
            self.assertEqual(len(rightkeys), n)

            # print(g.left)
            # print(g.right)
            for i in range(n):
                for j in range(n):
                    l = g.left[i]
                    r = g.right[j+n]
                    x = (i, j+n)
                    par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': random.random()}
                    e = Edge(par)
                    g.add_in_edge(e)
            # print(g.in_edges)

            results = g.opt_matching(level_of_cycles=3)
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


tests = [TestSymDif, TestNode, TestEdge, TestBipartiteGraph]
for test in tests:
    unittest.TextTestRunner(verbosity=2, failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
