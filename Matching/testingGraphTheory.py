import unittest
import random
from graphtheory import *

class Test_sym_dif(unittest.TestCase):
    def test(self):
        x = [1, 2, 3]
        y = [5, 6, 7]
        self.assertTrue(x+y == sym_dif(x,y))

        y.append(3)
        y.append(2)
        x.append(5)
        x.append(6)
        self.assertTrue(1 in sym_dif(x,y))
        self.assertTrue(7 in sym_dif(x,y))
        self.assertTrue(len(sym_dif(x,y))==2)

        x.append(0)
        self.assertTrue(0 in sym_dif(x,y))
        self.assertTrue(1 in sym_dif(x, y))
        self.assertTrue(7 in sym_dif(x, y))
        self.assertTrue(len(sym_dif(x, y)) == 3)

        x = random.sample(range(100),60)
        y = random.sample(range(100),60) # Pigeonhole principle implies at least 20 collisions
        z = sym_dif(x,y)
        self.assertTrue(len(z) >= 20)
        zp = [w for w in x if w not in y]+[w for w in y if w not in x]
        self.assertEqual(len(z),len(zp))
        for i in z:
            self.assertTrue(i in zp)
            self.assertTrue((i in x and i not in y) or (i in y and i not in x))


class Test_Node(unittest.TestCase):
    def test_init(self):
        par = {'data':None, 'ident':None}
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

        par = {'data': 3.0, 'ident': (0,1), 'left':n, 'right':m, 'weight': 2.3}
        e = Edge(par)

        n._add_in_edge(e)
        m._add_in_edge(e)

        self.assertTrue(len(n.in_edges) == 1)
        self.assertTrue(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)

    def test_del_edge(self):
        x=0
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        x=1
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        x=(0,1)
        par = {'data': 3.0, 'ident': x, 'left':n, 'right':m, 'weight': 2.3}
        e = Edge(par)
        n._add_in_edge(e)
        m._add_in_edge(e)

        self.assertTrue(len(n.in_edges) == 1)
        self.assertTrue(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)

        n._del_edge(e)
        self.assertTrue(len(n.in_edges) == 0)
        self.assertFalse(e in n.in_edges)
        self.assertTrue(len(m.in_edges)==1)
        self.assertTrue(e in m.in_edges)

        m._del_edge(e)
        self.assertTrue(len(n.in_edges) == 0)
        self.assertFalse(e in n.in_edges)
        self.assertTrue(len(m.in_edges)==0)
        self.assertFalse(e in m.in_edges)


class Test_Edge(unittest.TestCase):
    def test_init(self):
        x=0
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        x=1
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        x=(0,1)
        par = {'data': 3.0, 'ident': x, 'left':n, 'right':m, 'weight': 2.3}
        e = Edge(par)
        n._add_in_edge(e)
        m._add_in_edge(e)

        self.assertTrue(len(n.in_edges) == 1)
        self.assertTrue(e in n.in_edges)
        self.assertTrue(len(m.in_edges) == 1)
        self.assertTrue(e in m.in_edges)
        self.assertTrue(e.data == 3.0)
        self.assertFalse(e.data == None)
        self.assertTrue(e.ident == x)
        self.assertFalse(e.ident == None)
        self.assertTrue(e.ident == (0,1))
        self.assertFalse(e.ident == None)
        self.assertTrue(e.left == n)
        self.assertFalse(e.weight < 0)
        self.assertTrue(e.weight == 2.3)
        self.assertFalse(e.weight == 2.4)


class Test_BipartiteGraph(unittest.TestCase):
    def test_init(self):
        print("Testing init")
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x, 'left':[], 'right':[], 'in_edges':[], 'out_edges':[] }
        G = BipartiteGraph(par)
        N = 15 # nodeset size
        K = 5 # neighbor size

        ct = 0
        while len(G.left) < N:
            #print("Adding new leftnode: ", len(G.left))
            while ct in G.left or ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            numLeft = len(G.left)
            rejected = G._add_left(n)
            newNumLeft = len(G.left)
            self.assertTrue(not rejected and newNumLeft - numLeft == 1)
        while len(G.right) < N:
            #print("Adding new rightnode: ", len(G.right))
            while ct in G.right or ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            numRight = len(G.right)
            rejected = G._add_right(n)
            newNumRight = len(G.right)
            self.assertTrue(not rejected and newNumRight - numRight == 1)

        leftnodekeys = list(G.left.keys())
        rightnodekeys = list(G.right.keys())
        for i in range(N):
            sig_idx = rightnodekeys[i]
            right_node = G.right[sig_idx]
            idxs = random.sample(range(N), K)
            assert len(idxs) == K
            for j in idxs:
                otk_idx = leftnodekeys[j]
                left_node = G.left[otk_idx]
                x = (left_node.ident,right_node.ident)
                par = {'data':1.0, 'ident':x, 'left':left_node, 'right':right_node, 'weight':0}
                e = Edge(par)
                G._add_in_edge(e)

        self.assertTrue(len(G.left)+len(G.right) == 2*N)
        self.assertTrue(len(G.in_edges) == K*N)

    def test_add_delete(self):
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x, 'left':[], 'right':[], 'in_edges':[], 'out_edges':[] }
        G = BipartiteGraph(par)
        N = 25 # nodeset size... G actually has 2*N nodes in total
        K = 5 # neighbor size

        node_ct = 0

        while len(G.left) < N:
            while(node_ct in G.left or node_ct in G.right):
                node_ct += 1
            par = {'data': 1.0, 'ident': node_ct}
            n = Node(par)
            numleft = len(G.left)
            rejected = G._add_left(n)
            self.assertFalse(rejected)
            newnumleft = len(G.left)
            self.assertTrue(newnumleft - numleft == 1)            

        while len(G.right) < N:
            while (node_ct in G.right or node_ct in G.left):
                node_ct += 1
            par = {'data': 1.0, 'ident': node_ct}
            n = Node(par)
            numright = len(G.right)
            rejected = G._add_right(n)
            self.assertFalse(rejected)
            newnumright = len(G.right)
            self.assertTrue(newnumright - numright == 1)

        self.assertTrue(len(G.left)+len(G.right) == 2*N)
        self.assertTrue(len(G.in_edges)==0)
        self.assertTrue(len(G.out_edges)==0)

        leftnodekeys = list(G.left.keys())
        rightnodekeys = list(G.right.keys())

        r = int(random.choice(rightnodekeys))
        l = []
        while(len(l) < K):
            nextl = random.choice(leftnodekeys)
            if nextl not in l:
                l.append(nextl)
        edge_idents = [(lefty, r) for lefty in l]
        self.assertTrue(len(edge_idents)==K)
        for i in range(K):
            par = {'data':1.0, 'ident':edge_idents[i], 'left':G.left[l[i]], 'right':G.right[r], 'weight':0}
            e = Edge(par)
            G._add_in_edge(e)
        self.assertTrue(len(G.in_edges) == K)

        r = random.choice(rightnodekeys)
        l = random.choice(leftnodekeys)
        while((l,r) in G.in_edges):
            r = random.choice(rightnodekeys)
            l = random.choice(leftnodekeys)
            
        eid = (l,r)
        par = {'data':1.0, 'ident':eid, 'left':G.left[l], 'right':G.right[r], 'weight':0}
        e = Edge(par)
        G._add_out_edge(e)
        self.assertTrue(len(G.out_edges) == 1)

        while (node_ct in G.left or node_ct in G.right):
            node_ct += 1
        l = node_ct
        par = {'data': 1.0, 'ident': l}
        n = Node(par)
        G._add_left(n)
        self.assertEqual(len(G.left), N+1)
        self.assertEqual(len(G.right),N)
        self.assertTrue(len(G.in_edges) == K)
        self.assertTrue(len(G.out_edges) == 1)

        while (node_ct in G.right or node_ct in G.left):
            node_ct += 1
        r = node_ct
        par = {'data': 2.0, 'ident': r}
        m = Node(par)
        G._add_right(m)
        self.assertEqual(len(G.left),N+1)
        self.assertEqual(len(G.right),N+1)
        self.assertTrue(len(G.in_edges) == K)
        self.assertTrue(len(G.out_edges) == 1)

        b = random.choice([0,1])
        par = {'data':1.0, 'ident':(l,r), 'left':G.left[l], 'right':G.right[r], 'weight':0}
        e = Edge(par)
        if b==0:
            G._add_in_edge(e)
        elif b==1:
            G._add_out_edge(e)

        self.assertTrue(len(G.in_edges) in [K, K+1])
        self.assertTrue(len(G.out_edges) in [1,2])
        self.assertEqual(len(G.in_edges) + len(G.out_edges), K+2)

        ek = random.choice(list(G.in_edges.keys()))
        e = G.in_edges[ek]
        rejected = G._del_edge(e)
        self.assertFalse(rejected)
        self.assertTrue(len(G.in_edges) in [K-1, K])
        self.assertTrue(len(G.out_edges) in [1,2])
        self.assertEqual(len(G.in_edges) + len(G.out_edges), K+1)

        mk = random.choice(list(G.left.keys()))
        m = G.left[mk]
        edges_lost = len(m.in_edges)
        rejected = G._del_node(m)
        self.assertFalse(rejected)
        self.assertTrue(len(G.left)+len(G.right) == 2*N+1)
        self.assertTrue(len(G.in_edges) in [K-1-edges_lost,K-edges_lost])

    def test_make_graph(self):
        for i in range(2,20):
            for r in range(2,i):
                G = make_graph(i,r)
                self.assertTrue(len(G.left)+len(G.right) == 2 * i)
                self.assertTrue(len(G.in_edges) == r * i)
                for j in range(i):
                    right_node_key = list(G.right.keys())[j]
                    right_node = G.right[right_node_key]
                    self.assertEqual(len(right_node.in_edges), r)


    def test_bfs_one(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        G = BipartiteGraph(par)
        N = 2
        ct = 0
        while len(G.left) < N:
            while ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        self.assertTrue(0 in leftkeys)
        self.assertTrue(1 in leftkeys)
        self.assertEqual(len(leftkeys),2)

        rightkeys = list(G.right.keys())
        self.assertTrue(2 in rightkeys)
        self.assertTrue(3 in rightkeys)
        self.assertEqual(len(rightkeys),2)

        l = G.left[0]
        r = G.right[2]
        x = (l.ident, r.ident)
        par = {'data': 1.0, 'ident': x, 'left':l, 'right':r, 'weight': 0}
        e = Edge(par)
        G._add_in_edge(e)

        l = G.left[1]
        r = G.right[2]
        x = (l.ident, r.ident)
        par = {'data': 1.0, 'ident': x, 'left':l, 'right':r, 'weight': 0}
        e = Edge(par)
        G._add_in_edge(e)

        l = G.left[1]
        r = G.right[3]
        x = (l.ident, r.ident)
        par = {'data': 1.0, 'ident': x, 'left':l, 'right':r, 'weight': 0}
        e = Edge(par)
        G._add_in_edge(e)

        eid1 = (0,2)
        eid2 = (1,2)
        eid3 = (1,3)

        self.assertTrue(eid1 in G.in_edges)
        self.assertTrue(eid2 in G.in_edges)
        self.assertTrue(eid3 in G.in_edges)

        match = []
        match.append(G.in_edges[eid2])

        results = G._bfs(match)
        self.assertTrue(len(results)==1)
        self.assertTrue(len(results[0])==3)
        self.assertTrue(results[0][0].left.ident == 0)
        self.assertTrue(results[0][0].right.ident == 2)
        self.assertTrue(results[0][1].left.ident == 1)
        self.assertTrue(results[0][1].right.ident == 2)
        self.assertTrue(results[0][2].left.ident == 1)
        self.assertTrue(results[0][2].right.ident == 3)
        #print("results from bfs_one= " + str(results))


    def test_bfs_disconnected(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        G = BipartiteGraph(par)
        N = 4
        ct = 0
        while len(G.left) < N:
            while ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        for i in range(N):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys),N)

        rightkeys = list(G.right.keys())
        for i in range(N,2*N):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys),N)

        edge_idents = [[0,4], [1,4], [2,4], [2, 5], [2, 6], [3, 7]]
        eid = []
        for ident in edge_idents:
            l = G.left[ident[0]]
            r = G.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left':l, 'right':r, 'weight': 0}
            e = Edge(par)
            G._add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2,4)
        match.append(G.in_edges[match_edge])

        results = G._bfs(match)
        #print(results, type(results))
        self.assertTrue(len(results)==1)
        self.assertTrue(len(results[0])==1)
        self.assertTrue(results[0][0].left.ident == 3)
        self.assertTrue(results[0][0].right.ident == 7)


    def test_bfs_fourshortest(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        G = BipartiteGraph(par)
        N = 3
        ct = 0
        while len(G.left) < N:
            while ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        for i in range(N):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys),N)

        rightkeys = list(G.right.keys())
        for i in range(N,2*N):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys),N)

        edge_idents = [[0,3], [1,3], [2,3], [2, 4], [2, 5]]
        eid = []
        for ident in edge_idents:
            l = G.left[ident[0]]
            r = G.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left':l, 'right':r, 'weight': 0}
            e = Edge(par)
            G._add_in_edge(e)
            eid.append(x)

        #for e in G.in_edges:
        #    print(e, type(e), str(e))
        match = []
        match_edge = (2,3)
        match.append(G.in_edges[match_edge])

        results = G._bfs(match)
        self.assertTrue(len(results)==4)
        #print("RESULTS FROM TEST BFS FOURSHORTEST")
        for entry in results:
            self.assertTrue(len(entry)==3)
            #print([e.ident for e in entry])

        self.assertTrue(len(results)==4)

        self.assertTrue(len(results[0])==3)

        self.assertTrue(results[0][0].left.ident in [0,1] and results[0][0].right.ident == 3)
        self.assertTrue(results[0][1].left.ident == 2 and results[0][1].right.ident == 3)
        self.assertTrue(results[0][2].left.ident == 2 and results[0][2].right.ident in [4,5])
        

    def test_get_augmenting_path(self):
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges':[]}
        G = BipartiteGraph(par)
        N = 3
        ct = 0
        while len(G.left) < N:
            while ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        for i in range(N):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys),N)

        rightkeys = list(G.right.keys())
        for i in range(N,2*N):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys),N)

        edge_idents = [[0,3], [1,3], [2,3], [2, 4], [2, 5]]
        eid = []
        for ident in edge_idents:
            l = G.left[ident[0]]
            r = G.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left':l, 'right':r, 'weight': 0}
            e = Edge(par)
            G._add_in_edge(e)
            eid.append(x)


        match = []
        match_edge = (2,3)
        match.append(G.in_edges[match_edge])

        results = G._get_augmenting_paths(match, wt=False)
        line = [[e.ident for e in p] for p in results]
        #print("Results from _get_augmenting_paths = " + str(line))
        self.assertTrue(len(results)==1)
        self.assertTrue(len(results[0])==3)
        self.assertTrue(results[0][0].left.ident == 0 or results[0][0].left.ident == 1)
        self.assertTrue(results[0][0].right.ident == 3)
        self.assertTrue(results[0][1].left.ident == 2)
        self.assertTrue(results[0][1].right.ident == 3)
        self.assertTrue(results[0][2].left.ident == 2)
        self.assertTrue(results[0][2].right.ident == 4 or results[0][2].right.ident == 5)

    def test_max_matching_weightless(self):
        #print("Beginning test_max_matching)")
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges':[]}
        G = BipartiteGraph(par)
        N = 3
        ct = 0
        while len(G.left) < N:
            while ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        for i in range(N):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), N)

        rightkeys = list(G.right.keys())
        for i in range(N, 2 * N):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), N)

        edge_idents = [[0, 3], [1, 3], [2, 3], [2, 4], [2, 5]]
        eid = []
        for ident in edge_idents:
            l = G.left[ident[0]]
            r = G.right[ident[1]]
            x = (l.ident, r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': 0}
            e = Edge(par)
            G._add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2,3)
        match.append(G.in_edges[match_edge])

        results = G.max_matching(match)
        # Test results here
        pass

    def test_max_matching_weighted(self):
        #print("Beginning test_max_matching)")
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges': []}
        G = BipartiteGraph(par)
        N = 3
        ct = 0
        while len(G.left) < N:
            while ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        for i in range(N):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), N)

        rightkeys = list(G.right.keys())
        for i in range(N, 2 * N):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), N)

        edge_idents_and_weights = [[0, 3, 1], [1, 3, 2], [2, 3, 3], [2, 4, 4], [2, 5, 5]]
        eid = []
        for ident in edge_idents_and_weights:
            l = G.left[ident[0]]
            r = G.right[ident[1]]
            x = (l.ident,  r.ident)
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': ident[2]}
            e = Edge(par)
            G._add_in_edge(e)
            eid.append(x)

        match = []
        match_edge = (2,3)
        match.append(G.in_edges[match_edge])

        results = G.max_matching(match, True)
        edge_idents = [e.ident for e in results]
        #print(edge_idents)
        assert (1,3) in edge_idents
        assert (2,5) in edge_idents
        line = [e.ident for e in results]
        #print("Maximal matching is " + str(line))

    def test_get_improving_cycles(self):
        #print("Beginning test_max_matching)")
        s = random.random()
        x = str(hash(str(1.0) + str(s)))
        par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges':[]}
        G = BipartiteGraph(par)
        N = 4
        ct = 0
        while len(G.left) < N:
            while ct in G.left or ct in G.right:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_left(n)
            ct += 1
        while len(G.right) < N:
            while ct in G.right or ct in G.left:
                ct += 1
            par = {'data': 1.0, 'ident': ct}
            n = Node(par)
            G._add_right(n)
            ct += 1

        leftkeys = list(G.left.keys())
        for i in range(N):
            self.assertTrue(i in leftkeys)
        self.assertEqual(len(leftkeys), N)

        rightkeys = list(G.right.keys())
        for i in range(N, 2 * N):
            self.assertTrue(i in rightkeys)
        self.assertEqual(len(rightkeys), N)


        weids = [ [0, 4, 0], [0, 5, 1], [1, 4, 2], [1, 5, 3], [1, 6, 4], [2, 5, 5], [2, 7, 6], [3, 6, 7], [3, 7, 8]]
        eids = []
        for ident in weids:
            l = G.left[ident[0]]
            r = G.right[ident[1]]
            x = (ident[0], ident[1])
            par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': ident[2]}
            e = Edge(par)
            G._add_in_edge(e)
            eids.append(x)

        bad_ids = [(0,4), (1,6), (2,5), (3,7)]
        bad_match = [G.in_edges[i] for i in bad_ids]
        for e in bad_match:
            self.assertTrue(isinstance(e, Edge))
        results = G._get_improving_cycles(bad_match)
        #print("Readable results:")
        #readable_results = [[e.ident for e in c] for c in results]
        #for r in readable_results:
        #    print("\n"+str(r))
        self.assertEqual(len(results),0)

        G.in_edges[(3,6)].weight = 100
        results = G._get_improving_cycles(bad_match)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 6)
        resulting_identities = [e.ident for e in results[0]]
        self.assertTrue((3,6) in resulting_identities)
        self.assertTrue((1,6) in resulting_identities)
        self.assertTrue((1,5) in resulting_identities)
        self.assertTrue((2,5) in resulting_identities)
        self.assertTrue((2,7) in resulting_identities)
        self.assertTrue((3,7) in resulting_identities)

    def test_get_opt_matching(self, sample_size=10**3, verbose=False):
        for tomato in range(sample_size):
            # print("Beginning test_max_matching)")
            s = random.random()
            x = str(hash(str(1.0) + str(s)))
            par = {'data': 1.0, 'ident': x, 'left': [], 'right': [], 'in_edges': [], 'out_edges':[]}
            G = BipartiteGraph(par)
            N = 4
            ct = 0
            while len(G.left) < N:
                while ct in G.left:
                    ct += 1
                par = {'data': 1.0, 'ident': ct}
                n = Node(par)
                G._add_left(n)
                ct += 1
            while len(G.right) < N:
                while ct in G.right:
                    ct += 1
                par = {'data': 1.0, 'ident': ct}
                n = Node(par)
                G._add_right(n)
                ct += 1

            leftkeys = list(G.left.keys())
            for i in range(N):
                self.assertTrue(i in leftkeys)
            self.assertEqual(len(leftkeys), N)

            rightkeys = list(G.right.keys())
            for i in range(N, 2 * N):
                self.assertTrue(i in rightkeys)
            self.assertEqual(len(rightkeys), N)

            #print(G.left)
            #print(G.right)
            for i in range(N):
                for j in range(N):
                    l = G.left[i]
                    r = G.right[j+N]
                    x = (i, j+N)
                    par = {'data': 1.0, 'ident': x, 'left': l, 'right': r, 'weight': random.random()}
                    e = Edge(par)
                    G._add_in_edge(e)
            #print(G.in_edges)

            results = G.opt_matching(level_of_cycles=3)
            readable_results = [(e.weight, e.ident) for e in results]
            net_weight = sum(x[0] for x in readable_results)
            if verbose:
                print("optmatch results = " + str(readable_results))

            # There are 24 possible matchings, we are going to compute the weight of each.
            all_m = []
            all_m.append([(0,4), (1,5), (2,6), (3,7)])
            all_m.append([(0,4), (1,5), (2,7), (3,6)])
            all_m.append([(0,4), (1,6), (2,5), (3,7)])
            all_m.append([(0,4), (1,6), (2,7), (3,5)])
            all_m.append([(0,4), (1,7), (2,5), (3,6)])
            all_m.append([(0,4), (1,7), (2,6), (3,5)])
            all_m.append([(0,5), (1,4), (2,6), (3,7)])
            all_m.append([(0,5), (1,4), (2,7), (3,6)])
            all_m.append([(0,5), (1,6), (2,4), (3,7)])
            all_m.append([(0,5), (1,6), (2,7), (3,4)])
            all_m.append([(0,5), (1,7), (2,4), (3,6)])
            all_m.append([(0,5), (1,7), (2,6), (3,4)])
            all_m.append([(0,6), (1,4), (2,5), (3,7)])
            all_m.append([(0,6), (1,4), (2,7), (3,5)])
            all_m.append([(0,6), (1,5), (2,4), (3,7)])
            all_m.append([(0,6), (1,5), (2,7), (3,4)])
            all_m.append([(0,6), (1,7), (2,4), (3,5)])
            all_m.append([(0,6), (1,7), (2,5), (3,4)])
            all_m.append([(0,7), (1,4), (2,5), (3,6)])
            all_m.append([(0,7), (1,4), (2,6), (3,5)])
            all_m.append([(0,7), (1,5), (2,4), (3,6)])
            all_m.append([(0,7), (1,5), (2,6), (3,4)])
            all_m.append([(0,7), (1,6), (2,4), (3,5)])
            all_m.append([(0,7), (1,6), (2,5), (3,4)])


            weighted_matches = []
            for m in all_m:
                this_weight = 0.0
                for eid in m:
                    this_weight += G.in_edges[eid].weight
                weighted_matches.append((this_weight, m))
            if verbose:
                print("==Weight==\t\t\t==Match==")
                for wm in weighted_matches:
                    print(str(wm[0]) + "\t" + str(wm[1]))
                    self.assertTrue(wm[0] <= net_weight)

tests = [Test_sym_dif, Test_Node, Test_Edge, Test_BipartiteGraph]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
