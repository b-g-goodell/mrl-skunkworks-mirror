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
        par = {'data':None, 'ident':str(None)}
        nelly = Node(par)
        self.assertTrue(nelly.data is None)
        self.assertTrue(nelly.ident == str(None))
        self.assertTrue(nelly.edges == [])

        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x}
        nelly = Node(par)
        self.assertFalse(nelly.data is None)
        self.assertTrue(nelly.data == 1.0)
        self.assertFalse(nelly.ident is None)
        self.assertTrue(nelly.ident == x)

    def test_add_edge(self):
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        r = random.random()
        x = str(hash(str(2.0) + str(r)))
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        r = random.random()
        x = str(hash(str(3.0) + str(r)))
        par = {'data': 3.0, 'ident': x, 'endpoints': [n, m], 'weight': 2.3}
        e = Edge(par)

        for node in e.endpoints:
            node._add_edge(e)

        self.assertTrue(len(n.edges) == 1)
        self.assertTrue(e in n.edges)
        self.assertTrue(len(m.edges) == 1)
        self.assertTrue(e in m.edges)

        r = random.random()
        x = str(hash(str(4.0) + str(r)))
        par = {'data': 4.0, 'ident': x, 'endpoints': [n, m], 'weight': -0.2}
        e = Edge(par)

        for node in e.endpoints:
            node._add_edge(e)

        self.assertTrue(len(n.edges) == 2)
        self.assertTrue(e in n.edges)
        self.assertTrue(len(m.edges) == 2)
        self.assertTrue(e in m.edges)

    def test_del_edge(self):
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        r = random.random()
        x = str(hash(str(2.0) + str(r)))
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        r = random.random()
        x = str(hash(str(3.0) + str(r)))
        par = {'data': 3.0, 'ident': x, 'endpoints': [n, m], 'weight': 2.3}
        e = Edge(par)

        for node in e.endpoints:
            node._add_edge(e)

        self.assertTrue(len(n.edges) == 1)
        self.assertTrue(e in n.edges)
        self.assertTrue(len(m.edges) == 1)
        self.assertTrue(e in m.edges)

        n._del_edge(e)
        self.assertTrue(len(n.edges) == 0)
        self.assertFalse(e in n.edges)

        m._del_edge(e)
        self.assertTrue(len(m.edges) == 0)
        self.assertFalse(e in m.edges)


class Test_Edge(unittest.TestCase):
    def test_init(self):
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x}
        n = Node(par)

        r = random.random()
        x = str(hash(str(2.0) + str(r)))
        par = {'data': 2.0, 'ident': x}
        m = Node(par)

        r = random.random()
        x = str(hash(str(3.0) + str(r)))
        par = {'data': 3.0, 'ident': x, 'endpoints': [n, m], 'weight': 2.3}
        e = Edge(par)

        for node in e.endpoints:
            node._add_edge(e)

        self.assertTrue(len(n.edges) == 1)
        self.assertTrue(e in n.edges)
        self.assertTrue(len(m.edges) == 1)
        self.assertTrue(e in m.edges)
        self.assertTrue(e.data == 3.0)
        self.assertFalse(e.data == None)
        self.assertTrue(e.ident == x)
        self.assertFalse(e.ident == None)
        self.assertTrue(e.ident == str(hash(str(3.0) + str(r))))
        self.assertFalse(e.ident == None)
        self.assertTrue(e.endpoints == [n,m])
        self.assertFalse(len(e.endpoints) != 2)
        self.assertFalse(e.weight < 0)
        self.assertTrue(e.weight == 2.3)
        self.assertFalse(e.weight == 2.4)

        r = random.random()
        x = str(hash(str(4.0) + str(r)))
        par = {'data': 4.0, 'ident': x, 'endpoints': [n, m], 'weight': -0.2}
        e = Edge(par)

        for node in e.endpoints:
            node._add_edge(e)

        self.assertTrue(len(n.edges) == 2)
        self.assertTrue(e in n.edges)
        self.assertTrue(len(m.edges) == 2)
        self.assertTrue(e in m.edges)

class Test_Graph(unittest.TestCase):
    def test_init(self):
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x, 'nodes':[], 'edges':[] }
        G = Graph(par)
        N = 15 # nodeset size
        K = 5 # neighbor size

        ct = 0
        while len(G.nodes) < 2*N:
            while str(ct) in G.nodes:
                ct += 1
            par = {'data': 1.0, 'ident': str(ct)}
            n = Node(par)
            G._add_node(n)

        nodekeys = list(G.nodes.keys())
        for i in range(N):
            sig_idx = nodekeys[i]
            right_node = G.nodes[sig_idx]
            idxs = random.sample(range(N), K)
            assert len(idxs) == K
            for j in idxs:
                otk_idx = nodekeys[j+N]
                left_node = G.nodes[otk_idx]
                x = left_node.ident + "," + right_node.ident
                par = {'data':1.0, 'ident':x, 'endpoints':[left_node, right_node], 'weight':0}
                e = Edge(par)
                G._add_edge(e)

        self.assertTrue(len(G.nodes) == 2*N)
        self.assertTrue(len(G.edges) == K*N)


    def test_add_delete(self):
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x, 'nodes':[], 'edges':[] }
        G = Graph(par)
        N = 15 # nodeset size
        K = 5 # neighbor size

        node_ct = 0

        while len(G.nodes) < 2*N:
            while(str(node_ct) in G.nodes):
                node_ct += 1
            par = {'data': 1.0, 'ident': str(node_ct)}
            n = Node(par)
            G._add_node(n)

        nodekeys = list(G.nodes.keys())
        for i in range(N):
            sig_idx = nodekeys[i]
            left_node = G.nodes[sig_idx]
            idxs = random.sample(range(N), K)
            assert len(idxs) == K
            for j in idxs:
                otk_idx = nodekeys[j+N]
                right_node = G.nodes[otk_idx]
                x = left_node.ident + "," + right_node.ident
                while x in G.edges:
                    x += ",0"
                par = {'data':1.0, 'ident':x, 'endpoints':[left_node, right_node], 'weight':0}
                e = Edge(par)
                G._add_edge(e)

        self.assertTrue(len(G.nodes) == 2*N)
        self.assertTrue(len(G.edges) == K*N)

        while (str(node_ct) in G.nodes):
            node_ct += 1
        par = {'data': 1.0, 'ident': str(node_ct)}
        n = Node(par)
        G._add_node(n)
        self.assertTrue(len(G.nodes) == 2*N + 1)
        self.assertTrue(len(G.edges) == K*N)

        while (str(node_ct) in G.nodes):
            node_ct += 1
        par = {'data': 2.0, 'ident': str(node_ct)}
        m = Node(par)
        G._add_node(m)
        self.assertTrue(len(G.nodes) == 2 * N + 2)
        self.assertTrue(len(G.edges) == K*N)

        right_node = G.nodes[list(G.nodes.keys())[random.randrange(N)]]
        left_node = G.nodes[list(G.nodes.keys())[random.randrange(N)+N]]
        x = left_node.ident + "," + right_node.ident
        while(x in G.edges):
            x += ",0"
        par = {'data': 1.0, 'ident':x, 'endpoints': [left_node, right_node], 'weight': 0}
        e = Edge(par)
        G._add_edge(e)

        self.assertTrue(len(G.edges) == K*N+1)

        ek = random.choice(list(G.edges.keys()))
        e = G.edges[ek]
        G._del_edge(e)
        self.assertTrue(ek not in G.edges)
        self.assertTrue(len(G.edges) == K*N)

        m = G.nodes[random.choice(list(G.nodes.keys()))]
        edges_lost = len(m.edges)
        G._del_node(m)

        self.assertTrue(len(G.nodes) == 2*N+1)
        self.assertTrue(len(G.edges) == K*N - edges_lost)

    def test_maximal_matching(self):
        #print("Beginning test_maximal_matching\n")
        #print("Selecting random identity for a new graph\n")
        r = random.random()
        x = str(hash(str(1.0) + str(r)))
        par = {'data': 1.0, 'ident': x, 'nodes':[], 'edges':[] }
        #print("Creating new graph\n")
        G = Graph(par)
        #print("Graph is now empty. adding nodes and edges.\n")
        N = 2 # nodeset size
        K = 2 # neighbor size

        while len(G.nodes) < 2*N:
            #print("Adding " + str(len(G.nodes)) + "-th node\n")

            #print("Selecting random identity for new node\n")
            node_ct = 0
            while(str(node_ct) in G.nodes):
                node_ct += 1
            par = {'data': 1.0, 'ident': str(node_ct)}
            #print("Creating new node.\n")
            n = Node(par)
            G._add_node(n)

        #print("adding edges")
        nodekeys = list(G.nodes.keys())
        for i in range(N):
            #print("picking k neighbors for node with index in range(N)\n")
            sig_idx = nodekeys[i]
            right_node = G.nodes[sig_idx]
            #print("Selecting neighbors of node\n")
            idxs = random.sample(range(N), K)
            #print("Selected neighbors " + str(idxs) + "\n")
            assert len(idxs) == K
            for j in idxs:
                #print("Getting neighbor node\n")
                otk_idx = nodekeys[j+N]
                left_node = G.nodes[otk_idx]
                #print("Assigning new random identity for edge\n")
                x = left_node.ident + "," + right_node.ident
                par = {'data':1.0, 'ident':x, 'endpoints':[left_node, right_node], 'weight':0}
                #print("Creating edge\n")
                e = Edge(par)
                #print("adding edge\n")
                G._add_edge(e)


        #print("Testing node set size\n")
        self.assertTrue(len(G.nodes) == 2*N)
        #print("Testing edge set size\n")
        self.assertTrue(len(G.edges) == K*N)

        #print("Computing maximal matching")
        M = G.maximal_matching()

        good_result_one = [G.edges["2,1"], G.edges["3,0"]]
        good_result_two = [G.edges["2,0"], G.edges["3,1"]]
        self.assertTrue(len(M) == len(good_result_one))
        self.assertTrue(M[0] in good_result_one or M[0] in good_result_two)
        if M[0] in good_result_one:
            self.assertTrue(M[1] in good_result_one)
        elif M[0] in good_result_two:
            self.assertTrue(M[1] in good_result_two)
        else:
            self.assertTrue(False)



        output_report = ""
        output_report +=  "K-regular bipartite graph with 2*N vertices; K = " + str(K) + "; N = " + str(N) + "\n"
        output_report += "===================================================================================\n"
        output_report += "Nodes of G:\n"
        output_report += "Right-nodes of G:\n"
        nodekeylist = list(G.nodes.keys())
        for i in range(N):
            nodekey = nodekeylist[i]
            n = G.nodes[nodekey]
            output_report += "\tNode ident :\t" + str(n.ident) + "\n"
            output_report += "\tEdge list : \n"
            for e in n.edges:
                output_report += "\t\tEdge ident :\t" + str(e.ident) + "\n"
                output_report += "\t\tEdge endpoints :\t" + str((e.endpoints[0].ident, e.endpoints[1].ident)) + "\n"
                output_report += "\t\tEdge weight :\t" + str(e.weight) + "\n\n"

        output_report += "Left-nodes of G:\n"
        nodekeylist = list(G.nodes.keys())
        for i in range(N):
            nodekey = nodekeylist[i+N]
            n = G.nodes[nodekey]
            output_report += "\tNode ident :\t" + str(n.ident) + "\n"
            output_report += "\tEdge list : \n"
            for e in n.edges:
                output_report += "\t\tEdge ident :\t" + str(e.ident) + "\n"
                output_report += "\t\tEdge endpoints :\t" + str((e.endpoints[0].ident, e.endpoints[1].ident)) + "\n"
                output_report += "\t\tEdge weight :\t" + str(e.weight) + "\n"

        output_report += "Edges of G:\n"
        for eid in G.edges:
            e = G.edges[eid]
            output_report += "\tEdge ident :\t" + str(e.ident) + "\n"
            output_report += "\tEdge endpoints :\t" + str(e.endpoints[0].ident) + ", " + str(e.endpoints[1].ident) + "\n"
            output_report += "\tEdge weight :\t" +str(e.weight) + "\n"

        output_report += "\n\n======================================================================================\n\n"
        output_report += "Matching:\n"
        for e in M:
            output_report += "\tEdge ident: \t" + e.ident + "\n"
        #print(output_report)
        with open("output.txt", "w") as wf:
            wf.write(output_report)

    def test_make_graph(self):

        for i in range(2,20):
            for r in range(2,i):
                G = make_graph(i,r)
                self.assertTrue(len(G.nodes) == 2 * i)
                self.assertTrue(len(G.edges) == r * i)
                for j in range(i):
                    right_node_key = list(G.nodes.keys())[j]
                    right_node = G.nodes[right_node_key]
                    self.assertEqual(len(right_node.edges), r)



tests = [Test_Node, Test_Edge, Test_Graph, Test_sym_dif]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))