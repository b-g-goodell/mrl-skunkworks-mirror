import unittest
import random
from Spectre import *

class TestNode(unittest.TestCase):
    def test_init(self):
        params = {'label':hash('arbitrary string'), 'payload':None, 'parents':[]}
        x = Node(params)
        self.assertEqual(x.label, hash('arbitrary string'))
        self.assertTrue(x.payload is None)
        self.assertEqual(len(x.parents),0)

class TestDAG(unittest.TestCase):
    def test_init(self):
        G = DirAcyGraph()
        self.assertEqual(len(G.nodes),0)
        self.assertEqual(len(G.leaves),0)

    def test_add_node(self):
        G = DirAcyGraph()
        params = {'label':hash('arbitrary string'), 'payload':None, 'parents':[]}
        G._add_node(params)
        self.assertEqual(len(G.nodes),1)
        self.assertEqual(len(G.leaves),1)

        params = {'label':hash('another arbitrary string'), 'payload':None, 'parents':[hash('arbitrary string')]}
        G._add_node(params)
        self.assertEqual(len(G.nodes),2)
        self.assertEqual(len(G.leaves),1)
        self.assertTrue(hash('arbitrary string') in G.nodes)
        self.assertTrue(hash('arbitrary string') not in G.leaves)
        self.assertTrue(hash('another arbitrary string') in G.nodes)
        self.assertTrue(hash('another arbitrary string') in G.leaves)

        params = {'label':str(2), 'payload':None, 'parents':[hash('arbitrary string')]}
        G._add_node(params)
        self.assertEqual(len(G.nodes),3)
        self.assertEqual(len(G.leaves),2)
        self.assertTrue(str(2) in G.nodes)
        self.assertTrue(str(2) in G.leaves)

        params = {'label':str(3), 'payload':None, 'parents':[str(2), hash('another arbitrary string')]}
        G._add_node(params)
        self.assertEqual(len(G.nodes),4)
        self.assertEqual(len(G.leaves),1)
        self.assertTrue(str(3) in G.nodes)
        self.assertTrue(str(3) in G.leaves)
        self.assertFalse(str(2) in G.leaves)
        self.assertTrue(str(2) in G.nodes)

    def test_rem_leaf(self):
        G = DirAcyGraph()
        params = {'label':str(0), 'payload':None, 'parents':[]}
        G._add_node(params)
        self.assertEqual(len(G.nodes),1)
        self.assertEqual(len(G.leaves),1)

        params = {'label':str(1), 'payload':None, 'parents':[str(0)]}
        G._add_node(params)
        self.assertEqual(len(G.nodes),2)
        self.assertEqual(len(G.leaves),1)
        self.assertTrue(str(0) in G.nodes)
        self.assertTrue(str(0) not in G.leaves)
        self.assertTrue(str(1) in G.nodes)
        self.assertTrue(str(1) in G.leaves)

        G._rem_leaf(str(1))

        self.assertEqual(len(G.nodes),1)
        self.assertEqual(len(G.leaves),1)
        self.assertTrue(str(0) in G.nodes)
        self.assertTrue(str(0) in G.leaves)
        self.assertFalse(str(1) in G.nodes)
        self.assertFalse(str(1) in G.leaves)

        params = {'label':str(1), 'payload':None, 'parents':[str(0)]}
        G._add_node(params)

        params = {'label':str(2), 'payload':None, 'parents':[str(1)]}
        G._add_node(params)

        params = {'label':str(3), 'payload':None, 'parents':[str(1)]}
        G._add_node(params)

        self.assertEqual(len(G.nodes),4)
        self.assertEqual(len(G.leaves),2)
        self.assertTrue(str(2) in G.leaves)
        self.assertTrue(str(3) in G.leaves)

        G._rem_leaf(str(2))
        self.assertEqual(len(G.nodes),3)
        self.assertEqual(len(G.leaves),1)
        self.assertFalse(str(2) in G.leaves)
        self.assertTrue(str(3) in G.leaves)

    def test_get_past(self):
        G = DirAcyGraph()
        params = {'label':str(0), 'payload':None, 'parents':[]}
        G._add_node(params)
        params = {'label':str(1), 'payload':None, 'parents':[str(0)]}
        G._add_node(params)
        params = {'label':str(2), 'payload':None, 'parents':[str(1)]}
        G._add_node(params)
        params = {'label':str(3), 'payload':None, 'parents':[str(2)]}
        G._add_node(params)
        params = {'label':str(4), 'payload':None, 'parents':[str(2)]}
        G._add_node(params)
        params = {'label':str(5), 'payload':None, 'parents':[str(3), str(4)]}
        G._add_node(params)
        params = {'label':str(6), 'payload':None, 'parents':[str(3)]}
        G._add_node(params)
        params = {'label':str(7), 'payload':None, 'parents':[str(5), str(6)]}
        G._add_node(params)

        pastZ = G._get_past(str(7))
        self.assertTrue(len(pastZ.nodes)==7)
        self.assertTrue(len(pastZ.leaves)==2)
        self.assertTrue(str(0) in pastZ.nodes)
        self.assertTrue(str(1) in pastZ.nodes)
        self.assertTrue(str(2) in pastZ.nodes)
        self.assertTrue(str(3) in pastZ.nodes)
        self.assertTrue(str(4) in pastZ.nodes)
        self.assertTrue(str(5) in pastZ.nodes)
        self.assertTrue(str(6) in pastZ.nodes)
        self.assertTrue(str(5) in pastZ.leaves)
        self.assertTrue(str(6) in pastZ.leaves)

    def test_spec(self):
        G = DirAcyGraph()
        params = {'label':str(0), 'payload':None, 'parents':[]}
        G._add_node(params)
        params = {'label':str(1), 'payload':None, 'parents':[str(0)]}
        G._add_node(params)
        params = {'label':str(2), 'payload':None, 'parents':[str(1)]}
        G._add_node(params)
        params = {'label':str(3), 'payload':None, 'parents':[str(2)]}
        G._add_node(params)
        params = {'label':str(4), 'payload':None, 'parents':[str(2)]}
        G._add_node(params)
        params = {'label':str(5), 'payload':None, 'parents':[str(3), str(4)]}
        G._add_node(params)
        params = {'label':str(6), 'payload':None, 'parents':[str(3)]}
        G._add_node(params)
        params = {'label':str(7), 'payload':None, 'parents':[str(5), str(6)]}
        G._add_node(params)

        vote = G._spec()
        assert (str(0), str(1)) in vote
        assert (str(1), str(2)) in vote
        assert (str(2), str(3)) in vote
        assert (str(3), str(4)) in vote
        assert (str(4), str(5)) in vote
        assert (str(5), str(6)) in vote
        assert (str(6), str(7)) in vote


def genGraph(params=None):
    G=DirAcyGraph()

    genesisBlockParams = {'label':str(0), 'payload':None, 'parents':[]}
    G._add_node(genesisBlockParams)
    
    nextParams = {'label':str(1), 'payload':None, 'parents':[str(0)]}
    G._add_node(nextParams)

    nextParams = {'label':str(2), 'payload':None, 'parents':[str(1)]}
    G._add_node(nextParams)

    nextParams = {'label':str(3), 'payload':None, 'parents':[str(1)]}
    G._add_node(nextParams)

    nextParams = {'label':str(4), 'payload':None, 'parents':[str(2), str(3)]}
    G._add_node(nextParams)

    nextParams = {'label':str(5), 'payload':None, 'parents':[str(3)]}
    G._add_node(nextParams)

    nextParams = {'label':str(6), 'payload':None, 'parents':[str(4), str(5)]}
    G._add_node(nextParams)
    
    nextParams = {'label':str(7), 'payload':None, 'parents':[str(3)]}
    G._add_node(nextParams)

    nextParams = {'label':str(8), 'payload':None, 'parents':[str(6), str(7)]}
    G._add_node(nextParams)

    return G
        
#class Test_Spectre(unittest.TestCase):
#    def test_one(self):
#        G = genGraph()
#        print("G nodes ", G.nodes.keys())
#        print("G leaves ", G.leaves.keys())
#        print("G.computeSpectre()", G.computeSpectre())

tests = [TestNode, TestDAG]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
