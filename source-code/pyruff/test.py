# Test suite for PyRuff

from dumb25519 import *
import random
import unittest
import pyruff
import multisig

class TestPyRuff(unittest.TestCase):
    def test_verify1(self):
        b = []
        b.append([Scalar(1),Scalar(0)])
        b.append([Scalar(0),Scalar(1)])
        r = Scalar(1)
        proof1 = pyruff.prove1(b,r)

        B = pyruff.matrix_commit(b,r)
        pyruff.verify1(B,proof1)

    def test_verify2_1(self):
        r = Scalar(1)
        s = Scalar(1)
        CO = []
        CO.append(pyruff.elgamal_commit(Scalar(0),r))
        CO.append(pyruff.elgamal_commit(Scalar(1),s))
        ii = 0
        base = 2
        exponent = 1
        proof2 = pyruff.prove2(CO,ii,r,base,exponent)
        pyruff.verify2(base,proof2,CO)

    def test_verify2_2(self):
        r = Scalar(1)
        CO = []
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(0),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        CO.append(pyruff.elgamal_commit(Scalar(1),Scalar(1)))
        ii = 5
        base = 3
        exponent = 2
        proof2 = pyruff.prove2(CO,ii,r,base,exponent)
        pyruff.verify2(base,proof2,CO)

    def test_verify2_3(self):
        r = Scalar(1)
        for k in range(1,4):
            CO = []
            for i in range(2**k):
                CO.append(pyruff.elgamal_commit(Scalar(i),Scalar(1)))
            ii = 0
            base = 2
            exponent = k
            proof2 = pyruff.prove2(CO,ii,r,base,exponent)
            pyruff.verify2(base,proof2,CO)

    def test_2_1_1(self):
        base = 2
        exponent = 1
        inputs = 1
        size = base**exponent # ring size

        sp = pyruff.SpendInput()
        sp.base = base
        sp.exponent = exponent
        
        # prepare the spent inputs
        input_list = [pyruff.Output(Scalar(10))]
        output_list = []

        # prepare the outputs
        output_list.append(pyruff.Output(Scalar(4)))
        output_list.append(pyruff.Output(Scalar(6)))

        sp.ii = random.randrange(0,size)

        # prepare input commitments
        input_commits = []
        for j in range(inputs):
            input_commits.append([])
            for i in range(size):
                if i == sp.ii:
                    input_commits[j].append(input_list[j].CO)
                else:
                    input_commits[j].append(random_point())

        # ring commitments
        sp.CO = []
        for i in range(size):
            sp.CO.append(input_commits[0][i])
            for j in range(1,inputs):
                sp.CO[i] += input_commits[j][i]
            for k in range(len(output_list)):
                sp.CO[i] -= output_list[k].CO

        sp.PK = []
        sp.sk = []
        sp.KI = []

        for j in range(inputs):
            sp.PK.append([])
            for i in range(size):
                if i == sp.ii:
                    sp.PK[j].append(input_list[j].PK)
                else:
                    sp.PK[j].append([random_point(),random_point()])
            sp.sk.append(input_list[j].sk)
            sp.KI.append(input_list[j].KI)

        # message
        sp.m = hash_to_scalar('test message')

        sp.s = Scalar(0)
        for i in range(inputs):
            sp.s += input_list[i].mask
        for i in range(len(output_list)):
            sp.s -= output_list[i].mask

        sig = pyruff.spend(sp)
        pyruff.verify(sp.KI,sp.PK,sp.CO,sig.CO1,sp.m,sig)

class TestMultisig(unittest.TestCase):
    def test_1(self):
        x = [random_scalar()]*1
        X = [G*i for i in x]
        m = hash_to_scalar('test message')
        multisig.verify(m,X,multisig.sign(m,x))

    def test_2(self):
        x = [random_scalar()]*2
        X = [G*i for i in x]
        m = hash_to_scalar('test message')
        multisig.verify(m,X,multisig.sign(m,x))

    def test_2_order(self):
        x = [random_scalar()]*2
        X = list(reversed([G*i for i in x]))
        m = hash_to_scalar('test message')
        multisig.verify(m,X,multisig.sign(m,x))

    def test_2_bad(self):
        x = [random_scalar()]*2
        X = [random_point() for i in x]
        m = hash_to_scalar('test message')
        with self.assertRaises(ArithmeticError):
            multisig.verify(m,X,multisig.sign(m,x))

class TestMatrixCommit(unittest.TestCase):
    def test_1_1(self):
        m = [[random_scalar()]]
        r = random_scalar()
        result = hash_to_point('pyruff 0 0')*m[0][0] + G*r
        self.assertEqual(pyruff.matrix_commit(m,r),result)

    def test_2_1(self):
        m = [[random_scalar()],[random_scalar()]]
        r = random_scalar()
        result = hash_to_point('pyruff 0 0')*m[0][0] + hash_to_point('pyruff 1 0')*m[1][0] + G*r
        self.assertEqual(pyruff.matrix_commit(m,r),result)

    def test_1_2(self):
        m = [[random_scalar(),random_scalar()]]
        r = random_scalar()
        result = hash_to_point('pyruff 0 0')*m[0][0] + hash_to_point('pyruff 0 1')*m[0][1] + G*r
        self.assertEqual(pyruff.matrix_commit(m,r),result)

class TestProduct(unittest.TestCase):
    def test_1(self):
        c = [random_scalar()]
        d = [Scalar(0)]
        result = [Scalar(0)]
        self.assertEqual(pyruff.product(c,d),result)
        self.assertEqual(pyruff.product(d,c),result)

        d = [random_scalar()]
        result = [c[0]*d[0]]
        self.assertEqual(pyruff.product(c,d),result)
        self.assertEqual(pyruff.product(d,c),result)

    def test_2(self):
        c = [random_scalar()]*2
        d = [random_scalar()]*2
        result = [c[0]*d[0],c[0]*d[1]+c[1]*d[0],c[1]*d[1]]
        self.assertEqual(pyruff.product(c,d),result)
        self.assertEqual(pyruff.product(d,c),result)

    def test_3(self):
        c = [random_scalar()]*2
        d = [random_scalar()]*3
        result = [c[0]*d[0],c[0]*d[1]+c[1]*d[0],c[1]*d[1]+c[0]*d[2],c[1]*d[2],Scalar(0)]
        self.assertEqual(pyruff.product(c,d),result)
        self.assertEqual(pyruff.product(d,c),result)

tests = [TestMultisig,TestMatrixCommit,TestProduct,TestPyRuff]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
