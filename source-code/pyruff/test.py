# Test suite for PyRuff

from dumb25519 import *
import random
import unittest
import pyruff
import multisig

class TestPyRuff(unittest.TestCase):
    def test_2_2_1(self):
        base = 2
        exponent = 2
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

unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(TestMultisig))
unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(TestPyRuff))
