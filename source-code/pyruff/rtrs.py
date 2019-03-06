# PyRuff: a dumb implementation of a sublinear ring signature scheme
#
# Use this code only for prototyping
# -- putting this code into production would be dumb
# -- assuming this code is secure would also be dumb

from dumb25519 import *


class innerProof:
    B = None
    b = None
    r = None
    randomness = None
    x = None
    
    def __init__(self, B, b, r, randomness):
        assert B == matrix_commit(b, r)

        m = len(b)
        n = len(b[0])
        self.B = B
        self.b = b
        self.r = r
        self.ra = randomness[0]
        self.rc = randomness[1]
        self.rd = randomness[2]
        self.a = pad(randomness[3])
            
        assert m == len(self.a) 
        assert n == len(self.a[0])
        
        A = matrix_commit(self.a, self.ra)
        self.A = A
        
        if n == 1:
            if m == 1:
                next_matrix = [[self.a[0][0]*(Scalar(1) - self.b[0][0]*Scalar(2))]]
                next_next_matrix = [[-self.a[0][0]*self.a[0][0]]]
            else:
                next_matrix = [[self.a[j][0]*(Scalar(1) - self.b[j][0]*Scalar(2))] for j in range(m)]
                next_next_matrix = [[-self.a[j][0]*self.a[j][0]] for j in range(m)]
        else:
            if m == 1:
                next_matrix = [[self.a[0][i]*(Scalar(1) - self.b[0][i]*Scalar(2)) for i in range(n)]]
                next_next_matrix = [[-self.a[0][i]*self.a[0][i] for i in range(n)]]
            else:
                next_matrix = [[self.a[j][i]*(Scalar(1) - self.b[j][i]*Scalar(2)) for i in range(n)] for j in range(m)]
                next_next_matrix = [[-self.a[j][i]*self.a[j][i] for i in range(n)] for j in range(m)]
                
        C = matrix_commit(next_matrix, self.rc)
        D = matrix_commit(next_next_matrix, self.rd)
        
        self.C = C
        self.D = D
        
    def __call__(self, x = None):
        m = len(self.b)
        n = len(self.b[0])

        assert self.B == matrix_commit(self.b, self.r)
        
        if x is None:
            challenge = str(self.A) + str(self.B) + str(self.C) + str(self.D)
            x = hash_to_scalar(challenge)
        
        self.f = [[self.b[j][i]*x + self.a[j][i] for i in range(n)] for j in range(m)]
        self.za = self.r*x + self.ra
        self.zc = self.rc*x + self.rd
        
        self.b = None
        self.a = None
        self.r = None
        self.ra = None
        self.rc = None
        self.rd = None
        
        return True
        
def verifyInnerProof(inp, x = None):
    A = inp.A
    B = inp.B
    C = inp.C
    D = inp.D
    f = inp.f
    za = inp.za
    zc = inp.zc
    
    if not isinstance(A, Point) or not isinstance(B, Point):
        raise TypeError
        
    if not isinstance(C, Point) or not isinstance(D, Point):
        raise TypeError

    m = len(f)
    n = len(f[0])
    
    for effRow in f:
        for entry in effRow:
            if not isinstance(entry, Scalar):
                raise TypeError
                
    if not isinstance(za, Scalar) or not isinstance(zc, Scalar):
        raise TypeError

    if x is None:
        challenge = str(A) + str(B) + str(C) + str(D)
        x = hash_to_scalar(challenge)
        
    if not isinstance(x, Scalar):
        raise typeError
        
    for j in range(m):
        jthsum = Scalar(0)
        for i in range(1,n):
            jthsum += f[j][i]
        try:
            assert f[j][0] == x - jthsum
        except:
            AssertionError
        
    EffOne = matrix_commit(f, za)
    try:
        assert x*B + A == EffOne
    except:
        raise AssertionError
    
    next_matrix = [[f[j][i]*(x-f[j][i]) for i in range(n)] for j in range(m)]
    EffTwo = matrix_commit(next_matrix, zc)    
    try:
        assert x*C + D == EffTwo
    except:
        raise AssertionError
    
    return True

def pad(x):
    assert len(x) > 1
    for t in x:
        i = len(t)
        t.append(Scalar(0))
        s = Scalar(0)
        while i > 0:
            t[i] = t[i-1]
            s = s - t[i-1]
            i = i - 1
        t[0] = s
    return x
        


def elgamal_encrypt(X, r):
    return [X + H*r, G*r]

def elgamal_commit(x,r):
    return [G*x + H*r, G*r]

# Decompose an integer with a given base
# INPUT
#   base: type int
#   n: integer to decompose; type int
#   exponent: maximum length of result; type int
# OUTPUT
#   int list
def decompose(base,n,exponent):
    if not isinstance(base,int) or not isinstance(n,int) or not isinstance(exponent,int):
        raise TypeError
    if base < 2 or n < 0 or exponent < 1:
        raise ValueError

    result = []
    for i in range(exponent-1,-1,-1):
        base_pow = base**i
        result.append(n/base_pow)
        n -= base_pow*result[-1]
    return list(reversed(result))

# Kronecker delta function
# INPUT
#   x,y: any type supporting equality testing
# OUTPUT
#   Scalar: 1 if the inputs are the same, 0 otherwise
def delta(x,y):
    try:
        if x == y:
            return Scalar(1)
        return Scalar(0)
    except:
        raise TypeError

# Scalar matrix commitment
# INPUT
#   m: matrix; list of Scalar lists
#   r: mask; type Scalar
# OUTPUT
#   Point
def matrix_commit(m,r):
    if not isinstance(r,Scalar):
        raise TypeError

    data = [[G,r]] # multiexp data
    for i in range(len(m)):
        for j in range(len(m[0])):
            if not isinstance(m[i][j],Scalar):
                raise TypeError
            data.append([hash_to_point('pyruff '+str(i)+' '+str(j)),m[i][j]])
    return multiexp(data)










# Test suite 

import random
import unittest

class TestInnerProof(unittest.TestCase):
    def test_innerProof(self):
        m = random.randrange(2, 10)
        n = random.randrange(2, 10)
        
        b = []
        for i in range(m):
            next_row = []
            ell = random.randrange(0, n)
            idx = 0
            while(idx < ell and idx < n):
                next_row.append(Scalar(0))
                idx += 1
            next_row.append(Scalar(1))
            idx += 1
            while(idx < n):
                next_row.append(Scalar(0))
                idx += 1
            b.append(next_row)
        r = random_scalar()
        B = matrix_commit(b, r)
        randomness = [random_scalar(), random_scalar(), random_scalar(), []]
        for j in range(m):
            randomness[-1].append([random_scalar()]*(n-1))
        ip = innerProof(B, b, r, randomness)
        ip()
        self.assertTrue(verifyInnerProof(ip))
        

tests = [TestInnerProof]
for test in tests:
    unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestLoader().loadTestsFromTestCase(test))
        





    
class outerProof:
    c = None
    ell = None
    r = None
    randomness = None
    
    def __init__(self, CO, ell, r, randomness, base, exponent):
        N = len(c)
        size = base**exponent
        
        assert size == N
        
        rb = randomness[0]
        rho = randomness[1]
        subrandomness = randomness[2:]
        #ra = subrandomness[0]
        #rc = subrandomness[1]
        #rd = subrandomness[2]
        a = pad(subrandomness[3])

        
        assert len(rho) == N
        
        ell_seq = decompose(base, ell, exponent)
        d_matrix = []
        for j in range(exponent):
            d_matrix.append([])
            for i in range(base):
                d_matrix[j].append(delta(ell_seq[j],i))
        B = matrix_commit(d_matrix, rb)
        
        ip = innerProof(B, d_matrix, rb, subrandomness)
        
        assert ip.a == a # Note: probably don't want these to be extractable from ip
        
        coefs = coefficients(a, ell, ell_seq)
        
        ring = []
        for k in range(exponent):
            data0 = [[H,rho[k]]]
            data1 = [[G,rho[k]]]
            for i in range(N):
                data0.append([CO[i][0], coefs[i][k]])
                data1.append([CO[i][1], coefs[i][k]])
            ring.append([multiexp(data0),multiexp(data1)])

        challenge = str(CO) + str(ip.A) + str(ip.B) + str(ip.C) + str(ip.D)
        for entry in ip.ring:
            challenge += str(entry)
        x = hash_to_scalar(challenge)
        if ip(x):
            z = r*x**exponent
            for i in range(m):
                z -= rho[i]*x**i
                
        self.A = ip.A
        self.B = ip.B
        self.C = ip.C
        self.D = ip.D
        self.ring = ring
        self.f = ip.f
        self.za = ip.za
        self.zc = ip.zc
        self.z = z
        
        return True
