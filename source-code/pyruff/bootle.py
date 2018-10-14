# PyBootle: a dumb implementation of a sublinear zkpok
#
# Use this code only for prototyping
# -- putting this code into production would be dumb
# -- assuming this code is secure would also be dumb


from dumb25519 import *


class SecretKey:
    x = None
    y = None

    def __init__(self, x, y):
        if not isinstance(r,Scalar) or not isinstance(r1,Scalar):
            raise TypeError
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x)+str(self.y)
        
class InnerProof:
    A = None
    B = None
    C = None
    D = None
    eff = None
    za = None
    zc = None
    
    b = None
    r = None
    randomness = None
    ra = None
    rc = None
    rd = None
    a = None
    
    # Usage:
    # construct b, pick r, pick randomness
    # ip = InnerProof(b, r, randomness)
    # msg = str(ip.A) + str(ip.C) + str(ip.D)
    # challenge = hash_to_scalar(msg)
    # ip.finalize(b, r, randomness, challenge)
    # trash b, r, randomness
    # Now [ ip.A, ip.C, ip.D, ip.eff, ip.za, ip.zc ] are the proof
    # Can call ip.verify(B, x, A, C, D, eff, za, zc)
    
    def __init__(self, b=None, r=None, randomness=None):
        if b is None and r is None and randomness is None:
            # In this case, only being used for verification.
            self.A = None
            self.B = None
            self.C = None
            self.D = None
            self.eff = None
            self.za = None
            self.zc = None
        ra = randomness['ra']
        rc = randomness['rc']
        rd = randomness['rd']
        a = randomness['a']
        
        extra = []
        for j in range(len(a)):
            print(a)
            extra.append([-sum(a[j][i] for i in range(len(a[j])))])
            for i in range(len(a[j])):
                extra[-1].append(a[j][i])
        a = extra
        A = matrix_commit(a, ra)
        B = matrix_commit(b, r)
        
        c = a
        d = a
        for j in range(len(a)):
            for i in range(len(a[j])):
                c[j][i] = a[j][i]*(1-2*b[j][i])
                d[j][i] = -a[j][i]*a[j][i]
        C = matrix_commit(c, rc)
        D = matrix_commit(d, rd)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.eff = [None for (j,i) in (range(len(a)),range(len(a[0])))]
        self.za = None
        self.zc = None
        
    def finalize(self, b, r, randomness, challenge):
        eff = a
        for j in range(len(a)):
            for i in range(len(a[j])):
                eff[j][i] = b[j][i]*challenge + a[j][i]
        self.eff = eff
        
        za = r*challenge + ra
        zc = rc*challenge + rd
        self.za = za
        self.zc = zc
        
    def verify(self, B, challenge, A, C, D, eff, za, zc):
        result = None
        for j in range(len(eff)):
            if eff[j][0] == challenge - sum([eff[j][i] for i in range(1,n)]):
                result = 1
            else:
                result = 0
                break
        EffUno = matrix_commit(eff, za)
        temp = []
        for j in range(len(eff)):
            temp.append([])
            for i in range(len(eff[j])):
                temp[-1].append(eff[j][i]*(challenge-eff[j][i]))
        # assumes the f matrix is square, which may not be correct?
        EffDos = matrix_commit(temp, zc)
        if challenge*B + A == EffUno and challenge*C + D == EffDos:
            result = 1
        else:
            result = 0
        return result
    
class OuterProof:
    CO = None
    ell = None
    r = None
    randomness = None
    base = None
    exponent = None
    
    # USAGE:
    # pick sequence of commitments CO
    # pick secret index ell
    # pick random r
    # pick randomness
    # set base and exponent
    # op = OuterProof(CO, ell, r, randomness, base, exponent)
    # To verify:
    #   op = OuterProof(CO, ell, r, randomness, base, exponent)
    #   verifier = OuterProof()
    #   print(verifier.verify(CO, op))
    
    def __init__(self, CO=None, ell=None, r=None, randomness=None, base=None, exponent=None):
        if CO is None and ell is None and r is None and randomness is None and base is None and exponent is None:
            # in this case, only being used for verification.
            self.ip = InnerProof()
            self.CO = None
            self.base = None
            self.exponent = None
            self.B = None
            self.A = None
            self.C = None
            self.D = None
            self.Gk = None
            self.eff = None
            self.za = None
            self.zc = None
        else:
            self.base = base
            self.exponent = exponent
            self.CO = CO
            
            rb = randomness['rb']
            rho = randomness['rho']
            subrandomness = randomness['sub']
            idx_seq = decompose(base, ell, exponent)

            d = []
            for j in range(exponent):
                d.append([])
                for i in range(base):
                    d[j].append(delta(idx_seq[j],i))
            B = matrix_commit(d, rb)
            
            ip = InnerProof(d, rb, subrandomness)
            A = ip.A
            C = ip.C
            D = ip.D
            
            a = subrandomness['a']
            extra = []
            for j in a:
                extra.append([-sum(j)])
                for i in j:
                    extra[-1].append(i)
            a = extra
            
            coefs = coefficients(a, ell, idx_seq)
            
            Gk = []
            for k in range(exponent):
                data0 = [[H, rho[k]]]
                data1 = [[G, rho[k]]]
                for i in range(size):
                    data0.append([CO[i][0], coefs[i][k]])
                    data1.append([CO[i][1], coefs[i][k]])
                Gk.append([multiexp(data0),multiexp(data1)])
                
            x = hash_to_scalar(str(A) + str(B) + str(C) + str(D) + str(Gk))
            ip.finalize(d, rb, subrandomness, x)
            self.ip = ip
            
            z = r*x**exponent
            for k in range(exponent):
                z -= rho[k]*x**k
                
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.Gk = Gk
            self.eff = ip.eff
            self.za = ip.za
            self.zc = ip.zc
            self.z = z
    
    def verify(self, CO, op):
        A = op.A
        B = op.B
        C = op.C
        D = op.D
        Gk = op.Gk
        eff = op.eff
        za = op.ip.za
        zc = op.ip.zc
        z = op.z
        msg = str(CO) + str(A) + str(B) + str(C) + str(D) + str(Gk)
        x = hash_to_scalar(msg)
        result = None
        ip = InnerProof()
        if ip.verify(B, x, A, C, D, eff, za, zc) == 1:
            result = 1
        else:
            result = 0
            
        for j in range(len(eff)):
            if eff[j][0] == x - sum([eff[j][i] for i in range(1,len(eff[j]))]):
                result = result * 1
            else:
                result = result * 0
                break
                
        g = []
        g.append(eff[0][0])
        for j in range(1, exponent):
            g[0] *= self.eff[j][0]

        data0 = [[CO[0][0], g[0]]]
        data1 = [[CO[0][1], g[0]]]
        for i in range(1, base**exponent):
            idx_seq = decompose(base, i, exponent)
            g.append(self.eff[0][idx_seq[0]])
            for j in range(1, exponent):
                g[i] *= self.eff[j][idx_seq[j]]
            data0.append([CO[i][0], g[i]])
            data1.append([CO[i][1], g[i]])

        for k in range(exponent):
            data0.append([self.Gk[k][0],-x**k])
            data1.append([self.Gk[k][1],-x**k])

        if not [multiexp(data0),multiexp(data1)] == elgamal_encrypt(Z,proof.z):
            raise ArithmeticError('Failed verify2!')
        
        

        
      
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

def elgamal_encrypt(X,r):
    return [H*r+X, G*r]

def elgamal_commit(x,r):
    return [G*x+H*r, G*r]
    
    
import unittest


class TestPyRuff(unittest.TestCase):
    def test_two(self):
        N = 6
        base = 2
        exponent = 3
        CO = []
        for i in range(5):
            CO.append(elgamal_commit(random_scalar(), random_scalar()))
        CO.append(elgamal_commit(Scalar(0), Scalar(7))) # Everyone's lucky number
        ell = 5
        r = random_scalar()
        
        randomness = {}
        randomness.update({'rb':random_scalar(), 'rho':[random_scalar()]*N, 'sub':{'ra':random_scalar(), 'rc':random_scalar(), 'rd':random_scalar(), 'a':[random_scalar() for (j,i) in zip(range(base), range(exponent))]}})
        op = OuterProof(CO, ell, r, randomness, base, exponent)
        
        vicky = OuterProof()
        print(vicky.verify(CO, op))
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestPyRuff)
unittest.TextTestRunner(verbosity=1).run(suite)
