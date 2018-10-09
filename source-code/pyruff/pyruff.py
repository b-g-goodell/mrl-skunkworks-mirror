# PyRuff: a dumb implementation of a sublinear ring signature scheme
#
# Use this code only for prototyping
# -- putting this code into production would be dumb
# -- assuming this code is secure would also be dumb

from dumb25519 import *
import random

class SecretKey:
    r = None
    r1 = None

    def __init__(self,r,r1):
        self.r = r
        self.r1 = r1

    def __str__(self):
        return str(self.r)+str(self.r1)

class Output:
    sk = None
    KI = None
    PK = None

    mask = None
    CO = None
    amount = None

    def __init__(self,amount):
        self.amount = amount
        self.mask = random_scalar()
        self.CO = G*amount + H*self.mask
        self.sk = SecretKey(random_scalar(),random_scalar())
        self.KI = G*self.sk.r1
        self.PK = [G*self.sk.r+self.KI,H*self.sk.r]

    def __str__(self):
        return str(self.amount)+str(self.mask)+str(self.CO)+str(self.sk)+str(self.KI)+str(self.PK)

class F:
    KI = None # key image
    PK = None # public key matrix
    CO = None # commitment vector
    CO1 = None # commitment
    m = None # message

    def __init__(self,KI,PK,CO,CO1,m):
        self.KI = KI
        self.PK = PK
        self.CO = CO
        self.CO1 = CO1
        self.m = m

    def __str__(self):
        return str(self.KI) + str(self.PK) + str(self.CO) + str(self.CO1) + str(self.m)

class Proof1:
    A = None
    C = None
    D = None
    f_trim = None
    zA = None
    zC = None
    a = None

    def __init__(self,A,C,D,f_trim,zA,zC,a):
        self.A = A
        self.C = C
        self.D = D
        self.f_trim = f_trim
        self.zA = zA
        self.zC = zC
        self.a = a

    def __str__(self):
        return str(self.A)+str(self.C)+str(self.D)+str(self.f_trim)+str(self.zA)+str(self.zC)+str(self.a)

class Proof2:
    proof1 = None
    B = None
    G1 = None
    z = None

    def __init__(self,proof1,B,G1,z):
        self.proof1 = proof1
        self.B = B
        self.G1 = G1
        self.z = z

    def __str__(self):
        return str(self.proof1)+str(self.B)+str(self.G1)+str(self.z)

class SpendInput:
    ii = None
    PK = None
    sk = None
    KI = None
    CO = None
    m = None
    s = None
    base = None
    exponent = None

    def __init__(self):
        pass

class SpendProof:
    base = None
    exponent = None
    CO1 = None
    sigma1 = None
    sigma2 = None

    def __init__(self,base,exponent,CO1,sigma1,sigma2):
        self.base = base
        self.exponent = exponent
        self.CO1 = CO1
        self.sigma1 = sigma1
        self.sigma2 = sigma2

class Multisignature:
    R = None
    s = None

    def __init__(self,R,s):
        self.R = R
        self.s = s

def sub(f_in):
    L = len(f_in.PK) # number of inputs
    N = len(f_in.PK[0]) # ring size
    PKZ = []
    f = []
    for j in range(L):
        PKZ.append([f_in.KI[j],Z])
        f.append(hash_to_scalar(str(f_in.KI[j]) + str(f_in) + str(j)))
    C = []
    for i in range(N):
        data0 = [[f_in.CO[i],Scalar(1)]] # multiexp data (first component)
        data1 = [[f_in.CO1,Scalar(1)]] # multiexp data (second component)

        for j in range(L):
            data0.append([f_in.PK[j][i][0],f[j]])
            data0.append([PKZ[j][0],-f[j]])
            data1.append([f_in.PK[j][i][1],f[j]])
            data1.append([PKZ[j][1],-f[j]])

        C.append([multiexp(data0),multiexp(data1)])

    return C,f

def spend(s_in):
    s = s_in.s
    CO1 = G*s

    f = F(s_in.KI,s_in.PK,s_in.CO,CO1,s_in.m)
    sub_C,sub_f = sub(f)
    for i in range(len(s_in.sk)):
        s += s_in.sk[i].r*sub_f[i]

    sigma1 = prove2(sub_C,s_in.ii,s,len(s_in.PK),s_in.base,s_in.exponent)

    r1 = [s_in.sk[i].r1 for i in range(len(s_in.sk))]
    sigma2 = multisign(str(sigma1)+str(f),r1,None)
    
    return SpendProof(s_in.base,s_in.exponent,CO1,sigma1,sigma2)

def multisign(m,x,X):
    n = len(x)
    if X is None:
        X = []
        for i in range(n):
            X.append(G*x[i])

    rs = []
    r = Scalar(0)
    for i in range(n):
        rs.append(random_scalar())
        r += rs[i]

    R = G*r
    c = []
    ss = []
    s = Scalar(0)
    for i in range(n):
        c.append(hash_to_scalar(str(X[i])+str(R)+str(X)+str(m)))
        ss.append(rs[i]+x[i]*c[i])
        s += ss[i]

    return Multisignature(R,s)

def prove2(CO,ii,r,inputs,base,exponent):
    size = base**exponent
    u = [random_scalar()]*exponent

    ii_seq = decompose(base,ii,exponent)

    d = []
    for j in range(exponent):
        d.append([])
        for i in range(base):
            d[j].append(delta(ii_seq[j],i))

    rB = random_scalar()
    B = matrix_commit(d,rB)

    proof1 = prove1(d,rB)
    coefs = coefficients(proof1.a,ii,ii_seq)

    G1 = []
    for k in range(exponent):
        data0 = [[G,u[k]]] # multiexp data
        data1 = [[H,u[k]]]
        for i in range(size):
            data0.append([CO[i][0],coefs[i][k]])
            data1.append([CO[i][1],coefs[i][k]])
        G1.append([multiexp(data0),multiexp(data1)])

    x1 = hash_to_scalar(str(proof1.A)+str(proof1.C)+str(proof1.D))

    z = r*x1**exponent
    for i in range(exponent-1,-1,-1):
        z -= u[i]*x1**i

    return Proof2(proof1,B,G1,z)

def coefficients(a,ii,ii_seq):
    m = len(a) # exponent
    n = len(a[0]) # base
    size = n**m

    coefs = []
    for k in range(size):
        k_seq = decompose(n,k,m)
        coefs.append([a[0][k_seq[0]],delta(ii_seq[0],k_seq[0])])

        for j in range(1,m):
            coefs[k] = product(coefs[k],[a[j][k_seq[j]],delta(ii_seq[j],k_seq[j])])

    for k in range(size):
        coefs[k] = trim_list(coefs[k],m,m)

    return coefs

def trim_list(a,length,index):
    result = []
    for i in range(len(a)):
        if i < length:
            result.append(a[i])
        else:
            if i == index:
                if a[i] not in [Scalar(0),Scalar(1)]:
                    raise IndexError
            else:
                if a[i] != Scalar(0):
                    raise IndexError

    return result

def product(c,d):
    max_length = max([len(c),len(d)])
    result = [Scalar(0)]*(2*max_length-1)

    for i in range(max_length):
        for j in range(max_length):
            result[i+j] += c[i]*d[j]

    return result

def prove1(b,r):
    m = len(b) # exponent
    n = len(b[0]) # base

    a = []
    for j in range(m):
        a.append([Scalar(0)])
        for i in range(1,n):
            a[j].append(random_scalar())
    for j in range(m):
        for i in range(1,n):
            a[j][0] -= a[j][i]

    rA = random_scalar()
    A = matrix_commit(a,rA)

    c = []
    d = []
    for j in range(m):
        c.append([])
        d.append([])
        for i in range(n):
            c[j].append(a[j][i]*(Scalar(1) - b[j][i]*Scalar(2)))
            d[j].append(-a[j][i]**2)

    rC = random_scalar()
    rD = random_scalar()
    C = matrix_commit(c,rC)
    D = matrix_commit(d,rD)

    x = hash_to_scalar(str(A)+str(C)+str(D))

    f = []
    for j in range(m):
        f.append([])
        for i in range(n):
            f[j].append(b[j][i]*x+a[j][i])

    f_trim = []
    for j in range(m):
        f_trim.append([])
        for i in range(1,n):
            f_trim[j].append(f[j][i])

    zA = r*x+rA
    zC = rC*x+rD

    return Proof1(A,C,D,f_trim,zA,zC,a)

# Decompose an integer
def decompose(base,n,exponent):
    result = []
    for i in range(exponent-1,-1,-1):
        base_pow = base**i
        result.append(n/base_pow)
        n -= base_pow*result[-1]
    return list(reversed(result))

# A scalar delta function
def delta(x,y):
    if x == y:
        return Scalar(1)
    return Scalar(0)

# Matrix commitment
def matrix_commit(m,r):
    data = [[G,r]] # multiexp data
    for i in range(len(m)):
        for j in range(len(m[0])):
            data.append([hash_to_point('pyruff '+str(i)+' '+str(j)),m[i][j]])
    return multiexp(data)

def verify(KI,PK,CO,CO1,m,sig):
    f = F(KI,PK,CO,CO1,m)
    sub_C,sub_f = sub(f)

    if not multiverify(str(sig.sigma1)+str(f),KI,sig.sigma2):
        raise Exception('Failed multiverify!')

def multiverify(m,X,sig):
    n = len(X)

    c = []
    for i in range(n):
        c.append(hash_to_scalar(str(X[i])+str(sig.R)+str(X)+str(m)))
    SG = G*sig.s
    
    data = [[sig.R,Scalar(1)]]
    for i in range(n):
        data.append([X[i],c[i]])

    return multiexp(data) == SG
