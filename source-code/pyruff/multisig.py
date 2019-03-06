# Multisig: sign a message with a vector of keys
#
# Use this code only for prototyping
# -- putting this code into production would be dumb
# -- assuming this code is secure would also be dumb

from dumb25519 import *

class Multisignature:
    R = None
    s = None

    def __init__(self,R,s):
        if not isinstance(R,Point) or not isinstance(s,Scalar):
            raise TypeError
        self.R = R
        self.s = s

# Sign a message with a list of secret keys
# INPUT
#   m: message to sign; any type representable by a string
#   x: list of secret keys; type Scalar
# OUTPUT
#   Multisignature
def sign(m,x):
    if len(x) == 0:
        raise ValueError('Signature must use at least one secret key!')
    for i in x:
        if not isinstance(i,Scalar):
            raise TypeError('Secret key must be of type Scalar!')
    try:
        i = str(m)
    except:
        raise TypeError('Cannot convert message!')

    n = len(x)
    X = []
    for i in range(n):
        X.append(G*x[i])
    strX = ''.join([str(i) for i in sorted(X,key = lambda j: str(j))])

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
        c.append(hash_to_scalar(str(X[i])+str(R)+strX+str(m)))
        ss.append(rs[i]+x[i]*c[i])
        s += ss[i]

    return Multisignature(R,s)

# Verify a message with a list of public keys
# INPUT
#   m: message to verify; any type representable by a string
#   X: list of public keys; type Point
#   sig: signature; type Multisignature
def verify(m,X,sig):
    if len(X) == 0:
        raise ValueError('Signature must use at least one public key!')
    for i in X:
        if not isinstance(i,Point):
            raise TypeError('Public key must be of type Point!')
    try:
        i = str(m)
    except:
        raise TypeError('Cannot convert message!')
    if not isinstance(sig,Multisignature):
        raise TypeError('Signature must be of type Multisignature!')

    n = len(X)
    strX = ''.join([str(i) for i in sorted(X,key = lambda j: str(j))])

    c = []
    for i in range(n):
        c.append(hash_to_scalar(str(X[i])+str(sig.R)+strX+str(m)))
    SG = G*sig.s
    
    data = [[sig.R,Scalar(1)]]
    for i in range(n):
        data.append([X[i],c[i]])

    if not multiexp(data) == SG:
        raise ArithmeticError('Bad signature verification!')
