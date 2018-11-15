import timeit

def timing_opt_matching(par):
    min_nodes = par['min_nodes']
    max_nodes = par['max_nodes']
    min_regularity = par['min_regularity'] # regularity = ring size
    max_regularity = par['max_regularity']
    sample_size = par['sample_size']

    assert min_regularity >= 2
    assert min_regularity <= max_nodes
    assert max_regularity >= min_regularity
    assert max_regularity <= max_nodes
    assert max_nodes >= min_nodes
    assert min_nodes >= 3

    result = {}
    for i in range(min_nodes,max_nodes+1):
        print(i)
        for r in range(min_regularity,min(max_regularity+1,i+1)):
            result.update({(i, r): {}})
            #print("Timing make_graph(i,r) for i=" + str(i) + ", r=" + str(r))
            stp = 'from graphtheory import sym_dif, Node, Edge, BipartiteGraph, make_graph; i='+str(i)+'; r='+str(r)
            t = timeit.timeit('make_graph(i,r,wt=\"random\")', stp, number=sample_size)
            #result[(i,r)].update({'gentime':t})

            #print("Timing make_graph(i,r).maximal_matching() for i=" + str(i) + ", r=" + str(r))
            t = timeit.timeit('make_graph(i, r, wt=\"random\").opt_matching()', stp, number=sample_size)
            result[(i, r)].update({'matchtime':t})
    return result

par = {'max_nodes': 20, 'sample_size': 1, 'min_nodes': 11, 'min_regularity': 11, 'max_regularity': 11}
with open("output.txt", "w") as wf:
    results = timing_opt_matching(par)
    for entry in results:
        wf.write(str(entry[0])+","+str(entry[1])+","+str(results[entry]['matchtime'])+"\n")

T = sorted(list(results.keys()))
Y = [[results[t]['matchtime']] for t in T]
#EY = float(sum([y[0] for y in Y]))/float(len(Y))
#Z = [[y[0] - EY] for y in Y]
X = [[t[1]*t[0]**1.5, 1.0] for t in T]
#XTZ = [sum([X[j][0]*Z[j][0] for j in range(len(Z))]), sum([X[j][1]*Z[j][0] for j in range(len(Z))])]
XTY = [sum([X[j][0]*Y[j][0] for j in range(len(Y))]), sum([X[j][1]*Y[j][0] for j in range(len(Y))])]
x2 = sum([x[0]*x[0] for x in X])
simplesum = sum([x[0] for x in X])
n = len(X)
XTX = [[x2, simplesum], [simplesum, n]]
c = 1.0/float(float(n)*float(x2) - float(simplesum)*float(simplesum))
XTXI = [[float(n)*c, -1.0*float(simplesum)*c], [-1.0*float(simplesum)*c, float(x2)*c]]
#parameters = [XTXI[0][0]*XTZ[0] + XTXI[0][1]*XTZ[1],XTXI[1][0]*XTZ[0] + XTXI[1][1]*XTZ[1]]
parameters = [XTXI[0][0]*XTY[0] + XTXI[0][1]*XTY[1],XTXI[1][0]*XTY[0] + XTXI[1][1]*XTY[1]]
#print(parameters)
print("the approximate time it takes to find any maximal matching on an r-regular bipartite graph with 2*N  nodes on my computer is approximately a*r*N**1.5 (up to a constant) where a = " + str(parameters[0]))





