 # Matching
 
The Matching project is intended to study the traceability of output-based blockchains like that of Monero and Zcash. The core functionality of our approach utilizes graph theoretic algorithms to compute choices of self-consistent ledger histories, and uses this functionality to find a maximum-likelihood estimate of the "ground truth" ledger. Previous traceability approaches had some drawbacks in the validation of their approaches, or made unsupportable conclusions. We aim to generalize and improve those traceability analyses, as well as to provide a framework for assessing the quality of such analyses.

Right now, check out the matching-buttercup branch for the Matching folder.

 ## How far along is this project?
 
 Rounding the corner! It looks to me like graphtheory and simulator are passing all unit tests, but we are still having some errors: it appears that in playerchallenger, objects aren't being added to our graph quite appropriately, but this particular problem has not been captured by our unit tests elsewhere yet. 
 
 ### TODOs
 
 Seeking a unit test that uncovers the current problem with playerchallenger.

 ## How does it work?
 
In graphtheory.py, we present a BipartiteGraph object for constructing a bipartite graph with weighted bi-colored edges to represent a blockchain. 

This has functions for finding optimally-weighted maximal matches. The nodes on the left are one-time "output" keys (one-time keys in Monero, commitments in Zcash). The nodes on the right are authentications (ring signatures in Monero, or nullifier-SNARK pairs in Zcash). Red edges indicate input relationships, and blue edges indicate output relationships. For example, if one-time keys A, B, C are used to create ring signature S to authorize the creation of one-time keys D and E, then we have red edges (A, S), (B, S), and (C, S), and we have blue edges (D, S) and (E, S). Note that the number of edges in the Zcash case is much higher than in the Monero case.

In simulator.py, we present a Simulator object for simulating an economy from a Markov process, constructing a BipartiteGraph object representing the public version of the resulting ledger, and constructing a "ground truth" ownership dictionary underlying the ledger.

In playerchallenger.py, we present a Player object and a Challenger object. The Challenger object runs a Simulator, sending the public version of the ledger (the graph) to the Player along with some extra information about the Player's owned edges. The Player responds with an alleged matching. The task of the Player is two-fold: (i) for each right node, assign a left node as its owner, and (ii) make these assignments accurately. The Challenger then computes the confusion matrix comparing the Player's response to the ground truth, and outputs that.
 
 ### Assigning edge weights
 
 The Player always knows the reference/baseline wallet code, and therefore can formulate very accurately _most_ of the likelihood function that some edge indicated by ring membership should appear in a consistent ledger, i.e. is the true spender. The Player can do so without any null hypothesis about spending behavior at all. 
 
 Indeed, if ring members X_1, X_2, ..., X_n have ages a_1, a_2, ..., a_n, and the wallet distribution has PMF f, then the contribution to likelihood that key X_i is the true spender from the wallet code is f(X_1)f(X_2)...f(X_n)/f(X_i). If the Player has a null hypothesis PMF g, then the likelihood for X_i is exactly f(X_1)...f(X_n)g(X_i)/f(X_i).
 
 Assigning edge weights requires touching each edge.
 
 ### Assigning node ownership
 
The assignment problem or marriage problem in graph theory is well-studied. We employ a version of the Hopcroft-Karp algorithm to find a maximum matching on a weighted graph, which is to say a matching leaving none of the nodes on the right un-assigned. We then employ methods known at since before [this paper](https://dl.acm.org/citation.cfm?id=6502) to _very inefficiently_ find max-weight matching. Edge weights come from according to a likelihood function; the max weight matching is therefore the max likelihood estimate.

This part of the algorithm is highly parallelizable and has much room for improvement.

 
 ### Accurate assignment
 
The problem of [sensitivity vs. specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) and mis-interpreting [confusion tables](https://en.wikipedia.org/wiki/Confusion_matrix) leads earlier traceability approaches to be incomplete, at best. Fundamentally and concretely, with the real systems like Monero and Zcash, there are no large validation sets easily available.

That is, we cannot tell how well this approach will perform by generating totally random ledgers; we need some underlying truth governing the process, so that we can use goodness tests to determine the goodness of our approach at uncovering that truth.
 
We resolve this by simulating an economy between three parties using a simple Markov process. 

The parties are:
 (i) Alice, an unassuming cryptocurrency user, 
 (ii) Eve, a KYC/AML exchange who dominates the market and learns all identity information of all deposits and withdrawals, and
 (iii) Bob, a symbolic actor representing all users other than Eve and Alice. 
 
We need at least three parties, for otherwise Eve can merely assume all transactions she is not involved in must be Alice's.
 
We test the goodness of Eve's traceability strategy in a setting where Eve is given knowledge of most of the Markov process that simulated the ledger and must hypothesize about Alice's spend behavior. Eve is challenged to find the "ground truth" state of the ledger given her hypothesis. 

The idea is that if an adversary has sufficiently low performance at a simple traceability game, even under the generous conditions of being granted complete knowledge of the underlying Markov process as well as partial ownership knowledge of the ledger, then the system is concretely untraceable.

 ## Okay, but how do I use it?
 
 ### BipartiteGraph in graphtheory
 
 Import graphtheory.py to use a BipartiteGraph object. 
 ```python
 g = BipartiteGraph()
 ```
     
 Nodes come with tags that are NoneType by default. You can add left (right) nodes with tag left_tag (right_tag, respectively) and get its node ident in return with 
     ```python
     new_left_node_ident = g.add_node(0, left_tag) 
     new_right_node_ident = g.add_node(1, right_tag)
     ```
 Add blue (red) edges from new_left_node_ident and new_right_node_ident with weight blue_weight (red_weight) and tag blue_tag (red_tag, respectively):
     ```python 
     g.add_edge(0, (new_left_node_ident, new_right_node_ident, blue_tag), blue_weight)
     g.add_edge(1, (new_left_node_ident, new_right_node_ident, red_tag), red_weight)
     ```
You can delete nodes (edges) with node identity node_ident (or edge_ident, respectively) with
     ```python
     g.del_node(node_ident)
     g.del_edge(node_ident)
     ```
 The other functions in graphtheory.py are mainly helper functions except .optimize(0) (to get a heaviest-weight blue match) or .optimize(1) (to get a heaviest-weight red match). If you don't care about match weight so much as finding *any* maximal match (I don't know what your use-case is, you do you, boo) then you can just iterate ```python match = g.extend(match)``` until you get the same size lists in return and be done with it.
 
 ### Simulator
 
 Simulator generates a ledger and a BipartiteGraph object using Monero's approach to default privacy with a fixed ring size. Usage is simple:
     ```python
     sally = Simulator()
     sally.run()
     ```
 The resulting BipartiteGraph is stored in ```python sally.g``` which represents a Monero ledger. The "ground truth" of ownership in the ledger is stored in ```python sally.ownership```.
 
 #### TODOs:
 Right now, Simulator only uses a Monero style approach to ledgers. It is important to modify Simulator for a Zcash-style setting, where some transactions are transparent and the rest are "fully shielded," to compare the concrete traceability of the two approaches.
 
 ### PlayerChallenger
 
 Just run go(). This will automatedly run some statistical tests and output the resulting confusion matrices. This function will be modified later to provide more information, and to produce pretty graphs... once all the mechanics are working.
