 # Matching
 
The Matching project is intended to study the traceability of output-based blockchains like that of Monero and Zcash. The core functionality of our approach utilizes graph theoretic algorithms to compute choices of self-consistent ledger histories, and uses this functionality to find a maximum-likelihood estimate of the "ground truth" ledger. Previous traceability approaches had some drawbacks in the validation of their approaches, or made unsupportable conclusions. We aim to generalize and improve those traceability analyses, as well as to provide a framework for assessing the quality of such analyses.

Right now, check out the matching-buttercup branch for the Matching folder.

 ## How does it work?
 
 In graphtheory.py, we present a BipartiteGraph object for constructing a bipartite graph with bi-colored edges to represent a blockchain. The nodes on the left are one-time keys in Monero, or commitments in Zcash. The nodes on the right are ring signatures in Monero, or nullifier-SNARK pairs in Zcash. Red edges indicate input relationships, and blue edges indicate output relationships.
 
 For example, if one-time keys A, B, C are used to create ring signature S to authorize the creation of one-time keys D and E, then we have red edges (A, S), (B, S), and (C, S), and we have blue edges (D, S) and (E, S). Note that the number of edges in the Zcash case is much higher than in the Monero case.
 
 The task of traceability is two-fold: (i) for each right node, assign a left node as its owner, and (ii) make these assignments accurately.
 
 ### Assigning node ownership
 
 The assignment problem or marriage problem in graph theory is well-studied. We employ a version of the Hopcroft-Karp algorithm to find a maximal matching, which is to say a matching leaving none of the nodes on the right un-assigned. We then employ methods known at since before [this paper](https://dl.acm.org/citation.cfm?id=6502) to _very inefficiently_ find max-weight matching. We lastly weight each edge according to a likelihood function; the max weight matching is therefore the max likelihood estimate.
 
 ### Accurate assignment
 
 The problem of [sensitivity vs. specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) and mis-interpreting [confusion tables](https://en.wikipedia.org/wiki/Confusion_matrix) leads earlier traceability approaches to be incomplete, at best. Fundamentally and concretely, with the real systems like Monero and Zcash, there are no large validation sets easily available.
 
 We resolve this by simulating an economy between three parties using a simple Markov process. The parties are:
 (i) Alice, an unassuming cryptocurrency user, (ii), Eve, a KYC/AML exchange who dominates the market and learns all identity information of all deposits and withdrawals, and (iii), Bob, a symbolic actor representing all users other than Eve and Alice. We need at least three parties, for otherwise Eve can merely assume all transactions she is not involved in must be Alice's.
 
 We test the goodness of Eve's traceability strategy in a setting where Eve is given knowledge of most of the Markov process that simulated the ledger and must hypothesize about Alice's spend behavior. Eve is challenged to find the "ground truth" state of the ledger given her hypothesis.
 
 Note that Eve can use future data, or can reserve some of her KYC/AML data, in order to develop validation sets. This allows Eve to use Bayesian updating to form better hypotheses about Alice's spend behavior.
 
 ## Okay, but how do I use it?
 
 ### BipartiteGraph
 
 As long as you import graphtheory.py, you can create a BipartiteGraph object. You can add left nodes with .add_node(0) and right_nodes with .add_node(1). All edges exist and have weight zero, but you can re-weight a blue edge with edge identity eid to weight w by using .add_edge(0, eid, w), and you can do so with red edges using .add_edge(1, eid, w). You can delete node with node identity x with .del_node(x) (using .del_edge just resets the weight to zero).
 
 The other functions in graphtheory.py are mainly helper functions except .optimize(0) (to get a heaviest-weight blue match) or .optimize(1) (to get a heaviest-weight red match). If you don't care about match weight so much as finding *any* maximal match (I don't know what your use-case is, you do you, boo) then you can just iterate match = g.extend(match) until you get the same size lists in return.
 
 ### Simulator
 
 Simulator is not yet fully working, but constructing a ledger should require only using .run(). The simulator object will store the "ground truth" state of the ledger in its .ownership dictionary. You can use the ledger however you like.
 
 # This seems incomplete.
 
 We need two additional programs, which are incoming. 
 
 The first program, which we are going to call Challenger, will do this: (i) run simulator given some hypothesis on Alice's behavior, (ii) sends the simulated ledger to the matching algorithm *only* (leaving ownership data behind and (iii) uses the matching algorithm's result plus the simulator's knowledge of the ground truth of the ledger to compute a confusion table/matrix together with some relevant statistics.
 
 The second program, which we are going to call BigPictureExplorer, will do this: (i) generate a large family of Markov processes and hypotheses on Alice's behavior, (ii) run Challenger using these, and (iii) explore the larger-scale parameter space, looking for modes of Alice's behavior that make it difficult for Eve to distinguish her transactions from Bob's.
