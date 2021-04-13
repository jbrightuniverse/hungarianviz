A Python 3 implementation of the Hungarian Algorithm for optimal matching in bipartite weighted graphs, now equipped with a visual rendering system.

Based on the graph theory implementation in [these notes](http://www.cse.ust.hk/~golin/COMP572/Notes/Matching.pdf) combined with the matrix interpretation in [these notes](https://montoya.econ.ubc.ca/Econ514/hungarian.pdf).

For a detailed overview, see [this Jupyter notebook](https://github.com/jbrightuniverse/Hungarian-Algorithm-No.-5/blob/main/HungarianAlgorithm.ipynb).

Forked from the original edition of the algorithm, which you should use if you require speed over readability: [https://github.com/jbrightuniverse/hungarianalg](https://github.com/jbrightuniverse/hungarianalg).

# Usage

Installation: `pip3 install git+https://github.com/jbrightuniverse/hungarianviz`

See the included Jupyter notebook for a comprehensive example.

# Issues

IMPORTANT: Currently, support for decimals in the input matrix is unstable, as at large matrix dimensions, floating point errors propagate and cause the algorithm to fail to terminate. 

It is recommended instead to linearly scale the input matrix by a power of 10 to a set of whole numbers, run the algorithm, and then scale back to the original precision at the end.
