# Bayesian Network Inference

This project explores exact probabilistic inference in Bayesian networks. It
computes joint probabilities and posterior probabilities from Boolean networks
using brute-force enumeration and recursive tree inference. The project was
built for CSE 3521: Introduction to Artificial Intelligence.

The interactive explorer displays the networks as directed graphs and lets you
select a query, add evidence, and see the resulting probability.

## Files

- `prob_inf.py` loads networks and implements the inference algorithms and
  command-line interface.
- `explorer.py` runs the local server for the interactive explorer. `web/`
  contains the graph and interface.
- `burglary_alarm.csv` and `student_bn.csv` contain the burglary alarm network
  and an eight-variable custom network.
- `autochecker.py` checks joint probability, brute-force inference, and
  properties of the custom network. It does not test recursive inference.
- `student_bn.pdf` and `student_bn_inference.pdf` contain the custom network
  and the inference coursework. `hw3submission.zip` is the original submission
  archive, and `collaboration.txt` records the collaboration statement.

## Interactive network explorer

Run the following command from the project directory with Python 3.9 or later:

```sh
python3 explorer.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. You can use
`python3 explorer.py --port 8001` to choose another port. Press Ctrl+C in the
terminal to stop the server. The server uses the Python standard library and
runs on localhost. The browser needs an internet connection to load
Cytoscape.js and the optional fonts.

Choose a network and select a node to see its parents and conditional
probability table. You can drag nodes to rearrange the graph and scroll to zoom.
Choose a query variable and value, add any evidence, and calculate the
probability using either inference method. The graph highlights the query and
evidence in different colors. The query variable cannot also be evidence.

## How it works

A Bayesian network represents dependencies between variables with a directed
acyclic graph. Each variable has a conditional probability table (CPT) that
specifies its probability given its parents. The probability of a complete
assignment is the product of the corresponding conditional probabilities:

$$
P(X_1,\ldots,X_n)=\prod_{i=1}^{n}P(X_i\mid \mathrm{Parents}(X_i))
$$

To calculate a posterior probability, the inference algorithms sum over
variables that are neither the query nor evidence, then normalize the result.
Brute-force inference checks every possible assignment. Recursive tree
inference evaluates the factors recursively and sums over hidden variables.

For example, the custom network contains eight variables, A through H. For
`P(D=True | E=True, F=True, G=True, H=True)`, both methods sum over the hidden
variables A, B, and C and return approximately `0.18545`.

## Command-line usage

Run these commands from the project directory. For example, to calculate the
custom network query with either method:

```sh
python3 prob_inf.py --model student_bn.csv --mode brute --query D=True -e E=True -e F=True -e G=True -e H=True
python3 prob_inf.py --model student_bn.csv --mode tree --query D=True -e E=True -e F=True -e G=True -e H=True
```

To print the network's conditional probability tables or generate its global
joint probability table:

```sh
python3 prob_inf.py --model student_bn.csv --mode print
python3 prob_inf.py --model student_bn.csv --mode table
```

The `print` and `table` modes require `tabulate`, which can be installed with
`python3 -m pip install tabulate`. The command-line interface also lists an
`approx` mode, but approximate inference is not implemented.

Run the supplied checks with:

```sh
python3 autochecker.py
```

Both exact inference methods become expensive as networks grow. The recursive
method also searches permutations of hidden variables to choose a calculation
order.
