# Bayesian Network Inference

A Python implementation of exact probabilistic inference in Bayesian Networks, built from scratch as part of my Introduction to Artificial Intelligence coursework at The Ohio State University.

The project explores how probabilistic graphical models represent dependencies between random variables and how exact inference algorithms compute posterior probabilities given observed evidence.

## Features

- **Global Joint Probability:** Computes the probability of a complete variable assignment by multiplying conditional probabilities from the network's CPTs.
- **Brute-Force Exact Inference:** Enumerates all possible variable assignments, accumulates probabilities consistent with the evidence, and applies conditional probability.
- **Recursive Tree Inference:** Evaluates factored probability expressions recursively, marginalizing over hidden variables and normalizing the resulting probabilities.
- **Custom Bayesian Networks:** Loads Boolean Bayesian Networks and their conditional probability tables from CSV files.

## How It Works

A Bayesian Network represents a joint probability distribution using a directed acyclic graph (DAG). Each variable has a conditional probability distribution based on its parents.

The global joint probability is factorized as:

\[
P(X_1,\ldots,X_n)=\prod_{i=1}^{n}P(X_i\mid Parents(X_i))
\]

To calculate a posterior probability, the inference algorithms sum over hidden variables and normalize the results:

\[
P(Q\mid E)=\alpha\sum_H P(Q,E,H)
\]

where:

- \(Q\) is the query variable.
- \(E\) represents observed evidence.
- \(H\) represents hidden variables.
- \(\alpha\) is the normalization constant.

### Brute-Force Enumeration

The brute-force algorithm generates every possible True/False assignment, calculates its joint probability, and accumulates probabilities consistent with the query and evidence.

### Recursive Tree Inference

The recursive algorithm evaluates a factorized inference expression by branching on hidden variables, summing their contributions, and multiplying conditional probability factors. It simultaneously tracks the unnormalized probabilities of the query being True and False before normalization.

## Example: Custom Bayesian Network

The project includes an eight-variable Bayesian Network (A–H) with multiple dependencies and conditional probability tables.

An example inference query is:

\[
P(D=True\mid E=True,F=True,G=True,H=True)
\]

The hidden variables A, B, and C are marginalized out.

Both inference methods produce:

```text
Brute-force: 0.1854465622453203
Recursive:   0.18544656224532036
```

The results agree within floating-point precision.

## Usage

Run exact inference using brute-force enumeration:

```bash
python3 prob_inf.py --model student_bn.csv --mode brute --query D=True -e E=True -e F=True -e G=True -e H=True
```

Run recursive-tree inference:

```bash
python3 prob_inf.py --model student_bn.csv --mode tree --query D=True -e E=True -e F=True -e G=True -e H=True
```

Print the network's conditional probability tables:

```bash
python3 prob_inf.py --model student_bn.csv --mode print
```

Generate the global joint probability table:

```bash
python3 prob_inf.py --model student_bn.csv --mode table
```

## Technologies and Concepts

- Python
- Bayesian Networks
- Probabilistic Graphical Models
- Conditional Probability Tables (CPTs)
- Exact Probabilistic Inference
- Recursive Algorithms
- Marginalization and Normalization

## Project Structure

```text
bayesian-network-inference/
├── prob_inf.py
├── student_bn.csv
├── burglary_alarm.csv
├── README.md
├── .gitignore
└── docs/
    ├── network_diagram.png
    └── inference_derivation.pdf
```

## Background

Developed for CSE 3521: Introduction to Artificial Intelligence at The Ohio State University.

The project focuses on understanding exact probabilistic inference through direct implementation rather than relying on existing Bayesian Network libraries.