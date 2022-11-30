# Benchmark for FastAMI - A Monte Carlo Approach to the Adjustment for Chance in Clustering Comparison Metrics

This repository contains the research code for our paper *FastAMI - A Monte Carlo Approach to the Adjustment for Chance in Clustering Comparison Metrics* which will be presented at AAAI-23 in February 2023.
A standalone version of FastAMI for easier use in other projects will be released before the conference on PyPi. This benchmark version compares our implementation with the AMI in [scikit-learn](https://scikit-learn.org), the pairwise AMI [[3]](#3), and the SMI [[1]](#1) and contains a preprocessed version of the *Benchmark Suite for Clustering Algorithms - Version 1* [[3]](#3).

## Setup

To reproduce the results in our paper, you must first install Python 3.10.4 and the required dependencies:
```
pip install -r requirements.txt
```
For the direct SMI sampling, we use C code that must be compiled by executing.
```
fastami/rcont2/build.sh
```
For the clustered version of [*Benchmark Suite for Clustering Algorithms – Version 1*](https://github.com/gagolews/clustering-benchmarks) please unpack `gagolewski.zip` in
```
/data/gagolewski
```

## Running the Benchmarks

For the synthetic EMI and SMI Benchmarks execute
```
python synthetic_benchmark.py
```
The benchmarks on real datasets can be executed as follows
```
python gagolewski_benchmark.py
```
and
```
python snap_benchmark.py
```

## References

<a id="1">[1]</a>  S. Romano, J. Bailey, V. Nguyen, and K. Verspoor, “Standardized Mutual Information for Clustering Comparisons: One Step Further in Adjustment for Chance,” in Proceedings of the 31st International Conference on Machine Learning, Jun. 2014, pp. 1143–1151. Accessed: Dec. 08, 2021. [Online]. Available: https://proceedings.mlr.press/v32/romano14.html

<a id="2">[2]</a>  M. Gagolewski and others, “Benchmark Suite for Clustering Algorithms – Version 1.” 2020. doi: 10.5281/zenodo.3815066.

<a id="3">[3]</a>  D. Lazarenko and T. Bonald, “Pairwise Adjusted Mutual Information,” arXiv:2103.12641 [cs], Mar. 2021, Accessed: Sep. 16, 2021. [Online]. Available: http://arxiv.org/abs/2103.12641

