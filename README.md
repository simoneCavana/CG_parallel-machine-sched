# CG-par_machine_sched
The goal of this repository is to reproduce "Parallel Machine Scheduling by Column Generation" paper.
This project is made for a CS course and, in addition to the original [paper](https://core.ac.uk/download/pdf/80112183.pdf), my analysis of the article and slideshow in italian are made available into the doc folder.
However, the code is well commented in english.

## Dependencies
* Python 3.6
* Toml 0.10.2
* Numpy 1.19.5
* Scipy 1.5.4
* Gurobi 9.1 (license needed)

## File description
- config.toml: contains hyper-parameters configuration;
- benchmark.py: generate random benchmarks or read them from file;
- utilities.py: contains outline functionality;
- heuristic.py: contains the implementation of heuristics;
- optimization.py: contains the implementation the column generation including the pricing algorithm;
- \_\_main__.py: the main file.

## Execution instruction
The main script needs two arguments, the first one is mandatory and represent the type of benchmark to work on (beb, rnd), while the second one is the verbose option (-v) to print results on console and is an optional argument.   

### Example
For a single verbose execution on Barnes & Brennan benchmark:
```
python __main__.py beb -v
```
For a multi-execution on 25 random benchmark without printing output on console:
```
for i in {1..25}; do python __main__.py rnd; done
```

## To Do:
- implement Branch & Bound on time interval;
- implement the final optimization to the pricing subproblem.