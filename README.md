# CG-par_machine_sched
The goal of this repository is to reproduce "Parallel Machine Scheduling by Column Generation" paper.
This project is made for a CS course and, in addition to the original [paper](https://core.ac.uk/download/pdf/80112183.pdf), my analysis of the article and slideshow in Italian are made available into the doc folder.
However the code is well commented.

## Dependencies
* Python 3.6
* Numpy 1.19.5
* Gurobi 9.1 (license needed)

## File description
- config.toml: contains hyperparameters configuration;
- benchmark.py: generate random benchmarks or read them from file;
- utilities.py: contains outline functionality;
- optimization.py: contains the implementation of heuristics and column generation including the pricing algorithm;
- \_\_main__.py: main file.

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