from optimization import column_generation
import argparse
import toml

"""
 this file aim is to provide instruction to launch the program and
 if the right number of parameters is entered to execute the model on the specified benchmark.
 [https://docs.python.org/3/howto/argparse.html]
"""


if __name__ == "__main__":
    param = toml.load('config.toml')

    # create the parser
    parser = argparse.ArgumentParser(description='Solving Parallel Machine Scheduling with Column Generation')

    # add the arguments
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    # parser.add_argument('source', type=str, choices=['rnd', 'beb'],
    #                     help='the benchmark onto execute the algorithm (random or from Barnes & Brennan paper')

    # execute the parse_args() method for -h option and some instruction
    args = parser.parse_args()

    # call the function to solve the instance on the specified input
    # column_generation(param, args.source, args.verbose)

    # for debugging use this line and comment lines 20-21
    column_generation(param, 'rnd', True)