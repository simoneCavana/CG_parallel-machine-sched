import numpy as np
import os.path

"""
the structure of the output has to be something like this
benchmark = {'m': 3, 'n': 25,
             'w': np.array([1, 1, 1, 2, 2, 4, 4, 5, 6, 6, 6, 6, 4, 4, 4, 7, 7, 7, 5, 6, 7, 8, 8, 8, 9]),
             'p': np.array([9, 8, 8, 8, 8, 8, 8, 7, 8, 7, 6, 6, 3, 3, 3, 5, 3, 3, 2, 2, 2, 2, 2, 1, 1])}
"""


"""
 open a random file from the Barnes & Brennan instances
 and return the benchmark dictionary
"""


def import_from_file(path):
    file_name = f'beb/beb_0{np.random.randint(15)+1}.txt'

    if not os.path.isfile(path+file_name):
        print('File does not exist')
    else:
        keys = ['m', 'n', 'w', 'p']
        d = dict()
        with open(path+file_name) as file:
            for k, v in zip(keys, file.read().splitlines()):
                if k in ['m', 'n']:
                    d.update({k: int(v)})
                else:
                    d.update({k: np.array([int(n) for n in list(v[1:-1].split(','))])})

    return d


"""
 m: machines [2,3,4,5]
 n: jobs [20,...,50]
 t: type of problem, how to choose the distribution for p and w (see the roman list at page 16)
 OUTPUT: dictionary benchmark to work on with column generation
"""


def gen_instance(m_values, n_values):
    rng = np.random.default_rng()
    m = rng.choice(m_values)
    n = rng.choice(n_values)
    t = rng.choice(3)

    if t == 0:
        # w from uniform distribution [10, 100]; p from uniform distribution [1, 10]
        w = rng.choice(91, n) + 10
        p = rng.choice(10, n) + 1
    elif t == 1:
        # both from uniform distribution [1, 100]
        w = rng.choice(100, n) + 1
        p = rng.choice(100, n) + 1
    elif t == 2:
        # both from uniform distribution [10, 20]
        w = rng.choice(11, n) + 10
        p = rng.choice(11, n) + 10
    else:
        return 0

    return {'m': m, 'n': n, 'w': w, 'p': p}