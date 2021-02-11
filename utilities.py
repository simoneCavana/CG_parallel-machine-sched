import numpy as np

"""
 given the schedule and all job's weights and processing time
 return the array of jobs indices ordered by Smith's rule ( w/p )
 It works either for a subset of jobs or all the jobs
 
 s: array schedule to order
 p: array of the processing time of all jobs
 w: array of the weights of all job
 OUTPUT: jobs ordered by smith rule
"""


def smith_order(s, p, w):
    rate = w[s] / p[s]
    order = rate.argsort()[::-1]             # get indexes of rate into decreasing order
    return s[order]


"""
 compute the vector of schedule costs as the sum of the completion time of each
 job multiplied by it's weight

 s: array of the jobs into the schedule
 p: array of the processing time of all jobs
 w: array of the weights of all job
 order: if True reorder the s schedule with Smiths rule
 OUTPUT: schedule cost
"""


def sched_cost(s, p, w, order=False):
    if order:
        s = smith_order(s, p, w)
    cj = np.cumsum(p[s])
    return cj @ w[s]


"""
 given the dictionary of the best solution found by the heuristic + NS
 create the a_{js} matrix to make the solution useful for the master problem
 
 d: dictionary, key-int, value-list of schedule's solution
 j: # of jobs
 m: # of machine
 OUTPUT: sparse int matrix {0,1} of J x 10M
"""


def create_job_sched_matrix(d, j, m):
    A = np.empty((j, 1), dtype=np.uint16)

    for solution in d.values():
        z = np.zeros((j, m), dtype=np.uint16)
        for i in range(m):
            z[solution[i], i] = 1
            # [np.put(z[:, i], solution[i], 1) for i in range(m)]
        A = np.concatenate((A, z), axis=1)

    # remove the first column caused by the initialization
    return A[:, 1:]


"""
 given an instance return it's Upper Bound as the sum of the completion time
 of all job (C_j(s)) multiplied by it's weights as they execute on a single machine ordered by smith 
"""


def serial_cost_ub(benchmark):
    # return sched_cost(np.arange(benchmark['n']), benchmark['p'], benchmark['w'])      # ordered by index
    return sched_cost(smith_order(np.arange(benchmark['n']), benchmark['p'], benchmark['w']),
                      benchmark['p'], benchmark['w'])


"""
 processing lower and upper bound on m machine (see the paper for more details)
"""


def processing_bound(benchmark):
    pmax = max(benchmark['p'])
    hmin = (sum(benchmark['p']) - (benchmark['m'] - 1) * pmax) / benchmark['m']
    hmax = (sum(benchmark['p']) + (benchmark['m'] - 1) * pmax) / benchmark['m']
    return hmin, hmax


"""
 
"""


def print_solution(benchmark, model, A):
    # get the current schedule selected by the master problem and the total cost relative to them
    hmin, hmax = processing_bound(benchmark)
    sel_sched_idx = np.where(model.X)[0]
    sel_sched = [smith_order(np.where(A[:, s])[0], benchmark["p"], benchmark["w"]) for s in sel_sched_idx]
    first_part = (f'\nmachine: {benchmark["m"]}\njobs: {benchmark["n"]}'
                  f'\nw: {benchmark["w"]}'
                  f'\np: {benchmark["p"]}\n',
                  f'\n\t+++RESULTS+++',
                  f'\noptimum total cost: {model.objVal}',
                  f'\nserial schedule cost - UB: {serial_cost_ub(benchmark)}',
                  f'\nprocessing time LB: {hmin}\nprocessing time UB: {hmax}',
                  f'\nselected schedule: {list(sel_sched_idx)}')
    scnd_part = [f'\n\tM{i} <- {list(s)},\tsum(p_j)={sum(benchmark["p"][s])},'
                 f'\tc_s={sched_cost(s, benchmark["p"], benchmark["w"])}' for i, s in enumerate(sel_sched)]
    return "".join(list(first_part) + scnd_part)


"""

"""


def print_timing(n_iter, htime, btime, tot_time):
    timing = (f'\nTotal iteration: {n_iter}',
              f'\nHeuristic processing time: {int(htime / 60)} min {htime % 60} s',
              f'\nBranch & Bound processing time: {int(btime / 60)} min {btime % 60} s',
              f'\nTotal time elapsed: {int(tot_time / 60)} min {tot_time % 60} s\n')

    return "".join(timing)


"""
 this function aim to print all the result into a single file
 the file are named by the type of benchmark (beb, rnd), the number of machines and the number of jobs
 
 path: where to write the file
"""


def write2file(benchmark, master, A, n_iter, htime, btime, tot_time, file_type, path):
    sol = print_solution(benchmark, master, A)
    tim = print_timing(n_iter, htime, btime, tot_time)
    sep = "".join(['-'] * 50)

    file_name = path + file_type + str(benchmark['m']) + '_' + str(benchmark['n']) + '.txt'
    with open(file_name, 'a') as file:
        file.write("".join(sol + '\n' + tim + '\n' + sep))


"""
 make the iteration distribution plot; I've tried to use roughviz into a Jupyter notebook.
 so copy and paste the code below into a jupyter notebook cell and that's it!
 I have reported the code here for completeness.  
"""


# def plot_iter_dist():
#     from roughviz.charts import Bar
#     bar = Bar(data={"labels": ["[5, 20]", "30", "40", "50"],
#                     "values": [5000, 4000, 3000, 2000]},
#                     "values": [2000, 3000, 4000, 5000]},
#               title="Heuristic iterations distribution", title_fontsize=3)
#     bar.set_xlabel("Jobs", fontsize=2)
#     bar.set_ylabel("Iteration", fontsize=2)
#     bar.set_figsize((1200, 600))
#     bar.show()
