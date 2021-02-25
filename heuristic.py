from utilities import *

"""
 heuristic that initialize random schedule to perform the column generation algorithm on a restricted
 linear programming problem (RLP)
 
 benchmark: dictionary containing the instance values
 MIN_ITER: maximum number of iterations
 MAX_ITER: minimum number of iterations
 OUTPUT: binary matrix of jobs into schedule
"""


def randomized_list_heuristic(benchmark, MIN_ITER, MAX_ITER):
    rng = np.random.default_rng()   # random seed
    M = benchmark['m']

    job_ord = smith_order(np.arange(benchmark['n']), benchmark['p'], benchmark['w'])
    tmp_sol = dict()

    # set max_iter in function of the number of jobs
    # 1. distribution aimed at time
    # 2. distribution aimed at solution quality
    # 3. constant distribution, because it's really fast and we don't need this nicety
    # max_iter = MAX_ITER if benchmark['n'] <= 20 else MIN_ITER + (MAX_ITER - benchmark['n'] * 100) # 1
    max_iter = MIN_ITER if benchmark['n'] <= 20 else benchmark['n'] * 100                         # 2
    # max_iter = MAX_ITER                                                                           # 3

    for i in range(max_iter):
        # create a new solution and pseudo-random assign job to m schedule
        tmp_sol.update({i: [np.empty(0, dtype=np.uint8) for m in range(M)]})
        for idx, job in enumerate(job_ord):
            if idx < M:
                # find an empty machine and assign the job (with a random order)
                for v in rng.choice(M, M, replace=False):
                    if tmp_sol[i][v].size == 0:
                        tmp_sol[i][v] = np.append(tmp_sol[i][v], job)
                        break
            else:
                # assign the job to the first available machine (plus probability for the first three)
                tmp_list = [np.sum(benchmark['p'][msched]) for msched in tmp_sol[i]]
                tmp_list = np.array(tmp_list).argsort()
                if tmp_list.size < 3:
                    # check the single case of m=2
                    idx2assign = rng.choice(tmp_list, p=[0.8, 0.2])
                elif idx == benchmark['n']-1:
                    # check if it's the last job and assign it without probability at the first available machine
                    idx2assign = tmp_list[0]
                else:
                    idx2assign = rng.choice(tmp_list[:3], p=[0.8, 0.15, 0.05])
                tmp_sol[i][idx2assign] = np.append(tmp_sol[i][idx2assign], job)

    best_sched = np.empty(0)
    # compute solution cost based on each schedules c_s in it
    for solution in tmp_sol.values():
        res = 0
        for sched in solution:
            res += sched_cost(sched, benchmark['p'], benchmark['w'])
        best_sched = np.append(best_sched, res)

    # select 10 best solution
    # best_sched_idx = best_sched.argsort()[:10]        (1) FANCIER & FASTER WAY
    # select the 10 best DIFFERENT solution created     (2) MORE TRICKY AND TIME CONSUMING
    best_sched_idx = np.empty(0, dtype=np.int64)
    for idx_sol in best_sched.argsort():
        if len(best_sched_idx) == 10:
            # ten best different solutions found!
            break
        elif not len(best_sched_idx):
            # add first solution
            best_sched_idx = np.append(best_sched_idx, idx_sol)
        elif best_sched[idx_sol] in best_sched[best_sched_idx]:
            # if the cost of the new best solution is equal to one already selected
            # check that at least one schedule is different
            pos_same_cost = np.where(best_sched[best_sched_idx] == best_sched[idx_sol])[0]
            flag_add = True
            for check_sol in best_sched_idx[pos_same_cost]:
                c = 0
                for sched in tmp_sol[idx_sol]:
                    if np.array([np.array_equal(sched, i) for i in tmp_sol[check_sol]]).any():
                        c += 1
                if c == benchmark['m']:
                    # already into selected solution, leave this one out
                    flag_add = False
                    break
            # ENDFOR
            if flag_add:
                # if we checked all solution then we can add the new solution
                best_sched_idx = np.append(best_sched_idx, idx_sol)
        else:
            # not already at ten, not the first solution and not already selected a solution with that cost
            best_sched_idx = np.append(best_sched_idx, idx_sol)

    # call neighborhood_search on this 10 best schedule
    ns = neighborhood_search({k: v for k, v in tmp_sol.items() if k in best_sched_idx}, benchmark, rng)
    # create a_{js} matrix from ns solutions
    A = create_job_sched_matrix(ns, benchmark['n'], benchmark['m'])
    # return np.unique(A, axis=1), ns       # remove duplicate schedules
    return A, ns


"""
 this function receive 10 solution and have to optimize it with 2 simple operation
 around it's neighborhood (insert or swap).
 
 d: dictionary of the best solution founded with the heuristic
 benchmark: dictionary containing the instance values 
 rng: random generator object 
 OUTPUT: an optimize version of the solution
"""


def neighborhood_search(d, benchmark, rng):
    max_pivot = benchmark['n'] // 2 + 1         # number of pivot - same as np.floor(benchmark['n'] / 2) + 1
    # max_pivot = benchmark['n']                # does make sense to use every job as pivot?
    sol_space = dict()

    # for a max_iter number of pivot selection select a random job as pivot
    # and try all the possible move for that pivot and check the new cost at every move
    for key, solution in d.items():
        cur_cost = sum([sched_cost(sched, benchmark['p'], benchmark['w']) for sched in solution])
        for pivot in rng.choice(range(benchmark['n']), max_pivot, replace=False):
            # search the schedule in which there is the pivot
            mask = np.array([np.isin(pivot, sched).item() for sched in solution])
            pivot_sched_id = np.where(mask)[0].item()

            for new_sched_id in np.where(~mask)[0]:
                # INSERT for each neighborhood schedule
                new_sol = solution.copy()
                new_sol[new_sched_id] = smith_order(np.append(solution[new_sched_id], pivot),
                                                    benchmark['p'], benchmark['w'])
                new_sol[pivot_sched_id] = smith_order(np.setdiff1d(solution[pivot_sched_id], pivot),
                                                      benchmark['p'], benchmark['w'])
                new_cost = sum([sched_cost(sched, benchmark['p'], benchmark['w']) for sched in new_sol])
                if new_cost < cur_cost:
                    # add the new solution to the sol_space dictionary of dictionary
                    sol_space_key = 0 if not sol_space else max(sol_space) + 1
                    sol_space.update({sol_space_key: {new_cost: new_sol.copy()}})

                for swap_job in solution[new_sched_id]:
                    # SWAP for each neighborhood schedule jobs
                    new_sol = solution.copy()
                    new_sol[new_sched_id] = smith_order(np.append(np.setdiff1d(solution[new_sched_id], swap_job),
                                                                  pivot), benchmark['p'], benchmark['w'])
                    new_sol[pivot_sched_id] = smith_order(np.append(np.setdiff1d(solution[pivot_sched_id], pivot),
                                                                    swap_job), benchmark['p'], benchmark['w'])
                    new_cost = sum([sched_cost(sched, benchmark['p'], benchmark['w']) for sched in new_sol])
                    if new_cost < cur_cost:
                        # add the new solution to the sol_space dictionary of dictionary
                        sol_space_key = 0 if not sol_space else max(sol_space) + 1
                        sol_space.update({sol_space_key: {new_cost: new_sol.copy()}})
            # ENDFOR; end available move, change pivot

            # check the best solution found and update the old one
            # the below block of code should be positioned after the end of the moves for a pivot because
            # in that way every new move would affect already the best solution found before
            if sol_space:
                # print('** Found a better solution with NS!! **')
                best_sol_key = np.argmin(np.array([list(v) for k, v in sol_space.items()]).flatten())
                d[key] = list(sol_space[best_sol_key].values())[0]
                cur_cost = list(sol_space[best_sol_key].keys())[0]
                sol_space = dict()

        # ENDFOR; end random pivot list, change solution
    # ENDFOR; end solutions available

    return d
