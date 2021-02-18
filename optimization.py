from time import process_time
from benchmark import *
from heuristic import *
import gurobipy as gp

"""
 Our pricing algorithm is a dynamic programming algorithm that usually
 generates more than one column with negative reduced cost.
 pj: vector sum of the completion time for each job as they're ordered by index
 hmin = minimum completion time on m machine (LB)
 hmax = maximum completion time on m machine (UB)
 F_j(t): denote the minimum reduced cost for all machine schedules that consist
         of jobs from the set {J_1, ..., J_j} and complete their last job at time t
 
 benchmark: instance to work on
 lambda_0: dual variable of the first constraint
 lambda_j: dual j variable of the second constraint
 n_col2add: # of schedules to add at the problem
 OUTPUT: 3 new schedules with negative reduced cost 
"""


def pricing_algorithm(benchmark, lambda_j, n_col2add):
    pj = np.cumsum(benchmark['p'])
    hmin, hmax = processing_bound(benchmark)

    # INIT
    F = np.full((len(lambda_j), int(np.ceil(hmax)) + 1), np.inf)
    F[0, 0] = -lambda_j[0]

    # RECURSION PASS
    for job, fj in enumerate(F):
        # for each job (skip the base job 0)
        if job != 0:
            # set the actual job because dual variable lambda are n+1
            real_job = job - 1

            for t, fjt in enumerate(fj):
                # for each t = 0, ..., min{P(j), Hmax}
                if t > min(pj[real_job], int(np.ceil(hmax))):
                    # leave the current values (probably inf) and skip to a new job
                    break

                if benchmark['p'][real_job] <= t:
                    op2nd = F[job - 1, t - benchmark['p'][real_job]] + benchmark['w'][real_job] * t - lambda_j[job]
                else:
                    op2nd = F[job - 1, t]
                F[job, t] = min(F[job - 1, t], op2nd)

    # compute F^*
    if min(F[-1, int(np.ceil(hmin)):]) >= -1e-9:
        # optimal solution
        return np.empty(0)
    else:
        # -- backtracing -- return n_col2add new schedules
        # An empirically good choice appeared to be adding those three columns that
        # correspond to those three values of t for which F_n(t) is most negative
        cols2add = np.argsort(F[-1, int(np.ceil(hmin)):])[:n_col2add] + int(np.ceil(hmin))
        tmp = np.zeros((benchmark['n'], n_col2add), dtype=np.uint16)
        for idx, t_init in enumerate(cols2add):
            new_sched = list()
            t = t_init
            for j in range(benchmark['n'], 0, -1):
                # cycle from the last row of F through the beginning
                if t == 0:
                    break
                elif F[j, t] != F[j-1, t]:
                    new_sched.append(j-1)
                    t -= benchmark['p'][j-1]

            np.put(tmp[:, idx], new_sched, 1)

        return tmp.copy()


"""
 based on the set-covering formulation, it relaxed that initial formulation and
 resolve a RLP with the dual variables. After that, it iteratively add new column
 to find a new better solution if possible
 
 param: config dictionary from toml file
 b: string that specify from where import the benchmark
 verbose: boolean variable that imply more verbosity (then more printing on console)
 
 ---- REMARKS ----
 In the implementation of the column generation it is necessary to have two different models: one for the master
 and one for the subproblem. To solve the continuous relaxation in the master build a model with integer/binary
 variables and then relax it using the Gurobi method model.relax() to obtain a new (third) relaxed model.
 To add a new pattern to a master problem consider to read the current columns with mycol=Column() and
 adding a new variable with master.addVar() and new constraint elements for the new pattern using mycol.addTerms().
 -----------------
 
 ---- DOC LINKS ----
 [https://www.gurobi.com/documentation/9.1/refman/py_mvar.html]
 [https://www.gurobi.com/documentation/9.1/refman/py_model_addmvar.html#pythonmethod:Model.addMVar]
 [https://www.gurobi.com/wp-content/uploads/2020/04/gurobipy_matrixfriendly_webinar-slides.pdf]
 [https://www.gurobi.com/documentation/9.1/refman/pi.html#attr:Pi]
 [https://www.gurobi.com/documentation/9.1/refman/py_column.html]
"""


def column_generation(param, b, verbose):
    # ** VERBOSE SETTINGS **
    env = gp.Env(empty=True)
    if verbose:
        def verbose_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verbose_print = lambda *a, **k: None  # do-nothing function
        env.setParam('OutputFlag', 0)         # set model to un-verbose

    env.start()             # start env to update above setting
    start = process_time()  # start counting time

    # ** GENERATE BENCHMARK **
    # check b parameter and extract the benchmark dictionary
    if b == 'rnd':
        verbose_print('Generating random instance')
        benchmark = gen_instance(param['PARAMETERS']['m'], param['PARAMETERS']['n'])
    elif b == 'beb':
        verbose_print('Choosing random file from Barnes and Brennan')
        benchmark = import_from_file(param['PATH']['data_path'])

    # ** HEURISTIC INITIALIZATION **
    # get the schedules initialization from the heuristic function
    htime = process_time()
    verbose_print('Starting Randomized List Heuristic')
    A, heuristic_sol = randomized_list_heuristic(benchmark,
                                                 param['PARAMETERS']['MIN'],
                                                 param['PARAMETERS']['MAX'])
    htime = int(process_time() - htime)  # heuristic execution time

    J = benchmark['n']  # number of jobs
    S = A.shape[1]      # number of schedules

    # ** GENERATE a single set-covering MODEL and then add column at each iteration into the while loop **
    set_cover = gp.Model('Parallel identical processor scheduling with weighted completion times '
                         'as set-covering formulation', env=env)
    set_cover.reset()

    """ --- PARAMETERS & VARIABLE ---
     a_{js}: if job j is into schedule s
     C_j(s): completion time of job j into schedule s (used on the fly)
        c_s: schedule s cost, based on weight and processing time of each job into the schedule
        x_s: selected schedules
        
     OBSERVATION: I removed the use of Mvar because there is still no possibility
                  to append/concatenate elements to MVar of gurobi and retrieve it later
    """

    x = set_cover.addVars(S, vtype=gp.GRB.BINARY, name='x_s')
    # x = set_cover.addMVar(S, vtype=gp.GRB.BINARY, name='x_s')              # ** with MVar **
    # compute schedules cost - sc
    sc = np.array([sched_cost(np.where(A[:, col])[0], benchmark['p'], benchmark['w'], True) for col in range(S)])

    # OBJECTIVE FUNCTION
    set_cover.setObjective(gp.quicksum(sc[s] * x[s] for s in range(S)), gp.GRB.MINIMIZE)
    # set_cover.setObjective(sc @ x, gp.GRB.MINIMIZE)                               # ** with MVar **

    # CONSTRAINTS
    set_cover.addConstr(gp.quicksum(x[s] for s in range(S)) == benchmark['m'])                  # (1)
    set_cover.addConstrs(gp.quicksum(A[j, s] * x[s] for s in range(S)) == 1 for j in range(J))  # (2)
    # set_cover.addConstr(sum(x) == benchmark['m'])               # (1)             # ** with MVar **
    # set_cover.addConstrs(A[j, :] @ x == 1 for j in range(J))    # (2)             # ** with MVar **

    set_cover.write(param['PATH']['model_path'] + 'set_cover.lp')

    n_iter = 1      # number of total iteration
    verbose_print('Starting column generation')

    while True:
        master = set_cover.relax()              # solve the continuous relaxation of current master

        # write & solve master - RLP
        master.write(param['PATH']['model_path'] + 'master_model.rlp')
        master.optimize()

        # compute the dual variables lambda_j
        lambda_j = np.array([const.Pi for const in master.getConstrs()])

        # ** PRICING ALGORITHM **
        # call the pricing subproblem in case there are negatives rc
        new_schedules = pricing_algorithm(benchmark, lambda_j, param['PARAMETERS']['nnc'])
        if not len(new_schedules):
            # exit because there isn't new schedule to add
            verbose_print('No new schedule to add, already into the optimal solution')
            break

        # update A with the new schedules
        # check column existence is counterproductive because the algorithm don't go on the next column in this way
        A = np.concatenate((A, new_schedules), axis=1)

        for i in range(A.shape[1] - S):
            # for each new column
            new_sched = A[:, S + i].copy()
            # compute schedule cost and append it to sc variable
            new_cost = sched_cost(np.where(new_sched)[0], benchmark['p'], benchmark['w'], True)
            sc = np.append(sc, new_cost)
            # creates a new Column with the corresponding coefficients and constraints
            new_col = gp.Column(1., set_cover.getConstrs()[0])  # the first constraint is constant for all
            new_col.addTerms(list(new_sched[np.where(new_sched)[0]]),
                             list(np.array(set_cover.getConstrs()[1:])[np.where(new_sched)[0]]))
            # add a variable for the new column of the set-cover
            x[S + i] = set_cover.addVar(obj=new_cost, vtype=gp.GRB.BINARY, name=f'x_s[{S + i}]', column=new_col)

        S = A.shape[1]      # re-assign the number of schedule
        set_cover.update()  # update the set-cover model

        n_iter += 1
        # print partial time
        part_time = int(process_time() - start)
        verbose_print(f'Iteration: {n_iter}\n'
                      f'Partial time: {int(part_time / 60)} min {part_time % 60} s')
    # ENDWHILE; the column generation has found the best continuous solution

    btime = 0   # branch and bound processing time
    # check both the number of schedules selected and that the values xs of the selected schedules be >= EPS
    sel_sched_idx = np.where(master.x)[0]
    if len(sel_sched_idx) == benchmark['m'] and len(np.where(np.array(master.x) > 1-1e-9)[0]) == len(sel_sched_idx):
        # master solution is already int, print relaxed solution
        verbose_print(print_solution(benchmark, master, A))
    else:
        # ** Theorem 1 **
        # for each job check that C_j(s) is the same into every schedule of S*
        sel_sched = [smith_order(np.where(A[:, s])[0], benchmark['p'], benchmark['w']) for s in sel_sched_idx]
        cj_sched = [np.cumsum(benchmark['p'][s]) for s in sel_sched]
        res_list = list()
        for j in range(benchmark['n']):
            cjs = list()
            for ids, s in enumerate(sel_sched):
                if j in s:
                    cjs.append(cj_sched[ids][np.where(s == j)[0].item()])

            # check if all element in cjs are equals
            if len(set(cjs)) == 1:
                res_list.append(True)
            else:
                # as the first job that we found we can break and declare Theorem1 not respected
                break
        # ENDFOR;

        if len(res_list) == benchmark['n']:
            # Theorem 1 is True
            verbose_print('Theorem 1 validated')
        else:
            # ** B&B on the solution **
            btime = process_time()
            verbose_print('Starting Branch & Bound')
            branch_and_bound(master.objVal, A[:, sel_sched_idx])
            # verbose_print(print_solution())
            btime = int(process_time() - btime)  # B&B execution time

    set_cover.optimize()
    verbose_print(print_solution(benchmark, set_cover, A))

    # print timing
    tot_time = int(process_time() - start)
    verbose_print(print_timing(n_iter, htime, btime, tot_time))

    # write results into directory out
    # write2file(benchmark, set_cover, A, n_iter, htime, btime, tot_time, b, param['PATH']['res_path'])


"""
 2 types of branching:
    - execution intervals
    - immediate successor
    
 variable:
 OUTPUT: integer solution
"""


def branch_and_bound(lower_bound, A, benchmark):
    # take the fractional job of minimum index and branch on it
    sel_sched = [smith_order(np.where(col)[0], benchmark['p'], benchmark['w']) for col in A.T]
    cj_sched = [np.cumsum(benchmark['p'][s]) for s in sel_sched]

    min_fractional_job, cjmin = 0, 0

    for j in range(benchmark['n']):
        cjs = list()
        for ids, s in enumerate(sel_sched):
            if j in s:
                cjs.append(cj_sched[ids][np.where(s == j)[0].item()])

        # check if all element in cjs are equals
        if not len(set(cjs)) == 1:
            # branch on j
            min_fractional_job = j
            cjmin = min(cjs)
            break
    # ENDFOR;

    # set rj and dj of min_fractional_job
    # left branch dj = cjmin
    # right branch rj = cjmin + 1 - p_j
    rj, dj = cjmin + 1 - benchmark['p'][min_fractional_job], cjmin

    # compute all new rj and dj of predecessor and successor of min_fractional_job
    # rj = np.array()
    # dj = np.array()

    # call the column generation on this branch subproblem
    #   - run on master with the new time constraints