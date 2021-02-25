from time import process_time
from utilities import *
import gurobipy as gp
import re

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


def pricing_algorithm(benchmark, rj, dj, lambda_j, n_col2add):
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

                if rj[real_job] + benchmark['p'][real_job] <= t <= dj[real_job]:
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
        tmp = np.zeros((benchmark['n'], n_col2add), dtype=np.uint8)
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


def column_generation(benchmark, param, node, verbose_print):
    model, x, A, sc, rj, dj = node['value']
    S = A.shape[1]

    n_iter = 1  # number of total iteration
    verbose_print('Starting column generation')
    cgtime = process_time()  # start counting time

    while True:
        master = model.relax()  # solve the continuous relaxation of current master

        # write & solve master - RLP
        # master.write(param['PATH']['model_path'] + 'master_model.rlp')
        master.optimize()

        # compute the dual variables lambda_j
        lambda_j = np.array([const.Pi for const in master.getConstrs()])

        # compute reduced cost
        # rc = np.array([sc[s] - A[:, s] @ lambda_j for s in range(S)])
        # if rc < 1e-9:
        #     break

        # ** PRICING ALGORITHM **
        # call the pricing subproblem in case there are negatives rc
        new_schedules = pricing_algorithm(benchmark, rj, dj, lambda_j, param['PARAMETERS']['nnc'])
        if not len(new_schedules):
            # exit because there isn't new schedule to add
            verbose_print('No new schedule to add, already into the optimal solution')
            node['value'][2], node['value'][3] = A.copy(), sc.copy()
            return

        # update A with the new schedules
        # check column existence is counterproductive because the algorithm don't go on the next column in this way
        A = np.concatenate((A, new_schedules), axis=1)

        # simple check on the correspondence of the name with the index of the dictionary
        # in order to avoid that in case of branch and bound the variables take the same name
        last_var_name = x[len(x) - 1].getAttr(gp.GRB.Attr.VarName)
        var_idx = int(re.search(r'(\d+)', last_var_name).group(0))

        for i in range(A.shape[1] - S):
            # for each new column
            new_sched = A[:, S + i].copy()
            # compute schedule cost and append it to sc variable
            new_cost = sched_cost(np.where(new_sched)[0], benchmark['p'], benchmark['w'], True)
            sc = np.append(sc, new_cost)
            # creates a new Column with the corresponding coefficients and constraints
            new_col = gp.Column(list(np.ones(len(np.where(new_sched)[0]) + 1)),
                                list(np.array(model.getConstrs())[np.where(np.insert(new_sched, 0, 1))[0]]))
            # add a variable for the new column of the set-cover
            x[S + i] = model.addVar(obj=new_cost, vtype=gp.GRB.BINARY, name=f'x_s[{var_idx+i+1}]', column=new_col)

        S = A.shape[1]  # re-assign the number of schedule
        # model.reset()
        model.update()  # update the set-cover model

        n_iter += 1
        # print partial time
        part_time = int(process_time() - cgtime)
        verbose_print(f'Iteration: {n_iter}\n'
                      f'Partial time: {int(part_time / 60)} min {part_time % 60} s')
    # ENDWHILE; the column generation has found the best continuous solution