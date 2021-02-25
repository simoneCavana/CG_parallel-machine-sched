from benchmark import *
from heuristic import *
from optimization import *

"""
 create the first node and the set_cover model
"""


def declare_model(benchmark, A, env, param):
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
    set_cover.addConstr(gp.quicksum(x[s] for s in range(S)) == benchmark['m'])  # (1)
    set_cover.addConstrs(gp.quicksum(A[j, s] * x[s] for s in range(S)) == 1 for j in range(J))  # (2)
    # set_cover.addConstr(sum(x) == benchmark['m'])               # (1)             # ** with MVar **
    # set_cover.addConstrs(A[j, :] @ x == 1 for j in range(J))    # (2)             # ** with MVar **

    set_cover.write(param['PATH']['model_path'] + 'set_cover.lp')
    # constraint = {key: val for key, val in zip(np.arange(benchmark['n'] + 1), set_cover.getConstrs())}

    # initialize starting time and completion time for each job
    rj = np.zeros(benchmark['n'])
    dj = np.full(benchmark['n'], int(np.ceil(processing_bound(benchmark)[1])))

    # create base_node
    return {'key': 0, 'value': [set_cover, x, A, sc, rj, dj], 'job': None,
            'parent': None, 'children': [], 'visited': False}


"""
 return 1 if there is to branch or 0 otherwise (the solution is integral or respect Theorem1)
"""


def checkBEB(benchmark, node, verbose_print):
    model, x, A, sc, rj, dj = node['value']

    relaxed = model.relax()
    relaxed.optimize()

    sel_sched_idx = np.where(relaxed.x)[0]
    # check both the number of schedules selected and
    # that the values xs of the selected schedules be >= EPS
    if len(sel_sched_idx) == benchmark['m'] and \
            len(np.where(np.array(relaxed.x) > 1 - 1e-9)[0]) == len(sel_sched_idx):
        # master solution is already int, print relaxed solution
        return 0
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
            # Theorem 1 is True, we can exit
            verbose_print('Theorem 1 validated')
            return 0
        else:
            # we have to BEB
            return 1


"""
 given a node, compute it's minimum fractional job and create two branches on it

 benchmark: instance that we are working on
 node: node of the tree to work on
 OUTPUT: left and right values for the node's children
 
 [https://www.gurobi.com/documentation/9.1/refman/py_model_copy.html]
 [https://www.gurobi.com/documentation/9.1/refman/py_model_remove.html]
"""


def branch(benchmark, node):
    model, x, A, sc, rj, dj = node['value']
    old_beb_job = node['job']

    relaxed = model.relax()
    relaxed.optimize()

    # compute schedules into smith order and it's cjs
    schedules = np.array([smith_order(np.where(col)[0], benchmark['p'], benchmark['w']) for col in A.T], dtype=object)
    cj_schedules = np.array([np.cumsum(benchmark['p'][s]) for s in schedules], dtype=object)

    # schedules and completion time of jobs into solution
    sol_sched_idx = np.where(relaxed.x)[0]
    sol_sched = schedules[sol_sched_idx].copy()
    cj_sol_sched = cj_schedules[sol_sched_idx].copy()

    min_frac_job, cjmin = -1, 0

    # take the fractional job of minimum index and branch on it
    # for j in np.setdiff1d(np.arange(benchmark['n']), old_beb_job):
    for j in range(benchmark['n']):
        cjs = list()
        for ids, s in enumerate(sol_sched):
            if j in s:
                cjs.append(cj_sol_sched[ids][np.where(s == j)[0].item()])

        # check if all element in cjs are equals
        if not len(set(cjs)) == 1:
            # branch on j
            min_frac_job = j
            cjmin = min(cjs)
            break
    # ENDFOR;

    # compute predecessor and successor of min_fractional_job
    sm_order = smith_order(np.arange(benchmark['n']), benchmark['p'], benchmark['w'])
    j_pos = np.where(sm_order == min_frac_job)[0].item()
    pred = sm_order[:j_pos].copy()
    succ = sm_order[j_pos + 1:].copy()

    right_rj = rj.copy()
    left_dj = dj.copy()

    # set rj and dj of min_fractional_job
    # dj = cjmin            -- left branch
    # rj = cjmin + 1 - p_j  -- right branch
    right_rj[min_frac_job], left_dj[min_frac_job] = cjmin + 1 - benchmark['p'][min_frac_job], cjmin

    # compute all new rj and dj of predecessor and successor of min_fractional_job
    # rk = max{rk, rj}                 for the successor of min_frac_job
    # dk = min{dk, dj - pj + pk}       for the predecessor of min_frac_job
    right_rj[succ] = np.array([max(rj[k], right_rj[min_frac_job]) for k in succ])
    left_dj[pred] = np.array([min(dj[k], left_dj[min_frac_job] - benchmark['p'][min_frac_job] + benchmark['p'][k])
                              for k in pred])

    """
     keep only the schedules that respect the new time constraints and, for the schedules
     that are selected as current solution of the model, set the cost to bigM
     for all jobs Cj - dj <= 0         PAPER INSTRUCTION

     1. identify schedules infeasible
     2. for each schedules in (1), check if it is into the solution
        - pump cost to big M
        - prune otherwise
    """
    left_inf_idx, right_inf_idx = list(), list()
    for i, sched in enumerate(cj_schedules):
        if max(sched - left_dj[schedules[i]]) > 0:
            # sched INFEASIBLE for LEFT child
            left_inf_idx.append(i)

        if min((sched - benchmark['p'][schedules[i]]) - right_rj[schedules[i]]) < 0:
            # sched INFEASIBLE for RIGHT child
            right_inf_idx.append(i)

    left2prune = np.setdiff1d(left_inf_idx, sol_sched_idx)
    right2prune = np.setdiff1d(right_inf_idx, sol_sched_idx)

    left2pump = np.setdiff1d(left_inf_idx, left2prune)
    right2pump = np.setdiff1d(right_inf_idx, right2prune)

    model.update()
    left_model = model.copy()
    right_model = model.copy()

    # use the serial schedule UB cost as bigM to discard schedule
    # otherwise set to worst serial cost with 50 jobs: 12750000
    bigm = 12750000     # serial_cost_ub(benchmark)

    left_A, right_A = A.copy(), A.copy()
    left_sc, right_sc = sc.copy(), sc.copy()

    for i in left2pump:
        left_model.getVars()[i].obj = bigm
    for i in right2pump:
        right_model.getVars()[i].obj = bigm

    [left_model.remove(left_model.getVars()[i]) for i in left2prune]
    [right_model.remove(right_model.getVars()[i]) for i in right2prune]

    left_model.reset()
    left_model.update()
    right_model.reset()
    right_model.update()

    left_x = dict(zip(np.arange(len(model.getVars())), left_model.getVars()))
    right_x = dict(zip(np.arange(len(model.getVars())), right_model.getVars()))

    if len(left2pump):
        np.put(left_sc, left2pump, bigm)
    if len(right2pump):
        np.put(right_sc, right2pump, bigm)

    if len(left2prune):
        left_A = np.delete(left_A, left2prune, 1)
        left_sc = np.delete(left_sc, left2prune)
    if len(right2prune):
        right_A = np.delete(right_A, right2prune, 1)
        right_sc = np.delete(right_sc, right2prune)

    # return a couple of ['value'] to assign at the two sub-branch
    return [left_model, left_x, left_A, left_sc, rj, left_dj],\
           [right_model, right_x, right_A, right_sc, right_rj, dj],\
           min_frac_job


"""
 initialize everything and manage the tree for the branch and bound if it is needed
"""


def node_manager(param, b, verbose):
    # ** VERBOSE SETTINGS **
    env = gp.Env(empty=True)
    if verbose:
        def verbose_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verbose_print = lambda *a, **k: None  # do-nothing function
        env.setParam('OutputFlag', 0)         # set model to un-verbose

    env.start()  # start env to update above setting
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
    S_bar, heuristic_sol = randomized_list_heuristic(benchmark,
                                                     param['PARAMETERS']['MIN'],
                                                     param['PARAMETERS']['MAX'])
    htime = int(process_time() - htime)  # heuristic execution time

    # create the set_cover model and return it's parameters
    base_node = declare_model(benchmark, S_bar, env, param)

    # create the stack of dict to implement the BEB tree as depth first
    stack = list()
    stack.append(base_node)

    lower_bound = np.inf
    upper_bound = serial_cost_ub(benchmark)
    node_id, nbeb = 0, 0

    while True:
        if not stack[node_id]['visited']:
            # call column generation on the node if it's not already visited
            column_generation(benchmark, param, stack[node_id], verbose_print)

            if node_id == 0:
                # if it's the first time save the solution as lower bound
                relaxed = stack[node_id]['value'][0].relax()
                relaxed.optimize()
                lower_bound = relaxed.objVal

            # check if there is to apply branch and bound or we can exit
            if checkBEB(benchmark, stack[node_id], verbose_print):
                # create the two children nodes and re-iterate with col_gen on the first one feasible
                verbose_print('Starting Branch & Bound')
                nbeb += 1

                stack[node_id]['visited'] = True
                left_child, right_child = len(stack), len(stack) + 1
                stack[node_id]['children'].append(left_child)
                stack[node_id]['children'].append(right_child)

                left_value, right_value, beb_job = branch(benchmark, stack[node_id])

                stack.append({'key': left_child, 'value': left_value, 'job': beb_job,
                              'parent': node_id, 'children': [], 'visited': False})
                stack.append({'key': right_child, 'value': right_value, 'job': beb_job,
                              'parent': node_id, 'children': [], 'visited': False})

                # check children feasibility
                for child in stack[node_id]['children']:
                    # for all jobs dj - rj >= pj        MINE IDEA
                    if not (benchmark['p'] <= stack[child]['value'][5] - stack[child]['value'][4]).all():
                        stack[child]['visited'] = True

                # branched on the same job as before, prune children
                # this condition is needed because some solution continue to select
                # the old schedule with bigM cost
                if beb_job == stack[node_id]['job']:
                    stack[left_child]['visited'] = True
                    stack[right_child]['visited'] = True

                # going depth first if possible
                if not stack[stack[node_id]['children'][0]]['visited']:
                    node_id = stack[node_id]['children'][0]
                elif not stack[stack[node_id]['children'][1]]['visited']:
                    node_id = stack[node_id]['children'][1]
                else:
                    # none of the children is feasible, return to parent
                    node_id = stack[node_id]['parent']
            else:
                # optimal integral solution
                stack[node_id]['visited'] = True
                stack[node_id]['value'][0].optimize()

                new_cost = sum([stack[node_id]['value'][1][idx].obj
                                for idx in np.where(stack[node_id]['value'][0].x)[0]])
                # new_cost = sum(stack[node_id]['value'][3][np.where(stack[node_id]['value'][0].x)[0]])
                # check if discovered solution is better than the UB
                if new_cost < upper_bound:
                    break
        else:
            # it came here if we iterate over an already visited node,
            # an example case could be when backtrack to parent to go on the other branch
            if stack[node_id]['children']:
                if not stack[stack[node_id]['children'][0]]['visited']:
                    node_id = stack[node_id]['children'][0]
                elif not stack[stack[node_id]['children'][1]]['visited']:
                    node_id = stack[node_id]['children'][1]
                else:
                    # we can prune this branches
                    del stack[stack[node_id]['children'][1]]
                    del stack[stack[node_id]['children'][0]]
                    stack[node_id]['children'] = []
            elif stack[node_id]['parent'] is not None:
                # return to parent if exist
                node_id = stack[node_id]['parent']
            else:
                # visited all nodes without find an integral solution - bad tree implementation
                verbose_print('All nodes were computed, can\'t force integrality with Branch and Bound')
                return

    # verbose_print(print_solution(benchmark, master, A))
    # stack[node_id]['value'][0].optimize()
    verbose_print(print_solution(benchmark, stack[node_id]['value'][0], stack[node_id]['value'][2], lower_bound, nbeb))

    tot_time = int(process_time() - start)
    verbose_print(print_timing(htime, tot_time))

    # write results into directory out
    write2file(benchmark, stack[node_id]['value'][0], stack[node_id]['value'][2], lower_bound, nbeb,
               htime, tot_time, b, param['PATH']['res_path'])