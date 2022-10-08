# MIT 6.034 Constraints Lab
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    for var,domain in csp.domains.items():
        if len(domain) == 0:
            return True
    return False

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for constraint in csp.get_all_constraints():
        if csp.get_assignment(constraint.var1) is not None and csp.get_assignment(constraint.var2) is not None:
            if not constraint.check(csp.get_assignment(constraint.var1),csp.get_assignment(constraint.var2)):
                return False
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """

    q = [problem]
    ext = 0
    while q:
        curr_p = q.pop()
        ext += 1

        if has_empty_domains(curr_p) or not check_all_constraints(curr_p):
            continue

        else:

            if len(curr_p.unassigned_vars) == 0:
                return (curr_p.assignments,ext)
            
            var = curr_p.pop_next_unassigned_var()
            for val in reversed(curr_p.get_domain(var)):
                new_prob = curr_p.copy()
                new_prob.set_assignment(var,val)
                q.append(new_prob)
    
    return (None, ext)

            


# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = solve_constraint_dfs(get_pokemon_problem())[1]


#### Part 3: Forward Checking ##################################################

def reduce_domain(csp,var1,var2):
    allowed = []
    for val_n in csp.get_domain(var2):

        r = False
        for val in csp.get_domain(var1):

            valid = True
            for con in csp.constraints_between(var1,var2):
                if not con.check(val,val_n):
                    valid = False
            if valid:
                r = True 
        
        if r:
            allowed.append(val_n)
    if len(allowed) == len(csp.get_domain(var2)):
        return False
    else:
        csp.set_domain(var2,allowed) 
        return True



def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    # loop through neighbors, then for each neighbor
    #   loop through the neighbor domain values, and if a value in the neighbor violates the constraint for EVERY value in the original var, remove that value from the neighbor
    #   if any domain at any point is empty, return None
    #   modify csp, return sorted list of neighbors with shortened domain

    neighbors = csp.get_neighbors(var)
    reductions = [] 
    for neighbor in neighbors:
        if reduce_domain(csp,var,neighbor):
            reductions.append(neighbor)
        if has_empty_domains(csp):
            return None
    return reductions


# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    q = [problem]
    ext = 0
    while q:
        curr_p = q.pop()
        ext += 1

        if has_empty_domains(curr_p) or not check_all_constraints(curr_p):
            continue

        else:

            if len(curr_p.unassigned_vars) == 0:
                return (curr_p.assignments,ext)
            
            var = curr_p.pop_next_unassigned_var()
            for val in reversed(curr_p.get_domain(var)):
                new_prob = curr_p.copy()
                new_prob.set_assignment(var,val)
                forward_check(new_prob,var)
                q.append(new_prob)
    
    return (None, ext)



# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = solve_constraint_forward_checking(get_pokemon_problem())[1]


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if queue == None:
        queue = csp.get_all_variables()
    dq = [] 
    while queue:
        var = queue.pop(0)
        dq.append(var)
        reductions = forward_check(csp,var)
        if not reductions:
            if has_empty_domains(csp):
                return None
        for r in reductions:
            if r not in queue:
                queue.append(r)

    return dq

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?
problem = get_pokemon_problem()
domain_reduction(problem)
ANSWER_3 = 6 


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    q = [problem]
    ext = 0
    while q:
        curr_p = q.pop()
        ext += 1

        if has_empty_domains(curr_p) or not check_all_constraints(curr_p):
            continue

        else:

            if len(curr_p.unassigned_vars) == 0:
                return (curr_p.assignments,ext)
            
            var = curr_p.pop_next_unassigned_var()
            for val in reversed(curr_p.get_domain(var)):
                new_prob = curr_p.copy()
                new_prob.set_assignment(var,val)
                domain_reduction(new_prob,[var])
                q.append(new_prob)
    
    return (None, ext)



# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?
problem = get_pokemon_problem()
ANSWER_4 = solve_constraint_propagate_reduced_domains(problem)[1]


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue == None:
        queue = csp.get_all_variables()
    dq = [] 
    while queue:
        var = queue.pop(0)
        dq.append(var)
        reductions = forward_check(csp,var)
        if not reductions:
            if has_empty_domains(csp):
                return None
        for r in reductions:
            if enqueue_condition_fn(csp,r) and r not in queue:
                queue.append(r)

    return dq

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    if len(csp.get_domain(var)) == 1:
        return True
    else:
        return False

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False

#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    q = [problem]
    ext = 0
    while q:
        curr_p = q.pop()
        ext += 1

        if has_empty_domains(curr_p) or not check_all_constraints(curr_p):
            continue

        else:

            if len(curr_p.unassigned_vars) == 0:
                return (curr_p.assignments,ext)
            
            var = curr_p.pop_next_unassigned_var()
            for val in reversed(curr_p.get_domain(var)):
                new_prob = curr_p.copy()
                new_prob.set_assignment(var,val)
                if enqueue_condition:
                    propagate(enqueue_condition,new_prob,[var])
                q.append(new_prob)
    
    return (None, ext)


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = solve_constraint_generic(get_pokemon_problem(),condition_singleton)[1]


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) == 1:
        return True
    else:
        return False 

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) > 1 or abs(m-n)==0:
        return True
    else:
        return False

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    cons = []
    for i in range(len(variables)):
        for j in range(i):
            cons.append(Constraint(variables[i],variables[j],constraint_different))

    return cons


#### SURVEY ####################################################################

NAME = "Sami" 
COLLABORATORS = "None" 
HOW_MANY_HOURS_THIS_LAB_TOOK = "4" 
WHAT_I_FOUND_INTERESTING = "None" 
WHAT_I_FOUND_BORING = "None" 
SUGGESTIONS = "Testcases are not comprehensive and it feels less like programming and more like playing whack-a-mole. If there are tests per componenet, then a component should not pass its tests unless it is actually correct." 
