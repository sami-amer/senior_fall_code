# MIT 6.034 Bayesian Inference Lab
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    out = set()
    q = [var]

    while q:
        curr = q.pop()
        parents = net.get_parents(curr)
        out = out.union(parents)
        for parent in parents:
            q.append(parent)
        
    return out

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    out = set()
    q = [var]

    while q:
        curr = q.pop()
        children = net.get_children(curr)
        out = out.union(children)
        for child in children:
            q.append(child)
        
    return out

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    d = get_descendants(net,var)
    d.add(var)
    return set(net.get_variables()).difference(d)


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    if net.get_parents(var).issubset(givens.keys()) and get_descendants(net,var).intersection(givens.keys()) == set(): # all parents and no descendents in givens
        to_remove = get_nondescendants(net,var).difference(net.get_parents(var)) # non descendants, excluding parents
        return {k:v for k,v in givens.items() if k not in to_remove}
    else:
        return givens 

def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    for var in hypothesis.keys():
        simplified = simplify_givens(net,var,givens) if givens else givens
        if not givens: break
    try:
        return net.get_probability(hypothesis,parents_vals=simplified) if simplified else net.get_probability(hypothesis,givens)
    except ValueError:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    # we can make this a sum of probability(x|given:parents)
    prob = 1
    
    givens = {}
    for var in net.topological_sort():
        prob *= probability_lookup(net,{var:hypothesis[var]},{p:givens[p] for p in net.get_parents(var)})
        givens[var] = hypothesis[var]
    
    return prob
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    combos = net.combinations(net.get_variables(),hypothesis)
    prob = 0
    for comb in combos:
        tmp = comb
        for k,v in hypothesis.items():
            tmp[k]=v
        prob += probability_joint(net,tmp)
    
    return prob

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if not givens:
        return probability_marginal(net,hypothesis)
    else:
        if dict(hypothesis,**givens) != dict(givens,**hypothesis):
            return 0

        return probability_marginal(net,dict(hypothesis,**givens))/probability_marginal(net,givens)

def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net,hypothesis,givens)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    
    params = 0
    for var in net.get_variables():
        if net.get_parents(var):
            tmp = [len(net.get_domain(x)) for x in net.get_parents(var)]
            tmp.append(len(net.get_domain(var))-1)
            params += product(tmp)
        else:
            params += len(net.get_domain(var))-1
    
    return params



#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    givens = givens if givens else {}
    combs = net.combinations([var1,var2])
    for comb in combs:
        joint = probability(net,comb,givens)
        p1 = probability(net,{var1:comb[var1]},givens)
        p2 = probability(net,{var2:comb[var2]},givens)
        # if P(AB) == P(A)P(B), A and B are independent
        if approx_equal(joint, p1*p2):
            return True
    return False
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
