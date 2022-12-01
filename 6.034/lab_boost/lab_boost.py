# MIT 6.034 Boosting (Adaboost) Lab
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    weight = Fraction(1,len(training_points))
    return {x:weight for x in training_points}

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    out = {}
    for classifier in classifier_to_misclassified:
        missed = classifier_to_misclassified[classifier]
        total_weight = sum([point_to_weight[x] for x in missed])
        out[classifier] = total_weight
    
    return out

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    if use_smallest_error:
        best = min(classifier_to_error_rate, key = lambda x: classifier_to_error_rate[x]) 
        min_list = [x for x in classifier_to_error_rate if approx_equal(classifier_to_error_rate[x], classifier_to_error_rate[best])]
        best = min(min_list) if min_list else best
    else:
        best = max(classifier_to_error_rate, key = lambda x: abs(make_fraction((classifier_to_error_rate[x])-Fraction(1,2))))
        min_list = [x for x in classifier_to_error_rate if approx_equal(classifier_to_error_rate[x], classifier_to_error_rate[best])]
        best = min(min_list) if min_list else best
    if approx_equal(make_fraction(classifier_to_error_rate[best]), Fraction(1,2)): raise NoGoodClassifiersError()
    else: return best

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 1: return -INF

    return Fraction(1,2) * (ln(1-error_rate)-ln(error_rate)) if error_rate else INF

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    missed = set() 
    # weighted vote of the weak classifers
    for point in training_points:
        overall_vote = 0
        for classifier, power in H:
            if point in classifier_to_misclassified[classifier]:
                overall_vote -= power
            else:
                overall_vote += power
        if overall_vote <= 0:
            missed.add(point)
    
    return missed



def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    missed = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    if len(missed) > mistake_tolerance: return False
    else: return True

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for point in point_to_weight:
        if point in misclassified_points:
            point_to_weight[point] = Fraction(1,2) * Fraction(1,error_rate) * Fraction(point_to_weight[point])
        else:
            point_to_weight[point] = Fraction(1,2) * Fraction(1,1-error_rate) * Fraction(point_to_weight[point])
    return point_to_weight

#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    point_to_weight = initialize_weights(training_points)
    H = []
    if max_rounds == INF: max_rounds = 10**32


    for itr in range(max_rounds):
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        try:
            best = pick_best_classifier(classifier_to_error_rate,use_smallest_error)
        except NoGoodClassifiersError:
            return H
        best_voting_power = calculate_voting_power(classifier_to_error_rate[best])
        H.append((best,best_voting_power))
        missed = classifier_to_misclassified[best]
        point_to_weight = update_weights(point_to_weight, missed, classifier_to_error_rate[best])

        if is_good_enough(H, training_points, classifier_to_misclassified,mistake_tolerance):
            return H

    return H
#### SURVEY ####################################################################

NAME = "Sami Amer" 
COLLABORATORS = "None" 
HOW_MANY_HOURS_THIS_LAB_TOOK = "3" 
WHAT_I_FOUND_INTERESTING = "adaboost is cool" 
WHAT_I_FOUND_BORING = "none" 
SUGGESTIONS = "none" 
