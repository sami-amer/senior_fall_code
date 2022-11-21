# MIT 6.034 Support Vector Machines Lab
# Written by 6.034 staff

from svm_data import *
from functools import reduce
import math


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""

    s = 0

    for x,y in zip(u,v):
        s+= x*y
    
    return s

    

def norm(v):
    """Computes the norm (length) of a vector v, represented
    as a tuple or list of coords."""
    return math.sqrt(dot_product(v,v))


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w,point.coords) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""

    val = positiveness(svm,point)
    if val == 0: return 0
    elif val > 0: return 1 
    else: return -1

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    violations = set() 
    for point in svm.training_points:
        if point in svm.support_vectors:
            if positiveness(svm,point) != point.classification:
                violations.add(point)
        else:
            if -1 < positiveness(svm,point) < 1 :
                violations.add(point)
    
    return violations



#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""

    violations = set()

    for point in svm.training_points:
        if point.alpha < 0:
            violations.add(point)
        elif point.alpha == 0 and point in svm.support_vectors:
            violations.add(point)
        elif point not in svm.support_vectors and point.alpha != 0:
            violations.add(point)
    
    return violations



def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    dimensions = len(svm.training_points[0].coords)
    s1 = 0
    vec = Point(None,[0]*dimensions)
    for point in svm.training_points:
        ya = point.classification * point.alpha
        s1 += ya
        vec = vector_add(scalar_mult(ya,point.coords),vec)

    return True if s1 == 0 and vec == svm.w else False



#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    misclass = set()
    for point in svm.training_points:
        if classify(svm,point) != point.classification:
            misclass.add(point)
    
    return misclass
            


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    support_vecs = []

    for point in svm.training_points:
        if point.alpha > 0:
            support_vecs.append(point)
    
    svm.support_vectors = support_vecs

    new_w = Point(None,(0,0))
    for point in svm.training_points:
        ya = point.classification * point.alpha
        new_w = vector_add(scalar_mult(ya,point.coords), new_w)
    
    svm.w = new_w

    # for b, we want minimum value from a negative support, and max value from positive support, then average the two
    b_max = float('-inf')
    b_min = float('inf')
    for point in svm.support_vectors:
        new_b = point.classification - dot_product(svm.w,point.coords) 
        if point.classification < 0: # neg support
            b_min = min(b_min, new_b)
        else: # pos support
            b_max = max(b_max,new_b)
    
    b = (b_max + b_min)/2
    svm.set_boundary(new_w,b)

    return svm


#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11 
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2 

ANSWER_5 = ["A","D"] 
ANSWER_6 = ["A","B","D"] 
ANSWER_7 = ["A","B","D"]
ANSWER_8 = [] 
ANSWER_9 = ["A","B","D"] 
ANSWER_10 = ["A","B","D"] 

ANSWER_11 = False 
ANSWER_12 = True 
ANSWER_13 = False 
ANSWER_14 = False 
ANSWER_15 = False
ANSWER_16 = True 

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8] 
ANSWER_19 = [1,2,4,5,6,7,8]  

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Sami Amer" 
COLLABORATORS = "None" 
HOW_MANY_HOURS_THIS_LAB_TOOK = "2" 
WHAT_I_FOUND_INTERESTING = "Viz of SVMs was cool" 
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = "nothing" 
