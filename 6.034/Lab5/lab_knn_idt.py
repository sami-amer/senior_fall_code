# MIT 6.034 k-Nearest Neighbors and Identification Trees Lab
# Written by 6.034 Staff

from fileinput import close
from pyexpat import features
from turtle import distance
from api import *
from data import *
import math

log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    while not id_tree.is_leaf():
        id_tree = id_tree.apply_classifier(point)
    
    return id_tree.get_node_classification()

#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    import collections
    classes = collections.defaultdict(list)
    for point in data:
        classes[classifier.classify(point)].append(point)

    return classes
#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    n_b = len(data)
    branches = split_on_classifier(data,target_classifier)
    res = 0
    for k in branches:
        n_bc = len(branches[k])
        res += -n_bc/n_b*log2(n_bc/n_b)
    
    return res
        


def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    branches = split_on_classifier(data,test_classifier)
    l = len(data)
    res = 0
    for k in branches:
        res += branch_disorder(branches[k],target_classifier)*(len(branches[k])/l)

    return res

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab_nn.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    min_disorder = INF
    best_class = None

    for classifier in possible_classifiers:
        disorder = average_test_disorder(data,classifier,target_classifier)
        if disorder < min_disorder:
            min_disorder = disorder 
            best_class = classifier
    
    if len(split_on_classifier(data,best_class)) < 2:
        raise NoGoodClassifiersError
    else:
        return best_class


## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node == None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    # if node is leaf, add classification to node
    # if node not leaf and data can be divided, add best classifier
    # if node cannot be divdied and no good classifier, do not assign class to noode
    # if we added a classifier, recurse to complete subtrees
    # return original input node

    if not branch_disorder(data,target_classifier) == 0:
        try:
            classifier = find_best_classifier(data,possible_classifiers,target_classifier)
            features = split_on_classifier(data,classifier) 

            possible_classifiers.remove(classifier)

            id_tree_node.set_classifier_and_expand(classifier,features)
            branches = id_tree_node.get_branches()

            for branch in branches:
                construct_greedy_id_tree(features[branch],possible_classifiers,target_classifier,branches[branch])

        except NoGoodClassifiersError:
            id_tree_node.set_node_classification(None)
            return id_tree_node

    else:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
    
    return id_tree_node
# To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

# # To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = "bark_texture" 
ANSWER_2 = "leaf_shape" 
ANSWER_3 =  "orange_foliage" 

ANSWER_4 = [2,3] 
ANSWER_5 = [3] 
ANSWER_6 = [2] 
ANSWER_7 = 2 

ANSWER_8 = "No"
ANSWER_9 = "No" 


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3 
BOUNDARY_ANS_2 = 4 

BOUNDARY_ANS_3 = 1 
BOUNDARY_ANS_4 = 2 

BOUNDARY_ANS_5 = 2 
BOUNDARY_ANS_6 = 4 
BOUNDARY_ANS_7 = 1 
BOUNDARY_ANS_8 = 4 
BOUNDARY_ANS_9 = 4 

BOUNDARY_ANS_10 = 4 
BOUNDARY_ANS_11 = 2 
BOUNDARY_ANS_12 = 1 
BOUNDARY_ANS_13 = 4 
BOUNDARY_ANS_14 = 4 


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    s = 0

    for x,y in zip(u,v):
        s+= x*y
    
    return s
def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return math.sqrt(dot_product(v,v))

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    s = 0
    for x,y in zip(point1,point2):
        s += (x-y)**2
    
    return math.sqrt(s)

def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    s = 0
    for x,y in zip(point1,point2):
        s += abs(x-y) 
    
    return s 

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    hamming = 0
    for x,y in zip(point1,point2):
        if x != y:
            hamming+=1
    
    return hamming
def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    return 1 - (dot_product(point1,point2))/(norm(point1)*norm(point2))

#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""

    distances = [(distance_metric(point,x),x) for x in data]
    distances.sort(key= lambda x: (x[0],x[1].coords))
    return [x[1] for x in distances[:k]]

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    closest = get_k_closest_points(point,data,k,distance_metric)
    closest = [point.classification for point in closest]
    return max(closest, key = lambda x: closest.count(x))

## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""

    total_tested = len(data)
    correct = 0

    for i in range(len(data)):

        test_set = data[i]
        train_set = data[:i]+data[i+1:]
        classification = knn_classify_point(test_set,train_set,k,distance_metric)

        if classification == test_set.classification:
            correct += 1
    
    return correct / total_tested
        

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""

    ks = len(data) // 2
    distance_metrics = {"manhattan_distance":manhattan_distance, "cosine_distance":cosine_distance, "euclidean_distance":euclidean_distance, "hamming_distance":hamming_distance}

    best_score = float('-inf')
    best_params = (None,None)

    for k in range(1,ks):
        for name,metric in distance_metrics.items():

            curr_score = cross_validate(data, k, metric)

            if  curr_score > best_score:
                best_score = curr_score 
                best_params = (k, metric)
    
    return best_params 



## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = "Overfitting" 
kNN_ANSWER_2 = "Underfitting" 
kNN_ANSWER_3 = 4 

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1 
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = "Sami Amer" 
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "4" 
WHAT_I_FOUND_INTERESTING = "Never heard of ID trees before" 
WHAT_I_FOUND_BORING = "Nothing, same old" 
SUGGESTIONS = "adding a visualization tool for KNN would be very nice, but its not that hard of a concept so maybe not necessary" 
