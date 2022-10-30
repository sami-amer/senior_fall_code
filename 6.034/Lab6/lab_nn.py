# MIT 6.034 Neural Nets Lab
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else:
        return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+pow(e,-steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*((actual_output-desired_output)*(actual_output-desired_output))


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))

    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node

    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""

    outs = {}

    for neuron in net.topological_sort():
        val = 0
        for in_node in net.get_incoming_neighbors(neuron):
            node_val = node_value(in_node,input_values,outs)

            val += node_val * net.get_wires(in_node,neuron)[0].get_weight()

        threshed = threshold_fn(val)
        outs[neuron] = threshed
    
    return outs.get(net.get_output_neuron()),outs
            
            

#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""

    poss_perturbs = [step_size,-step_size,0]
    perturbs = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                perturbs.append([poss_perturbs[i],poss_perturbs[j],poss_perturbs[k]])

    max_inputs = None
    max_ret = float("-inf")

    for p in perturbs:
        mod_inputs = [x+y for x,y in zip(inputs,p)]
        ret = func(*mod_inputs)
        if ret > max_ret:
            max_inputs = mod_inputs
            max_ret = ret
    
    return max_ret, max_inputs


def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""

    # the dependencies are the output from node 1, weight of wire between node 1 and 2, an the output of node 2
    # we then repeat this until we hit the final layer
    # we want all wires, inputs, and neurons


    depend_set = {wire.startNode, wire.endNode, wire} 
    visited = set() 

    # wires = net.get_wires(startNode = wire.startNode,endNode = net.get_output_neuron())
    
    q = [wire.endNode]

    while q:

        curr = q.pop(0)
        depend_set.add(curr) # add current neuron to depend_set

        for wire in net.get_wires(startNode=curr): # for every wire that stems from this wire

            if wire.endNode not in visited:
                q.append(wire.endNode) # we don't add to set now, it gets added when its popped
                visited.add(wire.endNode)

            depend_set.add(wire)
    
    return depend_set

    
    
    return depend_set


def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """

    deltas = {}

    for n in reversed(net.topological_sort()):
        n_val = neuron_outputs[n]        
        if net.is_output_neuron(n): # final layer
            deltas[n] = n_val * (1-n_val)*(desired_output-n_val)
        else:
            s = 0 # sum of edges
            # for out_node in net.get_outgoing_neighbors(n): # for every edge
            #     s += deltas[out_node]*net.get_wires(n,out_node)[0].get_weight() # do the summation
            s = [deltas[x]*net.get_wires(n,x)[0].get_weight() for x in net.get_outgoing_neighbors(n)]
            deltas[n] = n_val*(1-n_val) * sum(s) # then get the val
    
    return deltas
                

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""

    deltas = calculate_deltas(net,desired_output,neuron_outputs)

    for wire in net.get_wires():

        node_val = node_value(wire.startNode,input_values,neuron_outputs) 
        deltaW = r * node_val  * deltas[wire.endNode]
        wire.set_weight(wire.get_weight()+deltaW)

    return net


def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""

    output, neuron_outputs = forward_prop(net,input_values, sigmoid)    
    iters = 0
    while accuracy(desired_output,output) <= minimum_accuracy:
        iters += 1
        # update weights
        net = update_weights(net,input_values,desired_output,neuron_outputs,r)
        # forward prop to get new output
        output, neuron_outputs = forward_prop(net,input_values, sigmoid)
    
    return net, iters
    



#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 45 
ANSWER_2 = 12 
ANSWER_3 = 5 
ANSWER_4 = 100 
ANSWER_5 = 70

ANSWER_6 = 1 
ANSWER_7 = "checkerboard" 
ANSWER_8 = ["small","medium","large"]
ANSWER_9 = "B" 

ANSWER_10 = "D" 
ANSWER_11 = ["A","C",] 
ANSWER_12 = ["A","E"]


#### SURVEY ####################################################################

NAME = "Sami Amer" 
COLLABORATORS = "None" 
HOW_MANY_HOURS_THIS_LAB_TOOK = "2" 
WHAT_I_FOUND_INTERESTING = "I have taken this before, but I liked the viz" 
WHAT_I_FOUND_BORING = "Back-prop" 
SUGGESTIONS = "More visual problem prompts" 
