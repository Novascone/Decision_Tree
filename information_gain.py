import numpy as np
from entropy import entropy

def information_gain(parent,left_child,right_child):
    left_num = len(left_child) / len(parent)
    right_num = len(right_child) / len(parent)
    
    gain = entropy(parent) - (left_num * entropy(left_child) + right_num * entropy(right_child))
    return gain


