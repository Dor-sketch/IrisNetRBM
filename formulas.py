"""
This file contains the formulas used in the Restricted Boltzmann Machine
"""

import numpy as np


def get_hidden_energy_difference(v=None, b=None, w=None, k=None) -> float:
    """
    calculate the energy difference given a chabge in a visible or hidden layer
    using the formula:
    For the hidden layer: (assum we check for neuron 1)
            ΔE_1 = b_1 + ΣW_{i1}v_i
    args:
    - v: visible layer units
    - b: bias of the hidden layer
    - w: weights of the connections between the visible and hidden layers
    - k: index of the hidden layer unit
    """
    # Vectorized calculation of energy difference to improve efficiency
    return b[k] + np.dot(v, w[:, k])


def get_visible_energy_difference(h=None, a=None, w=None, i=None) -> float:
    """
    calculate the energy difference given a chabge in a visible or hidden layer
    using the formula:
    For the visible layer: (assum we check for neuron 1)
            ΔE_1 = a_1 + ΣW_{1j}h_j
    args:
    - h: hidden layer units
    - a: bias of the visible layer
    - w: weights of the connections between the visible and hidden layers
    - i: index of the visible layer unit
    """
    # Vectorized calculation of energy difference to improve efficiency
    return a[i] + np.dot(h, w[i, :])


def get_probability(v=None, b=None, w=None, T=1, k=None):
    """
    calculate the probability of the hidden layer given the visible layer
    using the formula:
    P_j(h_j = 1|v) =
                             1
                       --------------
                       1 + exp[-ΔE/T]
    args:
    - h: hidden layer units
    - b: bias of the hidden layer
    - w: weights of the connections between the visible and hidden layers
    - T: temperature of the system
    - k: index of the hidden layer unit (optional)
    """
    if k is not None:
        # return the probability of the kth hidden unit given the visible layer
        activation_energy = get_hidden_energy_difference(v, b, w, k)
        return 1 / (1 + np.exp(-activation_energy / T))

    # Vectorized calculation of hidden unit probabilities to improve efficiency
    activation_energy = np.dot(v, w) + b
    # might cause nan or overflow
    return 1 / (1 + np.exp(-activation_energy))

def get_visible_probability(h=None, a=None, w=None, T=1, i=None):
    """
    calculate the probability of the visible layer given the hidden layer
    using the formula:
    P_i(v_i = 1|h) =
                             1
                       --------------
                       1 + exp[-ΔE/T]
    args:
    - h: hidden layer units
    - a: bias of the visible layer
    - w: weights of the connections between the visible and hidden layers
    - T: temperature of the system
    - i: index of the visible layer unit (optional)
    """
    if i is not None:
        # return the probability of the ith visible unit given the hidden layer
        activation_energy = get_visible_energy_difference(h, a, w, i)
        return 1 / (1 + np.exp(-activation_energy / T))

    # Vectorized calculation of visible unit probabilities to improve efficiency
    activation_energy = np.dot(h, w.T) + a
    # might cause nan or overflow
    return 1 / (1 + np.exp(-activation_energy))

v = np.array([0, 1, 1, 0, 0, 0, 0])
h = np.array([0, 1])
b = np.array([1, 1])
a = np.array([1, 1, 1, 1, 1, 1, 1])
w = np.array([[1,  1],
              [1, 1],
              [1, 1],
              [1,  1],
              [1, 1],
              [1,  1],
              [1, 1]])

print(get_hidden_energy_difference(v, b, w, 0))
print(get_hidden_energy_difference(v, b, w, 1))
print(get_probability(v, b, w, 1, 0))
# lets say that in a random sample, the hidden layer was activated to [1, 0]
# we can calculate the probability of the visible layer given the hidden layer
h = np.array([1, 0])
for i in range(7):
    print(get_visible_energy_difference(h, a, w, i))

print(get_visible_probability(h, a, w, 1, 0))