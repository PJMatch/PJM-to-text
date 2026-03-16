"""Module for generating Graph Signl Operator."""

import numpy as np
import torch

# fmt: off
hand_edges = [
    # fingers' bones
    (0,1), (1,2), (2,3), (3,4),         # thumb
    (5,6), (6,7), (7,8),         # index
    (9,10), (10,11), (11,12),    # middle
    (13,14), (14,15), (15,16),  # ring
    (17,18), (18,19), (19,20),  # pinky
    # writs-to-finger (mediapipe doesnt include these natively) 
    (0,5),      # to index
    (0,9),      # to middle 
    (0,13)      # to ring
    (0,17)      # to pinky
    # palm
    (5, 9),   # index to middle
    (9, 13),  # middle to ring
    (13, 17)  # ring to pinky
]
# fmt: on

# fmt: off
body_edges = [
    # right arm
    (12,14), (14,16), 
    # writs-to-finger (mediapipe doesnt include these natively) 
    (0,5),      # to index
    (0,9),      # to middle 
    (0,13)      # to ring
    (0,17)      # to pinky
    # palm
    (5, 9),   # index to middle
    (9, 13),  # middle to ring
    (13, 17)  # ring to pinky
]
# fmt: on


def create_gso(n_vertex, edges):
    """Creates a GSO for given graph edges."""
    A = np.zeros((n_vertex, n_vertex))  # adjecancy matrix
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    A_tilde = A + np.eye(n_vertex)

    D_tilde = np.diag(np.sum(A_tilde, axis=1))

    # normalization as in the CoSign paper:
    # D^-0.5 * A * D^-0.5
    D_inv_sqrt = np.power(D_tilde, -0.5, where=D_tilde != 0)
    D_inv_sqrt = np.diag(D_inv_sqrt)

    GSO = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return torch.tensor(GSO, dtype=torch.float32)
