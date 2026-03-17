"""Module for generating Graph Signl Operator."""

import json

import numpy as np
import torch
from mediapipe.tasks.python import vision


class GSOGenerator:
    """Class for dynamic GSO generation."""

    def __init__(self, config_file):
        """Constructor of GSOGenerator class.

        Args:
            config_file: path to json file with skeleton configuration

        The json file needs to resemble a dict in this form:
            {
                "face": [],
                "mouth": [],
                "hands": [],
                "body": []
            }
        where the lists are lists of points that you want to inclue in traininig
        """
        self.master_edges = {
            "face": [
                (conn.start, conn.end)
                for conn in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
            ],
            "mouth": [
                (conn.start, conn.end)
                for conn in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
            ],
            "hands": [
                (conn.start, conn.end) for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS
            ],
            "body": [
                (conn.start, conn.end) for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS
            ],
        }

        # wrist_to_fingers hold wrist-to-fingers
        # connections absent in mediapipe but helpful (maybe) for the model
        wrist_to_fingers = [
            (0, 5),  # to index
            (0, 9),  # to middle
            (0, 13),  # to ring
            # to pinky already in mediapipe so we dont add it manually
        ]
        self.master_edges["hands"].extend(wrist_to_fingers)

        with open(config_file, "r") as config_f:
            self.config = json.load(config_f)

        # TODO: generate GSOs for each part

    def get_local_gso(self, target_idx: list, group_type: str):
        """Creates GSO only for a given subset of the MediaPipe's graph.

        Args:
            target_idx (list): list of global Mediapipe IDs
            group_type (str): 'face', 'body' or 'hands'
        Returns:
            local_gso: GSO of a target subset
        """
        n_vertex = len(target_idx)
        # Mapowanie: Global ID -> Local Index (0 do n-1)
        # Przykład: punkt 468 (nos) staje się indeksem 0 w nowej macierzy
        id_map = {global_id: i for i, global_id in enumerate(target_idx)}

        global_edges = self.master_edges.get(group_type, [])

        A = np.zeros((n_vertex, n_vertex))
        for start_id, end_id in global_edges:
            if start_id in id_map and end_id in id_map:
                i, j = id_map[start_id], id_map[end_id]
                A[i, j] = 1
                A[j, i] = 1

        local_gso = self.normalize_adj_matrix(A)

        return local_gso

    def normalize_adj_matrix(self, adj_matrix):
        """Normalize adjacancy matrix acording to the CoSign paper."""
        A = adj_matrix

        A_tilde = A + np.eye(A.shape[0])

        D_tilde = np.diag(np.sum(A_tilde, axis=1))

        # normalization as in the CoSign paper:
        # D^-0.5 * A * D^-0.5
        D_inv_sqrt = np.power(D_tilde, -0.5, where=D_tilde != 0)
        D_inv_sqrt = np.diag(D_inv_sqrt)

        GSO = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        return torch.tensor(GSO, dtype=torch.float32)


n_hands = 21
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
    (0,13),      # to ring
    (0,17),      # to pinky
    # palm
    (5, 9),   # index to middle
    (9, 13),  # middle to ring
    (13, 17)  # ring to pinky
]
# fmt: on

hands_conn = (n_hands, hand_edges)


n_body = 30
# fmt: off
body_edges = [
    # right arm
    (12,14), (14,16), 
    # left arm
    (11,13), (13,15), 
    # torso
    (11,12), (12,24), (24,23), (23,11),
    # right leg
    (24,26), (26,28), (28,30), (30,32), (32,28),
    # left leg
    (23,25), (25,27), (27,29), (31,32), (32,28)
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


if __name__ == "__main__":
    mouth = [(conn.start, conn.end) for conn in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS]
    print(len(mouth))
    dwd = GSOGenerator("./example_config.json")
