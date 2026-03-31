"""Module for generating Graph Signl Operator."""

import json

import numpy as np
import torch
from mediapipe.tasks.python import vision


class GSOGenerator:
    """Class for dynamic GSO generation."""

    def __init__(self, config_path):
        """Constructor of GSOGenerator class.

        Args:
            config_path: path to json file with skeleton configuration

        The json file MUST resemble a dict in this form:
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

        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.gsos = {}
        self._generate_all_gsos()

    def _generate_all_gsos(self):
        """Iterates over config and generates GSO matricies."""
        topology_map = {
            "l_hand": "hands",
            "r_hand": "hands",
            "face": "face",
            "mouth": "mouth",
            "body": "body",
        }
        for group_name, indices in self.config.items():
            if not indices:
                print(f"Warning: Group {group_name} has no indices in config. Skipping.")
                continue

            base_topology = topology_map.get(group_name, group_name)

            self.gsos[group_name] = self.get_local_gso(indices, base_topology)

    def get_local_gso(self, target_idx: list, group_type: str):
        """Creates GSO only for a given subset of the MediaPipe's graph.

        Args:
            target_idx (list): list of global Mediapipe IDs
            group_type (str): 'face', 'body' or 'hands'
        Returns:
            local_gso: GSO of a target subset
        """
        n_vertex = len(target_idx)
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
        """Normalize adjacancy matrix acording to the CoSign paper (eqn. 3)."""
        A = adj_matrix
        n = A.shape[0]
        Eye = np.eye(n)
        epsilon = 0.001

        # calculate GSO for k=0 (A0 = I)
        d0 = np.sum(Eye, axis=1) + epsilon
        d0_inv_sqrt = np.power(d0, -0.5)
        D0_inv_sqrt = np.diag(d0_inv_sqrt)
        GSO_0 = D0_inv_sqrt @ Eye @ D0_inv_sqrt

        # calculate GSO for k=1 (A1 = A)
        d1 = np.sum(A, axis=1) + epsilon  # Lambda_ii dla A
        d1_inv_sqrt = np.power(d1, -0.5)
        D1_inv_sqrt = np.diag(d1_inv_sqrt)
        GSO_1 = D1_inv_sqrt @ A @ D1_inv_sqrt

        GSO = np.stack([GSO_0, GSO_1], axis=0)

        return torch.tensor(GSO, dtype=torch.float32)


if __name__ == "__main__":
    pose = ([(conn.start, conn.end) for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS],)
    print(pose)
    dwd = GSOGenerator("./example_config.json")
