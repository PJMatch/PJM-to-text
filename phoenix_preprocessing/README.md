Phoenix Dataset Preprocessing

Pipeline to extract kinematic features from the RWTH-PHOENIX-Weather 2014T dataset using the MediaPipe Tasks API.

Scripts
extract_phoenix.py: Batch processes train, dev, and test video splits into npy feature sequences.
extract_phoenix_samples.py: Initial test script used on a small data sample to verify the extraction pipeline before full-scale processing.
result_visualization.py: 2D OpenCV visualizer specifically used to sanity-check the npy skeletons generated from the initial sample.

Extracted Data Format
The files were extracted to project storage outside github
Each npy file holds a 2D array of shape (N, 1659), where N is the total frame count. Missing detections are strictly zero-padded (0.0).


Feature Vector Map (1659 values per frame):
Values represent normalized X, Y, Z spatial coordinates.
0 to 98 (99 values) Pose: 33 body landmarks.
99 to 1532 (1434 values) Face: 478 facial and iris landmarks.
1533 to 1595 (63 values) Left Hand: 21 joint landmarks.
1596 to 1658 (63 values) Right Hand: 21 joint landmarks.

You can find the extracted dataset here:
https://drive.google.com/drive/folders/1gqOBRRVJk1FILYV09y9bxQuxtuoWvaVO