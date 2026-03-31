This directory contains the custom PyTorch DataLoader for the PHOENIX-2014T feature sequences.

The extracted .npy features required for training are hosted on Google Drive. You can download them here:
https://drive.google.com/drive/folders/1gqOBRRVJk1FILYV09y9bxQuxtuoWvaVO

Note: You must download the dataset to your local machine before initiating the training pipeline. Do not stream directly from Google Drive.

PhoenixDataset: A custom PyTorch Dataset class scans the provided local directory (data_dir) for all .npy files and loads them dynamically during training.

sequence_collate_fn: A custom collate function used by the DataLoader. The video sequences vary in frame length, this function pads shorter sequences with zeros to match the longest sequence in the current batch. It returns both the padded batch tensor and a tensor containing the original sequence lengths.