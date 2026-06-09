# PJM-to-text module

This repo is an attempt to create a deep learning model that does CSLR (Continous Sign Language recognition) on PJM (Polish Sign Language).

## Custom Dataset
The model is trained on a dataset we collected with the help of volunteers. It currently consists of over **5,000 recordings** of Polish Sign Language (PJM) sentences.

## Key Features & Current Performance
* **Dual Modality:** Supports both **ISLR** (Isolated Sign Language Recognition) for single gloss classification and **CSLR** (Continuous Sign Language Recognition) for full sentence translation.
* **Skeleton-Based Processing:** Utilizes MediaPipe landmarks (Body, Face, Left/Right Hand) instead of raw video frames, reducing computational overhead and focusing on pure motion.
* **ISLR Performance (Reliable):** Our isolated sign classifier is currently achieving **83.82% - Top 1 Accuracy** and **76.3% Macro Recall** (with further optimizations still in progress).
* **CSLR Status (Disclaimer):** Hits **12% WER**, but currently overfits to sequence priors (memorizing sentence structures instead of translating signs). This metric is temporarily unreliable while we work on fixes.
* **Advanced Architecture:** Custom implementation of the CoSign model combined with Spatio-Temporal Graph Convolutional Networks (ST-GCN).
* **Robust Decoding:** CSLR inference supports standard Greedy CTC decoding as well as advanced Beam Search powered by a KenLM language model.

## Model Architecture

The diagram below illustrates the complete data flow and architecture of our modified CoSign model used for processing the skeleton graphs.

![PJMatch Graph Architecture](docs/PJMatch_graph.svg)

### Architecture Breakdown
The pipeline is divided into four primary modules:
1. **Preprocessing:** Raw skeleton data (543 landmarks: x, y, z, confidence) is normalized and filtered to isolate the coordinates used mostly for sign language execution.
2. **Spatial Module:** The data is split into distinct physical components (Body, Face, Mouth, Left/Right Hand). Each stream is processed through dedicated Spatio-Temporal Graph Convolutional Networks (ST-GCN).
3. **CoSign Module (Masking & Fusion):** The core mechanism of the network. The feature stream splits into a **Main Branch** (multiplied by an attention mask $\Phi$) and an **Inverse Branch** (multiplied by $1-\Phi$).
4. **Temporal & Prediction Module:** 1D Temporal CNNs (C3-P2) extract temporal patterns. These go to an Auxiliary CTC Head and pass through a BiLSTM layer before the final prediction by the Main CTC Gloss Head.

## Training

Training is controlled via `config.yaml`. Here is an example configuration and what the parameters do:

```yaml
training:
  epochs: 40
  batch_size: 8
  learning_rate: 0.0004
  weight_decay: 0.0001
  grad_clip: 1.0
  deterministic: false

model:
  dropout: 0.2

data:
  data_dir: "/pjm/extracted"           # Directory containing your .npy sequence files
  annotation_dir: "/pjm/ann_training"  # Directory with JSON annotations
  train_ann: "annotations/PJM_gloss.train.txt"
  dev_ann: "annotations/PJM_gloss.dev.txt"
  num_workers: 0
  dev_num_workers: 0
  cache_videos: false
  warmup_cache: false
  dev_cache_videos: false
  pin_memory: true
  mirror_prob: 0.5                     # Probability of applying horizontal flip augmentation
  temporal_scale: true                 # Randomly stretch or compress the sequence over time
  min_sentences: 4
  train_ratio: 0.75

optimizer:
  milestones: [20, 35]                 # Epochs at which the learning rate drops
  gamma: 0.1                           # Factor to multiply the learning rate by at milestones

system:
  checkpoint_dir: "checkpoints"        # Where weights are saved
  device: "auto"                       # Set to "cuda" or "cpu"

logging:
  tensorboard: true
  wandb: false
  log_dir: "runs"
  log_interval: 50
```

**Train Isolated Signs (ISLR):**
```bash
python src/train_islr.py
```

**Train Continuous Sentences (CSLR):**
```bash
python src/train_cslr.py
```

## Inference & Testing

Evaluate the trained models on the test/dev sets (make sure your `.npy` files are present in the `data_dir`). The scripts automatically load the best weights (`latest.pth` or `best_model.pth`) from your configured checkpoint directory.

**Test ISLR Model:**
Outputs overall accuracy and top confusion pairs for single signs.
```bash
python src/inference_islr.py
```

**Test CSLR Model:**
Outputs Word Error Rate (WER) and sentence-level predictions. By default, it attempts to use a Language Model (KenLM) with Beam Search if configured. To force standard greedy CTC decoding, use the `--greedy` flag.
```bash
python src/inference_cslr.py
python src/inference_cslr.py --greedy
```

## Citation
```
@inproceedings{10.5555/3304222.3304273,
author = {Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
title = {Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
year = {2018},
isbn = {9780999241127},
publisher = {AAAI Press},
booktitle = {Proceedings of the 27th International Joint Conference on Artificial Intelligence},
pages = {3634–3640},
numpages = {7},
series = {IJCAI'18}
}

@InProceedings{Jiao_2023_ICCV,
    author    = {Jiao, Peiqi and Min, Yuecong and Li, Yanan and Wang, Xiaotao and Lei, Lei and
                Chen, Xilin},
    title     = {CoSign: Exploring Co-occurrence Signals in Skeleton-based Continuous Sign
                Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision
                (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20676-20686}
}
@inproceedings{Yu_2018, series={IJCAI-2018},
   title={Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
   url={http://dx.doi.org/10.24963/ijcai.2018/505},
   DOI={10.24963/ijcai.2018/505},
   booktitle={Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence},
   publisher={International Joint Conferences on Artificial Intelligence Organization},
   author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
   year={2018},
   month=jul, pages={3634–3640},
   collection={IJCAI-2018} }

@misc{yan2018spatialtemporalgraphconvolutional,
      title={Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition}, 
      author={Sijie Yan and Yuanjun Xiong and Dahua Lin},
      year={2018},
      eprint={1801.07455},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1801.07455}, 
}
```
