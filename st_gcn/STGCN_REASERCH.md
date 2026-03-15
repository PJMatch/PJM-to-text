# ST-GCN Docs Piotr Haber

## Idea

ST-GCN (Spacial Time Graph Convolitional Network) is an *graphs-oriented* network analogue for classic convolutional networks woring on *grid-like* objects.

In ST-GCN the object is a graph $G$ consisting of a set of vertecies $V$ and a set of edges $E$. Set $E$ can be divided into 2 subsets:
- *intra-body* - edges connecting joints in a skeleton
- *inter-frame* - edges connecting one specific joint in time (connections inbetween frames)[1]

## Implementation

In the project we are using:
- **Original Repository:** [https://github.com/hazdzz/STGCN](https://github.com/hazdzz/STGCN)
- **Author:** Hazdzz

This is an implementation of ST-GCN that was written specifically for action recognition and not CSLR. The key difference is that in action recognition we want just one output for a video (or a stream of frames), in CSLR we want multiple (glosses)

### *class Align*




## Necessary modifications

1. Enable padding for time convolution - the implementatino defaults padding to *False*, we don't want that because we want to preserve all frames for our LSTM module and to satisfy the CTC loss requirements (n_frames >= n_glosses)
2. Multiple outputs - we need to change the classification logic  

[1] Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition - Sijie Yan, Yuanjun Xiong, Dahua Lin

