# Third-Party Code: ST-GCN Implementation

## Source
- **Original Repository:** [https://github.com/hazdzz/STGCN](https://github.com/hazdzz/STGCN)
- **Author:** Hazdzz
- **Retrieved on:** March 2026

## License
This directory contains code licensed under the **GNU Lesser General Public License v2.1 (LGPL-2.1)**. 
The original license file is preserved in this directory as `LICENSE`.

## Modifications
1. Converted imports to relative imports for compatibility.
2. Changed default values of *enable_padding* in CausalConv1d and CausalConv2d to *True* 
3. Commented out the output computation block in *forward* of STGCNGraphConv for CSLR purposes
4. Commented out the output layers configuration in the constructor of STGCNGraphConv
5. Changed the for loop range in the constructor of STGCNGraphConv (we don't use the output layers so we want the *blocks* argument to not be forced to include such layers)
6. In *layers.py*, line 112, in forward:
```
112        x_in = self.align(x)[:, :, self.Kt - 1:, :]
```
for:
```
112         x_in = self.align(x)
```
to preserve the original size 
7. Changed *GraphConv* class (*layers.py*) to be compatible with CoSign-proposed [Ks, V, V] GSO shape
8. Overlayed a parametrized mask onto the GSO in *layers.py* (both in *GraphConv* and *ChebGraphConv*)
```
self.edge_importance = nn.Parameter(torch.ones_like(self.gso))
```