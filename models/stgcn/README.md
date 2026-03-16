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
*in progres*