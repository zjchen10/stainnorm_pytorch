# stainnorm_pytorch

## WARNING: THIS IMPLEMENTATION IS STILL IN BETA!

PyTorch implementation of Macenko stain normalization (https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf).

Works with most PyTorch version. This implementation is adapted from TIA Toolbox (https://tia-toolbox.readthedocs.io/en/stable/_notebooks/jnb/02-stain-normalization.html). Check the Jupyter Notebook for how to use the algorithm.

Note: the algorithm depends on torch.linalg.eigh. Different hardware and software may compute slightly different eigenvectors. The same image can be normalized slightly differently on different machines/devices or with different versions of pytorch.
