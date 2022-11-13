# Project Name: SimCLR Frame-work for defect detection on unknown context

This is the SimCLR Frame work developed on the contrastive learning approach for image classification and objetec detection. This work is built on the ideas from Ting Chen et. al. https://arxiv.org/abs/2006.10029 and the basic code was developed using ideas from Andras Beres: https://keras.io/examples/vision/semisupervised_simclr/

## Architecture

An illustration of the proposed SimCLR framework is shown below. The CNN and MLP layers are trained simultaneously to yield projections that are similar for augmented versions of the same image, while being dissimilar for different images, even if those images are of the same class of object. The trained model not only does well at identifying different transformations of the same image, but also learns representations of similar concepts (e.g., chairs vs. dogs), which later can be associated with labels through fine-tuning (https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)

![](https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s640/image4.gif)

