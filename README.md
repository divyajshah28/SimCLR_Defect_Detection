# Project Name: SimCLR Frame-work for defect detection on unknown context

This is the SimCLR Frame work developed on the contrastive learning approach for image classification and objetec detection. This work is built on the ideas from Ting Chen et. al. https://arxiv.org/abs/2006.10029 and the basic code was developed using ideas from Andras Beres: https://keras.io/examples/vision/semisupervised_simclr/

## Architecture

An illustration of the proposed SimCLR framework is shown below. The CNN and MLP layers are trained simultaneously to yield projections that are similar for augmented versions of the same image, while being dissimilar for different images, even if those images are of the same class of object. The trained model not only does well at identifying different transformations of the same image, but also learns representations of similar concepts (e.g., chairs vs. dogs), which later can be associated with labels through fine-tuning (https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)

![](https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s640/image4.gif "Image taken from Google research blog post")

The idea of SimCLR framework is very simple. An image is taken and random transformations are applied to it to get a pair of two augmented images. Each image in that pair is passed through an encoder to get representations. Then a non-linear fully connected layer is applied to get representations z. The task is to maximize the similarity between these two representations and for the same image.

![Alt text](https://github.com/divyajshah28/SimCLR_Defect_Detection/blob/main/files/simclr-general-architecture.png "Image taken from Amit Chaudhary's blog")

SimCLR uses a contrastive loss called “NT-Xent loss” (Normalized Temperature-Scaled Cross-Entropy Loss) given by the equation below: 

![](https://github.com/divyajshah28/SimCLR_Defect_Detection/blob/main/files/loss.png)

Once the SimCLR model is trained on the contrastive learning task, it can be used for transfer learning. For this, the representations from the encoder are used instead of representations obtained from the projection head. These representations can be used for downstream tasks like defect detection.

![](https://github.com/divyajshah28/SimCLR_Defect_Detection/blob/main/files/simclr-downstream.png)

## Data 

* Dataset used - MVTEC Anomaly Detection Dataset. This dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), which means it is not allowed to use it for commercial purposes.
* Images resized to 224x224.
* To use the dataset, download the datasets for the object types you want to train/test with. Create two sub-directories one named labelled and the other one named unlabelled. Create two new sub-directories inside the labelled directory called "good" and "anomaly". Put all your images of interest in the corresponding folders. Work is in progress to make this proess automated.

## Evaluation

* The basic structure of the SimCLR code developed here can only handle image classification and cannot do object detection yet. Hence, the evaluation was only done by using labelled and unlabelled dataset as training and another set as validation. A contrastive loss accuracy is calculated on the validation set. 
* For the current implementation, the model was trained on a dataset comprising of 175 unlabelled images, 342 training images and 280 validation images. The model was run for 10 epochs and the validation accuracy achieved was 

## Project Structure

/Contrastive_learning.py/i contains the main executable file
/constants.py/i contains the constants and hyperparameters

## References

* Amit Chaudhary, [The Illustrated SimCLR Framework](https://amitness.com/2020/03/illustrated-simclr/)
* András Béres, [Semi-supervised image classification using contrastive pretraining with SimCLR](https://keras.io/examples/vision/semisupervised_simclr/)
* Sik-Ho Tsang', [Review — SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sh-tsang.medium.com/review-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-5de42ba0bc66)
* Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4. [Paper](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf)
* Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982. [Paper](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)
* Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, A Simple Framework for Contrastive Learning of Visual Representations, arXiv preprint arXiv:2002.05709, 2020. [Paper](https://arxiv.org/abs/2002.05709)
* Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey, Big Self-Supervised Models are Strong Semi-Supervised Learners, arXiv preprint arXiv:2006.10029, 2020. [Paper](https://arxiv.org/abs/2006.10029)
