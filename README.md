# deep2Dspines
This Deep Learning algorithm allows the automatic detection of dendritic spines in 2D images. It was developed in Python 3.6. Here's the overview:

1 - The program starts by randomly extracting small images (windows) from every image placed in data/images.

2 - Then, it prompts the user to classify them as spines or not, making a dataset for the deep learning algorithm.

3 - Each image of the dataset is then replicated a user-defined amount of times, under different transformations (rotations, etc.), to increase the size of the dataset.

4 - The synthesized dataset is then used to train and test a neural network, generating an accuracy report.

5 - After this, the user can use the trained neural network to identify spines in a whole image. The chosen image is chopped in windows, each fed to the network for classification. Finally, a grayscale image is generated where each pixel value represents the probability of that pixel being part of a spine. 

To run this program, simply download the project an run Main.py in a Python interpreter.
