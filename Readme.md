# Hand written Digit recognition. (SVM and CNN) 
# Project Description:-

SVM :- Implement linear and kernel SVM on MNIST dataset. You have to try different kernels (linear, polynomial, RBF) and compare results in your report. You can use any online toolbox for this, e.g. LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) or any MATLAB built-in function. You can apply PCA and LDA here for dimensionality reduction.

Deep Learning :- Build a Convolutional Neural Network. Train it on MNIST training set and test it on testing set. You can design your architecture or use the architecture introduced in LeCun's paper [1]. Dataset You can download MNIST from http://yann.lecun.com/exdb/mnist/. The description of Dataset is also on the website. Basically, it has a training set with 60000 28 x 28 grayscale images of handwritten digits (10 classes) and a testing set with 10000 images.

# Monkey species. (CNN + Transfer learning)
# Project Description:-

For this part of the project, you will be applying a deep learning technique commonly referred to as Transfer Learning. Specifically, Transfer Learning can be used for deep learning applications where either data or computational power are restricted. Here you are given a data set with ten classes (ten different monkey species) with only 140 images per class. The first task will be to train a simple convolutional neural network using these images (a very simple 3-5 convolutional network will suffice) and 1 test the accuracy of the model using the validation set. Because of the low number of training samples, you will see that the test accuracy of the model will be lower than expected. This is where transfer learning can help. The next task will be to download a pretrained model for image classification (for example VGG, Alexnet, InceptionNet) and use it as a feature extractor. To do this, you will remove the last fully connected layers of the pretrained model and replace it with untrained fully connected layers to classify the monkey species. During training, you will freeze the convolutional layer parameters so that they remain the same and only update the fully connected layers at the end. In this way, the convolutional layers act as generalized feature extractors that have already been pretrained on millions of other images (that werent necessarily all monkeys) while the fully connected layers are able to take these features and classify our images. You should see a substantial boost in accuracy even though we have the same amount of training samples. To further boost the performance of the network, you can unfreeze the convolutional layers and train for a few more epochs with a small step size to _ne tune the network to extract even more predictive power. Dataset You can download the dataset from here: https://www.kaggle.com/slothkong/10-monkey-species

# Run the Code:
STEP 1

To run the matlab file you will require Matlab toolbox as follows:-
Deep Learning Toolbox Model For AlexNet Network
Deep Learning Toolbox
Parallel Computing Toolbox
Optimization Toolbox
Symbolic Math Toolbox
Statistics and Machine Learning Toolbox

Library SVM required files for matlab already added in Code Folder.(*.mexw64 files)

--------------------------------------------------------------------------------------------

STEP 2

Include the dataset in the Code Folder after step 1.
Dataset:-
Part 1-MNIST
Files:-
t10k-labels.idx1-ubyte
t10k-images.idx3-ubyte
train-images.idx3-ubyte
train-labels.idx1-ubyte

Part 2-10 Monkey Species
Folder :- training --> n0,n1,....,n9 --> *.jpg
	  validation --> n0,n1,....,n9 --> *.jpg

*.jpg means Dataset Images

-----------------------------------------------------------------------------------------

STEP 3

Run all files in matlab
Open the files in Matlab and press the Run it after completing STEP 2

Part 1 

MNIST
SVM:- run SVM.m 

Line 11 :- DR = 2;    Select-> 1: PCA , 2: LDA
Line 29 :- kernel = 2; Select-> 0:linear , 1:polynomial , 2:rbf

CNN:- run CNN_MNIST.m 

Line 98 :-    for (i=1:4)
Change number of iterations in order to train the model number of times.

Part 2 

10 Monkey Species

CNN:- run Simple_CNN.m

Tranfer Learning :- run TF_CNN.m
Line 6 :- freeze = 1;   Select -> For Conv Layer param freeze = 1, freeze = 0(To unfreeze).

-------------------------------------------------------------------------------------------------


