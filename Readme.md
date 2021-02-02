SPR PROJECT 2
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
END


