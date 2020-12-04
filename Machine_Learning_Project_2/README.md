# CS599-Machine_Learning-Project-2
CS599 ( Machine Learning Research) Project 2
# Purpose of the repository:
Purpose of this repository is to share the code of the project with the partner and the instructor to get their feedback about the project. By this way I can develop and improve the quality of my project.

# Re-do analysis:
To run the code , need to download the repository and run the make file from the command line.Make  file contains all the required commands. But the software and depedencies that I have mentioned below must be installed in the system.

Another way to run the code is, download the repository and go to command promt from the repository. Then type python argument. Here argument should be the name of the code. 

Example: python initialcode.py

This code can be run on google coLab also. I used google coLad to develop this project so it can be run there. For the case of google coLab no software will be needed to install.

Note: You must have python 3.4 or above version to run the code.
# Required Software and depedencies:
Install python3

Install numpy, matplotlib, tensorflow, and keras

# Goal

Here we will use MNIST handwritten digit recognition having the pixel value of all handwritten digits. We will be going to use CNN, to classify the digits. The final output will be a figure where the scenario of errorrate vs epoch size will be represented. 

# Data Set: 
MNIST Handwritten digit dataset

Developed by: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges

Url: http://yann.lecun.com/exdb/mnist/

Dataset Dimension and attributes: 

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. In the dataset, Images of digits were taken from a variety of scanned documents, normalized in size and centered. This makes it an excellent dataset for evaluating models, allowing the developer to focus on the machine learning with very little data cleaning or preparation required. Each image is a 28 by 28 pixel square (784 pixels total). Each sample image is 28x28 and linearized as a vector of size 1x784. So, the training and test datasets are 2-d vectors of size 60000x784 and 10000x784 respectively It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict.

The MNIST digits are grayscale images, with each pixel represented as a single intensity value in the range 0 (black) to 1 (white). You can think of the whole image as consisting of 784 numbers arranged in a plane of 28 rows and 28 columns. For color (RGB) images, however, each pixel consists of three numbers (one for Red intensity, one for Green, and one for Blue). Therefore color images are represented as arrays of shape rows × columns × 3, where the 3 indicates the depth of the image. For consistency, the grayscale MNIST images are treated as images of depth 1, with shape rows × columns × 1.

Input X features:

Input  is image in a array from. The input image is a 28 × 28 × 1 array of floating-point numbers representing grayscale intensities ranging from 0 (black) to 1 (white)

Input Summary will be

shape : (28, 28, 1) ,range : (0.0, 1.0)

Output Y label:

In output/ target it shows the classification from 0 to 9
So, the output / target summary will be
shape : (10,)  range : (0.0, 1.0)


# Link of the project Proposal
 URL: https://github.com/ashiqrobin/CS599_Machine_Learning_Research/blob/main/Machine_Learning_Project_2/Project%202.pdf
 
# Image of the Project
![imgae](https://github.com/ashiqrobin/CS599_Machine_Learning_Research/blob/main/Machine_Learning_Project_2/figure.JPG)

# Citation: 
Wan Zhu, “Classification of MNIST Handwritten Digit Database using Neural Network”.  
Figure: 7, Page: 5
Url: http://users.cecs.anu.edu.au/~Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_117.pdf
