# Cross-Language-Image-Classification
#######################################################################################

The project involves implementing and analyzing the performance of Logistic Regression, Support Vector Machine, and Ensemble models on the CIFAR-10 dataset. The language used for this purpose was Python.

#######################################################################################

Dataset Link: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) University of Toronto

Download and extract the "cifar-10-python" dataset from the official website.

Run the Data2CSV.ipynb file to generate the dataset in the required format.

#######################################################################################

Python Execution (3.10.12)

Open project.ipynb in Jupyter Notebook.

Run all cells sequentially.

#######################################################################################

C++ Execution (14.2.0)

Download eigen library for matrix computation: https://eigen.tuxfamily.org/dox/GettingStarted.html

Ensure the eigen-3.4.0 library is in the root folder.

Compiling: g++ project.cpp -o project -I eigen-3.4.0

Execution: ./project

For Faster Execution: Use the -O optimization argument during compilation:

g++ project.cpp -o project -I eigen-3.4.0 -O2

Note: Loading of the dataset may take more time

##################################################################################################################

Authors
Jake Jarosik (jj3268@drexel.edu)
Milan Varghese (mv644@drexel.edu)
