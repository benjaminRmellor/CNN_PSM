
	 A NOVEL APPLICATION OF CONVOLUTIONAL NEURAL 
	NETWORKS: IMPROVING SOUTH AMERICAN TREE RING 
	∂18O ESTIMATIONS

This repo is a comprehensive build of all developments performed for the dissertation project. As such, it will include all Neural Network Applications. Three neural network implementations are presented:
	1.	Artificial Neural Network, denoted as fANN. Inspired by Fang & Li, 2019
	2.	Convolutional Neural Network (A), denoted as dCNN. That performs limited feature selection to perform a regression of sample indices. 
	3.	Convolutional Neural Network (B), denoted as gCNN. That performs a global region estimate of ∂18OTR. 

However, implementations of PRYSM’s LMM function are left out to prevent infringement. 

The three implementations are partitioned into their respective runner scripts '{NN name}_runner.py'.
In which an example version of the script is used to demonstrate the methodology. Example hyperparameters that are called, are those used in the building of the model. Optionally, the weights may be called. But these weights are for the overtrained versions, that are saved for visualisation purposes. 