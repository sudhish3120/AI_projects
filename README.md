# AI_projects

### Neural Network
Implemented a feedforward neural network from scratch that predicts the quality of wine based on intrinsic factors (acidity, pH, sugar content, etc.) using NumPy and Pandas. The program takes as input the number of layers, size of each layer and activation functions (from a given list). It performs k-fold cross validation for k=5 using a given dataset and produces a plot with x-axis as the epoch number and y-axis as the average training loss across all experiments for the given epoch.

### Reinforcement Learning
Leveraged Q-learning in implementing two agents (sender and receiver) capable of establishing a communication protocol in order to solve a maze problem using Python. The goal is for the receiver agent to escape the maze using the signals sent by the sender agent. Both agents get a reward of 1.0 upon the receiver escaping the maze. The agents are trained over a given number of episodes, and both agents update their Q-functions after each iteration. The program produces multiple plots showing average discounted reward obtained by either agents as a function of the number of episodes.

### Medical diagnostic prediction
Implemented an unsupervised learning model via a Bayesian network for genetic condition prediction. Used the EM algorithm to enhance diagnostic accuracy from given initial conditional probability tables (CPTs). The program produces a plot that shows prediction accuracy on a test set before and after running the EM algorithm for different initializations of the CPTs.
