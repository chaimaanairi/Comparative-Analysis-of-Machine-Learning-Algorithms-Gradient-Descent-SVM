# Comparative Analysis of Machine Learning Algorithms: Gradient Descent & SVM


This project explores the performance comparison of two popular machine learning algorithms for a binary classification problem:

1. **Artificial Neural Networks (ANN)** trained with different optimization techniques:
   - Stochastic Gradient Descent (SGD)
   - Batch Gradient Descent
   - Mini-Batch Gradient Descent  

2. **Support Vector Machines (SVM)** with various kernel functions:
   - Linear
   - Polynomial
   - Gaussian RBF  

The dataset is generated using Scikit-learn's `make_moons` function, containing two classes with added noise for realism.

## Objective
- Compare ANN and SVM performance on a binary classification task.
- Analyze the effect of different optimization methods for ANN.
- Evaluate the impact of kernel choice on SVM performance.
- Assess models using Accuracy, Precision, Recall, and F1-Score.

## Dataset
- **Source:** Scikit-learn `make_moons`
- **Samples:** 400
- **Features:** 2
- **Noise:** 20%
- Split into training, validation, and test sets.

## Methods
- **ANN:** Tested with 1, 2, and 3 hidden layers using Binary Cross-Entropy loss and Sigmoid activation.
- **SVM:** Optimized with GridSearchCV to find the best kernel parameters.

## Results
- Mini-batch gradient descent achieved the most stable ANN training.
- Gaussian RBF kernel in SVM achieved the highest performance:
  - Accuracy: 95%
  - Precision: 97%
  - Recall: 92%
  - F1-Score: 94%  

SVM with Gaussian RBF clearly outperformed ANN models on this dataset.

## Conclusion
- SVM models, particularly with Gaussian RBF kernel, performed best on the binary classification task.
- ANN performance varied with network depth and optimization method.
- Selecting the appropriate algorithm, parameters, and optimization technique is critical for model success.

