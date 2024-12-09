# Instagram Fake Account Detection with Machine Learning

## Overview
This project uses various machine learning models to classify Instagram accounts as either fake or genuine. The dataset consists of Instagram account features with a balanced distribution of 50% fake and 50% genuine accounts. We explore several models, including Random Forest, XGBoost, and Sequential Neural Networks, to determine the best-performing model based on accuracy, precision, and ROC-AUC.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [How to Run](#how-to-run)

## Installation
Install the required libraries by running the following:

```bash
pip install -r requirements.txt
```
You can also clone this repository to your local machine:
```bash
gh repo clone laurieye/Detection-of-Fake-Spammer-and-Genuine-Accounts-on-Instagram
```

## Dataset
The dataset used in this project contains 696 Instagram accounts with various features such as:

- Number of followers
- Number of posts
- Profile picture status
- Username characteristics, etc.

The data is divided into 50% fake and 50% real accounts, providing a balanced dataset to build and evaluate our models.

## Modeling
We implement and test the following models:

- **Regression-based Methods:Logistic Regression** (baseline)
- **Tree-based Methods: Decision Tree and Random Forest** (with Grid Search for hyperparameter tuning)
- **Boosting-based Methods: Gradient Boosting and XGBoost** (with Random Search for hyperparameter tuning)
- **Advanced Non-linear Methods: SVM and Sequential Neural Network** (designed for complex data patterns)

### Key Techniques:
- **Feature Engineering**: Focused on features such as followers count, posts count, and username length.
- **Hyperparameter Tuning**: Both Grid Search and Random Search are employed for tuning the models.
- **Cross-validation**: Used to ensure model robustness and prevent overfitting.

## Evaluation
The models are evaluated using the following metrics:

- **Accuracy**: Measures the overall performance.
- **Precision**: Helps minimize false positives, which is crucial in preventing unnecessary misclassification.
- **ROC-AUC**: Assesses the model's ability to distinguish between fake and genuine accounts across different decision thresholds.

Additionally, **Bootstrapping** is used to evaluate the model’s stability and generalization across different subsets of the data.

## Results
- **Best Model**: Random Forest demonstrated superior interpretability and feature importance while maintaining competitive performance in terms of accuracy and precision.
- **XGBoost**: Achieved the highest accuracy but did not provide as clear feature insights as Random Forest.
- **Sequential Neural Network**: Performed well in capturing non-linear patterns but showed slightly lower accuracy compared to Random Forest and XGBoost.

## Future Work
- **Larger Datasets**: Future improvements could focus on incorporating a larger, more diverse set of Instagram accounts to improve model generalization.
- **Behavioral Features**: Adding features like interaction patterns or engagement metrics may further improve model performance.
- **Deep Learning**: Testing deep learning techniques on more complex and high-dimensional datasets could improve performance in the future.

## How to Run
Open the notebook in Google Colab by clicking on the link below: 
`https://colab.research.google.com/drive/1kHsFbrw1QHFBs1iPIuib-JHoMj-zTICL#scrollTo=y8HChLvRVf6A`

### Instructions to Use:
1.  `https://github.com/laurieye/Detection-of-Fake-Spammer-and-Genuine-Accounts-on-Instagram`
2.  Google Colab: `https://colab.research.google.com/drive/1kHsFbrw1QHFBs1iPIuib-JHoMj-zTICL#scrollTo=y8HChLvRVf6A`


### Open Colab Notebook

Follow the instructions in the Colab notebook to load the dataset, train the models, and evaluate them based on the provided metrics.
