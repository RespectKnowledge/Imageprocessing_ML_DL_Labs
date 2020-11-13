# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:19:53 2020

@author: Abdul Qayyum
"""

####################################################### Feature matrix ##########################
# load Dataset
path='D:\LAB4Segmentation\LAB6\\2dFeaturematrix.csv'


import numpy as np
import pandas as pd
# Set random seed to ensure reproducible runs
RSEED = 50
df = pd.read_csv(path)
X=np.array(df)
# load label vector
path1='D:\LAB4Segmentation\LAB6\\Labels.csv'
df1 = pd.read_csv(path1)
y=np.array(df1)

from sklearn.model_selection import train_test_split

# Extract the labels
#labels = np.array(df.pop('label'))
# split dataset into training and testing 
# 30% examples in test data
train, test, train_labels, test_labels = train_test_split(X,y, 
                                                          stratify = y,
                                                          test_size = 0.3, 
                                                          random_state = 52)


########################################################## decision Tree classifiers#############################
from sklearn.tree import DecisionTreeClassifier

# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=52)
tree.fit(train, train_labels)
print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')

#print(tree.tree_.max_depth)

#Assess Decision Tree Performance
# Make probability predictions
train_probs = tree.predict_proba(train)[:, 1]
probs = tree.predict_proba(test)[:, 1]

train_predictions = tree.predict(train)
predictions = tree.predict(test)
print(predictions)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, predictions))

################################################################################## RF classifier ######################
#Train Random forest to compare how it does against SVM
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, random_state=30)
clf.fit(train, np.array(train_labels))
train_predictions = clf.predict(test)

print(train_predictions)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, train_predictions))




