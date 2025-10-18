import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)

#Preview train data
print(train.head())

#descriptive statistics
print(train.describe())