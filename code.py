# Importing libraries used in analysis 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
data = pd.read_csv('combined_dataset.csv')  # Adjust the filename as needed
print(data.head())

# Listing number of rows and columns parsed in the whole dataset 
print(f'Shape of dataset: {data.shape}')

# Summary Statistics 
print(data.describe())

# Sum for number of observations with missing values 
print(data.isnull().sum())

# Displaying the frequency chart for output variable (bstatus = bankruptcy status)
plt.figure(figsize=(10, 6))
plt.hist(data['bstatus'], bins=2, alpha=0.7, color='blue', edgecolor='black')
plt.title('Bankruptcy Status Frequency')
plt.xlabel('Bankruptcy Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non Bankrupt', 'Bankrupt'])
plt.show()

# Data preprocessing
data_preprocessing = data.copy()
data_preprocessing.drop(['ID', 'bstatus'], axis=1, inplace=True)

normalized_data = (data_preprocessing - data_preprocessing.mean()) / data_preprocessing.std()
print(normalized_data.head())

# Check for Correlation among all variables  
corr = normalized_data.corr()
plt.figure(figsize=(10, 10))
plt.matshow(corr, fignum=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Matrix')
plt.show()

# Check for Skewness in the dataset 
print(normalized_data.skew())

# Draw histograms 
plt.figure()
plt.hist(normalized_data['cf_td'], bins=30, color='blue', alpha=0.75)
plt.xlabel("cf_td")
plt.title("cf_td Histogram")
plt.show()

# Skewness transformation
skness_positive = []
skness_negative = []
for ratio in normalized_data.columns:
    sk = normalized_data[ratio].skew()
    if sk > 2:
        if np.min(normalized_data[ratio]) < 0:
            for value in normalized_data[ratio]:
                transformed_value = np.log10(value + np.abs(np.min(normalized_data[ratio])) + 1)
                skness_positive.append(transformed_value)
            a = pd.Series(skness_positive)
            print(ratio, a.skew())
            normalized_data.loc[:, ratio] = a
            skness_positive.clear()
        else: 
            for value in normalized_data[ratio]:
                c = np.log10(value + 1)
                skness_positive.append(c)
            a = pd.Series(skness_positive)
            print(ratio, a.skew())
            normalized_data.loc[:, ratio] = a
            skness_positive.clear()
    else:
        if np.min(normalized_data[ratio]) < 0:
            for value in normalized_data[ratio]:
                transformed_value = np.sqrt(value + np.abs(np.min(normalized_data[ratio])) + 1)
                skness_negative.append(transformed_value)
            x = pd.Series(skness_negative)
            print(ratio, x.skew())
            normalized_data.loc[:, ratio] = x
            skness_negative.clear()
        else: 
            for value in normalized_data[ratio]:
                y = np.sqrt(value + 1)
                skness_negative.append(y)
            x = pd.Series(skness_negative)
            print(ratio, x.skew())
            normalized_data.loc[:, ratio] = x
            skness_negative.clear()

normalized_data.columns = ['trans_cf_td', 'trans_ca_cl', 'trans_re_ta', 'trans_ni_ta', 'trans_td_ta', 'trans_s_ta', 'trans_wc_ta', 'trans_wc_s', 'trans_c_cl', 'trans_cl_e', 'trans_in_s', 'trans_mve_td']
data = data.join(normalized_data)

# Save the transformed dataset
data.to_csv('transformed_new_data.csv', sep=',', index=False)

# Data Reduction
data_reduced = data.copy()
data_reduced.drop(['cf_td', 'ca_cl', 're_ta', 'ni_ta', 'td_ta', 's_ta', 'wc_ta', 'wc_s', 'c_cl', 'cl_e', 'in_s', 'mve_td'], axis=1, inplace=True)

# Save the reduced dataset
data_reduced.to_csv('data.csv', sep=',', index=False)

# Check for Correlation among all variables in the reduced dataset  
corr1 = data_reduced.corr()
plt.figure(figsize=(10, 10))
plt.matshow(corr1, fignum=1)
plt.colorbar()
plt.xticks(range(len(corr1.columns)), corr1.columns, rotation=90)
plt.yticks(range(len(corr1.columns)), corr1.columns)
plt.title('Correlation Matrix after Reducing Dataset')
plt.show()

# Load training subsets
train_one = pd.read_csv('train_subset_one.csv')
print(train_one.head())

# Display frequency chart for output variable
plt.figure(figsize=(10, 6))
plt.hist(train_one['bstatus'], bins=2, alpha=0.7, color='blue', edgecolor='black')
plt.title('Bankruptcy Status Frequency in Train Subset One')
plt.xlabel('Bankruptcy Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non Bankrupt', 'Bankrupt'])
plt.show()

# Prepare the data for logistic regression
y, X = dmatrices('bstatus ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
                  train_one, return_type="dataframe")
print(X.columns)

# Flatten y into a 1-D array
y = np.ravel(y)

# Instantiate a logistic regression model and fit with X and y
model = LogisticRegression()
model.fit(X, y)

# Check the accuracy on the training set
print(f'Logistic Regression Training Accuracy: {model.score(X, y)}')

# Predict class labels for the data
predicted = model.predict(X)
print(predicted)

# Generate class probabilities
probs = model.predict_proba(X)
print(probs)

# Generate evaluation metrics
print(metrics.accuracy_score(y, predicted))
print(metrics.roc_auc_score(y, probs[:, 1]))

# Confusion matrix
cm = metrics.confusion_matrix(y, predicted)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non Bankrupt', 'Bankrupt'])
plt.yticks(tick_marks, ['Non Bankrupt', 'Bankrupt'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Check Variance Inflation Factor (VIF)
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print('VIF:', vif)

# Decision Tree Classifier
train_two = pd.read_csv('train_subset_two.csv')
print(train_two.head())

y2, X2 = dmatrices('bstatus ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
                   train_two, return_type="dataframe")
print(X2.columns)
y2 = np.ravel(y2)

# Building a decision tree on train data
classifier = DecisionTreeClassifier()
classifier.fit(X2, y2)
print(f'Decision Tree Training Accuracy: {classifier.score(X2, y2)}')

# Predict class labels for the data
predicted_tree = classifier.predict(X2)
print(predicted_tree)

# Generate evaluation metrics
print(metrics.accuracy_score(y2, predicted_tree))
print(metrics.confusion_matrix(y2, predicted_tree))
print(metrics.classification_report(y2, predicted_tree))
cm_tree = metrics.confusion_matrix(y2, predicted_tree)

# Plot confusion matrix for Decision Tree
plt.figure(figsize=(8, 6))
plt.imshow(cm_tree, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Decision Tree')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non Bankrupt', 'Bankrupt'])
plt.yticks(tick_marks, ['Non Bankrupt', 'Bankrupt'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Check VIF for Decision Tree
vif_tree = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
print('VIF for Decision Tree:', vif_tree)

# Neural Network Classifier
mlp = MLPClassifier(hidden_layer_sizes=(12, 12, 12))
mlp.fit(X2, y2)

# Predictions
predict_MLP = mlp.predict(X2)
print('MLP Predictions:', predict_MLP)
print(f'MLP Training Accuracy: {metrics.accuracy_score(y2, predict_MLP)}')
