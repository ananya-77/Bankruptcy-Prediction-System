# Bankruptcy-Prediction-System
A machine learning project that predicts bankruptcy risk for companies using financial data from 2000 to 2005, achieving an accuracy of 77.51% with neural networks.

# Predicting Bankruptcy Using Machine Learning Algorithms

## Introduction
The goal of this project is to predict the likelihood of bankruptcy for companies across various industries using machine learning techniques. We utilized financial data collected from 1980 to 2001, analyzing different financial ratios to assess bankruptcy risk.

## Data Collection and Preparation

### Data Sources
Data was gathered from various text files and loaded onto the Big Data platform, Hortonworks Ambari. The data was consolidated using HiveQL for efficient processing.

### Data Preprocessing
Initial preprocessing was conducted using Pig Script, where the data was cleaned and structured. The dataset was subsequently split into training and testing sets, which were exported as CSV files for further analysis.

### Exploratory Data Analysis (EDA)
Using Python, we conducted an exploratory data analysis to uncover patterns and insights in the data. This involved:
- **Summary Statistics**: Understanding the distribution and central tendencies of financial ratios.
- **Missing Values**: Identifying and addressing any gaps in the data.
- **Outliers**: Detecting outliers and determining appropriate treatment strategies.
- **Skewness Transformation**: Transforming skewed data to reduce bias in modeling.

## Data Manipulation
To ensure data quality and improve model accuracy, we performed several manipulations:
- **Handling Missing Values**: Missing entries were addressed through imputation or removal.
- **Outlier Treatment**: Outliers were either capped or removed based on their impact on the model.
- **Data Normalization**: Financial ratios were normalized to facilitate comparisons across different scales.

## Machine Learning Models
We applied multiple machine learning algorithms to build classification models for bankruptcy prediction:
1. **Logistic Regression**: A statistical model that uses a logistic function to model binary dependent variables.
2. **Decision Trees**: A model that uses a tree-like graph of decisions to make predictions.
3. **Support Vector Machines (SVM)**: A supervised learning model that analyzes data for classification.
4. **Neural Networks**: A set of algorithms modeled loosely after the human brain that are designed to recognize patterns.

### Model Evaluation
The models were evaluated based on accuracy and misclassification rates. The neural network model performed best, achieving:
- **Accuracy**: 78.21%
- **Misclassification Rate**: 0.2178

## Results

### Confusion Matrix
A confusion matrix was generated to visualize the performance of the final model (Neural Networks), allowing us to see the true positive, true negative, false positive, and false negative rates.

### ROC Curve
The Receiver Operating Characteristic (ROC) curve was plotted to evaluate the model's ability to distinguish between classes.

## Conclusion
The project successfully demonstrated the use of machine learning algorithms in predicting bankruptcy risks for companies. Through careful data preparation and analysis, we identified a neural network as the most effective model for this task. With an accuracy of 78.21%, the model serves as a valuable tool for stakeholders in assessing financial health and making informed decisions.

## Future Work
Further improvements can be made by:
- Incorporating additional features, such as macroeconomic indicators.
- Exploring ensemble methods to enhance predictive performance.
- Conducting more extensive hyperparameter tuning for each model.

## References
- Financial Data (1980-2001) from Company Records
- Machine Learning Frameworks and Libraries (Scikit-learn, TensorFlow)

