# Amazon Alexa Review Analysis and Classification
Overview
This project aims to analyze and classify Amazon Alexa reviews into positive and negative sentiments. The code involves data preprocessing, feature extraction using TF-IDF vectorization, balancing the dataset using SMOTE, scaling the data, training multiple classification models, and performing hyperparameter tuning.

https://github.com/user-attachments/assets/633a73a9-b1d3-4d88-be8a-936df617ca07

## Data Processing
**Loading Data:**

The dataset is loaded from a TSV file using Pandas.
Missing values are checked and removed to ensure data integrity.

**Feature Engineering:**

A new feature length is created to represent the length of each review, aiding in understanding the data distribution.

**Data Visualization:**

The distribution of review lengths is plotted to visualize the data characteristics.

**Text Preprocessing:**

Reviews are cleaned by removing non-alphabetical characters, converting to lowercase, and splitting into words.
Words are stemmed to their root form, and common stopwords are removed to reduce noise.

**TF-IDF Vectorization:**

The cleaned text data is converted into TF-IDF features, which represent the importance of words in the reviews relative to the entire dataset.
The TF-IDF vectorizer is saved for future use.

**Data Splitting:**

The dataset is split into training and testing sets, ensuring that the class distribution is preserved.

**Balancing the Dataset:**

SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training set to address class imbalance by generating synthetic samples for the minority class.

**Scaling the Data:**

The data is scaled using MinMaxScaler to ensure that all features are on a similar scale, which is important for many machine learning algorithms.
The scaler is saved for future use.

## Data Modeling
**Model Training and Evaluation:**

A function is defined to train models, make predictions, and evaluate performance using confusion matrix and accuracy score.

**Ensemble of Classifiers:**

Three classifiers (Random Forest, Gradient Boosting, and XGBoost) are combined into a Voting Classifier to improve prediction performance by leveraging the strengths of multiple models.
The ensemble model's accuracy is evaluated and displayed.

**Hyperparameter Tuning:**

GridSearchCV is used to perform hyperparameter tuning on the XGBoost model. This exhaustive search over specified parameter values helps in finding the best combination of hyperparameters for optimal performance.
The best parameter combination and cross-validation score are printed.

## Final Model Evaluation:

The best model from the hyperparameter tuning is evaluated on the test set to measure its final accuracy.
The final model is saved for future predictions.

**Why These Steps?**

**Data Cleaning and Preprocessing:** Ensures that the text data is in a uniform format, reducing noise and improving the quality of the features extracted.

**TF-IDF Vectorization:** Converts text data into numerical features, capturing the importance of words in the context of the entire dataset, which is essential for machine learning models.

**SMOTE:** Addresses class imbalance, a common issue in real-world datasets, by generating synthetic samples, leading to better model performance.

**Scaling:** Ensures that all features contribute equally to the model, which is especially important for distance-based algorithms.

**Ensemble Learning:** Combines multiple models to improve robustness and accuracy by leveraging the strengths of each model.

**Hyperparameter Tuning:** Optimizes model performance by finding the best set of hyperparameters, which can significantly impact the effectiveness of the model.

**Model Saving:** Facilitates future use of the trained models and transformers, ensuring consistency and reproducibility in predictions.
