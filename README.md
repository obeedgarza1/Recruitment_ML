# Hiring Decision Analysis

## Project Overview

This project aims to analyze and predict hiring decisions using a logistic regression model. We explore various factors that might influence the hiring decision, including age, gender, education level, years of experience, previous companies, distance from the company, interview score, skill score, personality score, and recruitment strategy.

## Data Description

The dataset consists of the following columns:

### Variables Description

- **Age**:
  - Data Range: 20 to 50 years
  - Data Type: Integer

- **Gender**:
  - Categories: Male (0) or Female (1)
  - Data Type: Binary

- **Education Level**:
  - Categories: 1: Bachelor's (Type 1), 2: Bachelor's (Type 2), 3: Master's, 4: PhD
  - Data Type: Categorical

- **Experience Years**:
  - Data Range: 0 to 15 years
  - Data Type: Integer

- **Previous Companies Worked**:
  - Data Range: 1 to 5 companies
  - Data Type: Integer

- **Distance From Company**:
  - Data Range: 1 to 50 kilometers
  - Data Type: Float (continuous)

- **Interview Score**:
  - Data Range: 0 to 100
  - Data Type: Integer

- **Skill Score**:
  - Data Range: 0 to 100
  - Data Type: Integer

- **Personality Score**:
  - Data Range: 0 to 100
  - Data Type: Integer

- **Recruitment Strategy**:
  - Categories: 1: Aggressive, 2: Moderate, 3: Conservative
  - Data Type: Categorical

- **Hiring Decision (Target Variable)**:
  - Categories: 0: Not hired, 1: Hired
  - Data Type: Binary (Integer)

## Exploratory Data Analysis (EDA)

### Correlation Analysis

We begin with a correlation analysis to identify relationships between the features and the target variable (Hiring Decision). This helps in understanding which factors are most influential in the hiring process.

### Age and Hiring Decision

Analyzing the distribution of ages and their impact on the hiring decision can reveal age-related trends and biases in the hiring process. Visualizations such as histograms and box plots are used to study the age groups.

### Experience Years and Hiring Decision

Similarly, we explore the relationship between the number of years of experience and the likelihood of being hired. Scatter plots and line graphs help visualize this relationship.

## Predictive Modeling

### Logistic Regression

We employ a logistic regression model to predict the hiring decision. The logistic regression model is suitable for binary classification problems like this one, where the target variable (Hiring Decision) has two possible outcomes: hired or not hired.

### Hyperparameter Tuning

To optimize the model, we use GridSearchCV to perform hyperparameter tuning. The parameters tuned include the regularization penalty (`'penalty'`: `'l1'`, `'l2'`) and the regularization strength (`'C'`: `[0.01, 0.1, 1, 10, 100]`).
