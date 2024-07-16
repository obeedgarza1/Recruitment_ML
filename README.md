# Hiring Decision Analysis

## Project Overview

This project aims to analyze and predict hiring decisions using a logistic regression model. We explore various factors that might influence the hiring decision, including age, gender, education level, years of experience, previous companies, distance from the company, interview score, skill score, personality score, and recruitment strategy.

## Data Description

The dataset consists of the following columns:

- **Age**: The age of the candidate.
- **Gender**: The gender of the candidate (1 for female, 0 for male).
- **EducationLevel**: The highest education level attained by the candidate.
- **ExperienceYears**: The number of years of relevant work experience the candidate has.
- **PreviousCompanies**: The number of companies the candidate has worked for previously.
- **DistanceFromCompany**: The distance (in kilometers) from the candidate's residence to the company.
- **InterviewScore**: The score obtained by the candidate in the interview.
- **SkillScore**: The score reflecting the candidate's technical skills.
- **PersonalityScore**: The score reflecting the candidate's personality traits.
- **RecruitmentStrategy**: The strategy used to recruit the candidate (1, 2, 3; higher numbers indicate less effective strategies).
- **HiringDecision**: The final hiring decision (1 for hired, 0 for not hired).

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
