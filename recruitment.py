import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from wolta.data_tools import col_types
from wolta.data_tools import seek_null
from wolta.data_tools import unique_amounts
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import sklearn.metrics as metrics

## Variables Description
# Age:
# Data Range: 20 to 50 years. Data Type: Integer.

# Gender:
# Categories: Male (0) or Female (1). Data Type: Binary.

# Education Level:
# Categories: 1: Bachelor's (Type 1), 2: Bachelor's (Type 2), 3: Master's, 4: PhD. Data Type: Categorical.

# Experience Years:
# Data Range: 0 to 15 years. Data Type: Integer.

# Previous Companies Worked:
# Data Range: 1 to 5 companies. Data Type: Integer.

# Distance From Company:
# Data Range: 1 to 50 kilometers. Data Type: Float (continuous).

# Interview Score:
# Data Range: 0 to 100. Data Type: Integer.

#Skill Score: 
# Data Range: 0 to 100. Data Type: Integer.

# Personality Score:
# Data Range: 0 to 100. Data Type: Integer.

# Recruitment Strategy: 
# Categories: 1: Aggressive, 2: Moderate, 3: Conservative. Data Type: Categorical.

# Hiring Decision (Target Variable):
# Categories: 0: Not hired, 1: Hired. Data Type: Binary (Integer).

df = pd.read_csv('recruitment_data.csv')

df.describe()

types = col_types(df, print_columns=True)

nulls = seek_null(df, print_columns=True)

unique_amounts(df)

# Heatmap of the correlation (to the Target Column)
corr = df.corr()
target_column = 'HiringDecision'

column_corr = corr[[target_column]].sort_values(by=target_column, ascending=False)

plt.figure(figsize=(4, 8))
sns.heatmap(column_corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

plt.title(f'Correlation with {target_column}')
plt.show()

### Creating a df for visualization purposes
viz_df = df.copy()
bins = [20, 29, 39, 49, 59]
labels = ['20-29', '30-39', '40-49', '50-59']
viz_df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
viz_df['Gender']=df['Gender'].map({0:'Male',1:'Female'})
viz_df['HiringDecision']=df['HiringDecision'].map({0:'Not Hired',1:'Hired'})

bins = [0, 3, 8, float('inf')]
labels = ['Early Entry 0-3 YoE', 'Mid-Level 4-8 YoE', 'Senior 9+ YoE']
viz_df['ExperienceLevel'] = pd.cut(df['ExperienceYears'], bins=bins, labels=labels, right=False)


### Chart to observe the variation of skill scores trhought the ages and if they were hired or not.
plt.figure(figsize=(12, 8))
boxplot = sns.boxplot(data=viz_df, x='AgeGroup', y='SkillScore', hue='HiringDecision', palette='Set2')

boxplot.set_title('Skill Score Distribution by Age Group and Gender', fontsize=16, fontweight='bold')
boxplot.set_xlabel('Age Group', fontsize=14, fontweight='bold')
boxplot.set_ylabel('Skill Score', fontsize=14, fontweight='bold')
boxplot.legend(title='Hiring Decision', title_fontsize='13', fontsize='11')
boxplot.grid(True, linestyle='--', linewidth=0.7)

boxplot.tick_params(axis='x', labelsize=12)
boxplot.tick_params(axis='y', labelsize=12)
plt.show()

## Chart to observe candidates and their profiles related to years of experience.
plt.figure(figsize=(12, 8))
histogram = sns.histplot(data=viz_df, x='ExperienceLevel', hue='HiringDecision', multiple='dodge', shrink=0.8, palette='Set2', edgecolor='black')

histogram.set_title('Count of Individuals by Experience Level and Hiring Decision', fontsize=16, fontweight='bold')
histogram.set_xlabel('Experience Level', fontsize=14, fontweight='bold')
histogram.set_ylabel('Count', fontsize=14, fontweight='bold')

histogram.grid(True, linestyle='--', linewidth=0.7)

histogram.tick_params(axis='x', labelsize=12)
histogram.tick_params(axis='y', labelsize=12)
plt.show()


##Logistic Regression
X = df.drop(['HiringDecision'], axis=1)
y = df['HiringDecision']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
log_reg = LogisticRegression(solver='liblinear')

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")

predictions = best_estimator.predict(x_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Matrix to compare results vs prediction
conf_matrix = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Hired', 'Hired'], yticklabels=['Not Hired', 'Hired'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_score, recall_score

# Metrics
precision = precision_score(y_test, predictions)

print("Precision:", precision)

##Predicting with my features and A/B testing with my input in 2 years from now
km = df['DistanceFromCompany'].mean()
ints = df['InterviewScore'].mean()

columns = ['Age', 'Gender', 'EducationLevel', 'ExperienceYears', 'PreviousCompanies', 
           'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore', 'RecruitmentStrategy']

me_values = np.array([[24, 0, 1, 2, 1, km, ints, 65, 65, 2]])
future_me = np.array([[26, 0, 3, 4, 2, km, 65, 73, 73, 2]])

df_me = pd.DataFrame(data=me_values, columns=columns)
df_future = pd.DataFrame(data=future_me, columns=columns)

my_prediction = best_estimator.predict(df_me)
future_me_prediction = best_estimator.predict(df_future)

print(f"Current Hiring Prediction: {'You´re Hired' if my_prediction[0] == 1 else 'Maybe next time :p'}")
print(f"Future Hiring Prediction: {'You´re Hired' if future_me_prediction[0] == 1 else 'Maybe next time :p'}")


from sklearn.model_selection import GridSearchCV, ParameterGrid
feature_ranges = {
    'Age': range(20, 51),  # Age from 20 to 50
    'EducationLevel': [1, 2, 3, 4],  # Education levels
    'ExperienceYears': range(0, 16),  # Experience years from 0 to 15
    'InterviewScore': range(0, 100),  # Interview scores from 0 to 100
    'SkillScore': range(0, 101),  # Skill scores from 0 to 100
    'PersonalityScore': range(0, 100)  # Personality scores from 0 to 100
}

# Generate all combinations of feature values within the specified ranges
param_grid = list(ParameterGrid(feature_ranges))

# Define other features that do not need optimization
gender = 0  # Assuming female
previous_companies = 3  # Assuming worked at 2 previous companies
distance_from_company = 20  # Assuming 10.5 km from the company
recruitment_strategy = 2  # Assuming moderate recruitment strategy

# Find the minimum set of feature values to be hired
for params in param_grid:
    # Create a feature vector with the current combination of parameters
    candidate_info = np.array([[params['Age'], gender, params['EducationLevel'], params['ExperienceYears'],
                                previous_companies, distance_from_company, params['InterviewScore'],
                                params['SkillScore'], params['PersonalityScore'], recruitment_strategy]])
    
    # Predict the hiring decision
    prediction = best_estimator.predict(candidate_info)

    # Check if the candidate would be hired
    if prediction[0] == 1:
        print("Minimum values to be hired found:")
        print(f"Age: {params['Age']}")
        print(f"Education Level: {params['EducationLevel']}")
        print(f"Experience Years: {params['ExperienceYears']}")
        print(f"Interview Score: {params['InterviewScore']}")
        print(f"Skill Score: {params['SkillScore']}")
        print(f"Personality Score: {params['PersonalityScore']}")
        break









#Create a df to see the mean of the hired candidtates to compare the profile against me
hired = df.query('HiringDecision == 1')
hired = hired.drop(columns='HiringDecision')

mean_of_hired = hired.mean()
mean_array = mean_of_hired.to_numpy()
mean_df = pd.DataFrame(data=mean_of_hired.to_numpy().reshape(1, -1), columns=columns)

concat = [df_me,mean_df]
df = pd.concat(concat)
df['Candidate'] = ''
df = df.reset_index(drop=True)

df.at[0,'Candidate'] = 'Me'
df.at[1,'Candidate'] = 'Mean Profile'
df = df.round(2)

# Data (help with chatgpt since this chart is hard lol)
labels = ['Age', 'Education Level', 'Experience Years', 'Previous Companies',
          'Distance From Company', 'Interview Score', 'Skill Score', 'Personality Score', 'Recruitment Strategy']
me = [24.00, 1.00, 2.00, 1.00, 25.51, 50.56, 65.00, 65.00, 2.00]
mean_profile = [35.17, 2.49, 8.54, 3.09, 25.14, 56.80, 60.03, 56.79, 1.40]

# Number of variables we're plotting
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop" and append the start value to the end.
me += me[:1]
mean_profile += mean_profile[:1]
angles += angles[:1]

# Size of the figure
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw y-labels
ax.set_rscale('log')
plt.xticks(angles[:-1], labels)

# Plot data with specified colors
ax.plot(angles, me, linewidth=1, linestyle='solid', label='Me', color='#2a9d8f')  # Color for 'Me'
ax.fill(angles, me, '#2a9d8f', alpha=0.1)

ax.plot(angles, mean_profile, linewidth=1, linestyle='solid', label='Mean Profile', color='#ef233c')  # Color for 'Mean Profile'
ax.fill(angles, mean_profile, '#ef233c', alpha=0.1)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Title
plt.title('Comparison: Me vs Mean Profile', size=20, y=1.1)

plt.show()



