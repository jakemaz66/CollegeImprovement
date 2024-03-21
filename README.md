# College Quality Regression Project

## Overview
This project focuses on estimating the overall quality of colleges based on several key factors such as salary, average debt, and likelihood of employment for graduates. The goal is to develop predictive models that can accurately predict the quality score of a college using controllable variables like faculty salary, admission rates, and programs offered.

## Data Sources
The data used in this project was collected from multiple sources, including:
- College Scorevard Data from collegescorecard.ed.gov

## Models Developed
Four different regression models were developed and compared for their predictive accuracy:
1. XGBoost Regression
2. Random Forest Regression
3. Linear Regression
4. Support Vector Machine (SVM) Regression

## Model Features
### Dependent Variable (Target):
- Quality Score: A composite score representing the overall quality of a college, derived from salary, average debt, and employment likelihood. This score is weighed and standardized, with more emphasis on the salary and debt compared to the employment.

### Independent Variables (Features):
- Faculty Salary: Average salary of faculty members at the college.
- Admission Rates: Percentage of applicants admitted to the college.
- Programs Offered: Number of academic programs offered by the college.
- Undergraduate Enrollment: Total amount of students
- Expenditures: Total instructional expenditures per student
- Tuition: Total Tuition of University

## Model Training and Evaluation
1. Data Preprocessing:
   - Missing value imputation
   - Feature scaling

2. Model Training:
   - Splitting data into training and testing sets 
   - Training each regression model on the training data

3. Model Evaluation:
   - Calculating metrics such as R-squared, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to evaluate model performance.
   - Comparing performance across different models to select the best-performing one.
   - The XGBoost model had the lowest error

## Results and Analysis
After training and evaluating the models, the following insights were gained:
- XGBoost and Random Forest models outperformed Linear Regression and SVM in terms of predictive accuracy.
- Programs offered and instructional expenditures per student had the most significant impact on the quality score, followed by admission rates and tution
- High faculty salaries and a diverse range of academic programs positively influenced the perceived quality of colleges.

## Conclusion
Based on the results, it can be concluded that factors such as faculty salary, admission rates, and programs offered play crucial roles in determining the overall quality of colleges. The predictive models developed in this project can be used to assess and compare the quality of colleges based on these factors, providing valuable insights for stakeholders in the education sector.

For detailed code implementation and analysis, please refer to the Jupyter Notebook or Python script provided in the repository.
