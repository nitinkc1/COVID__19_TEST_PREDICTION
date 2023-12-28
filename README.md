
# 💻 Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![MySQL](https://img.shields.io/badge/mysql-%2300000f.svg?style=for-the-badge&logo=mysql&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

# COVID-19 Prediction Project Overview

Welcome to the COVID-19 Prediction project repository! This project focuses on leveraging machine learning techniques to predict COVID-19 outcomes based on symptoms and demographic factors. Below is a comprehensive overview of the project, its objectives, and the methodologies employed.

## Section 1: Questions to Answer

### Key Questions

1. **Importance in Today's World:**
   - Why is accurately predicting COVID-19 crucial in today's world, and how can it improve medical treatment?

2. **Impact on Healthcare:**
   - How will accurate prediction impact effective screening and reduce the burden on healthcare systems?

3. **Applicability to Other Diseases:**
   - Are there knowledge gaps, and how might the proposed method be helpful for future diseases?

## Section 2: Initial Hypothesis

### Assumptions and Hypotheses

We hypothesize that specific symptoms and demographic features play a vital role in predicting COVID-19 outcomes, based on initial data analysis.

## Section 3: Data Analysis Approach

### Approach to Prove or Disprove Hypotheses

1. **Feature Engineering Techniques:**
   - Utilize relevant feature engineering tasks to enhance model performance.

2. **Data Visualization Techniques:**
   - Employ exploratory data analysis (EDA) techniques to identify patterns in the data.

3. **Characteristics of Important Features:**
   - Report characteristics of important features, including the total number and percentage of each category.

## Section 4: Machine Learning Approach

### Predictive Modeling

1. **Dataset Information:**
   - Utilize the "corona_tested_006" dataset from the 'ABC' government website.

2. **Model Training and Validation:**
   - Train and validate the model using data from 11th March 2020 to 15th April 2020, with a 4:1 ratio for training and validation.

3. **Test Set:**
   - Evaluate the model on data from 16th April 2020 to 30th April 2020.

4. **Feature Engineering:**
   - Perform appropriate feature engineering tasks on the dataset.

5. **Data Visualization:**
   - Use data visualization techniques to identify patterns in the data.

6. **Machine Learning Models:**
   - Implement multiple machine learning models relevant to the hypothesis.

7. **Cost Functions:**
   - Employ important cost functions to justify the superiority of the chosen model.

8. **Comparison of Models:**
   - Compare at least four models to determine the most effective for COVID-19 prediction.

## Machine Learning-Based Prediction of COVID-19 Diagnosis Based on Symptoms

We aim to provide a rapid and accurate diagnosis of COVID-19, crucial for effective SARS-CoV-2 screening and reducing the burden on healthcare systems. The dataset includes information from 2,78,848 individuals who underwent the RT-PCR test, covering the period from 11th March 2020 to 30th April 2020.

### Dataset Features:

#### A. Basic Information:

1. **ID (Individual ID)**
2. **Sex (Male/Female)**
3. **Age ≥60 above years (True/False)**
4. **Test Date (Date when tested for COVID)**

#### B. Symptoms:

5. **Cough (True/False)**
6. **Fever (True/False)**
7. **Sore Throat (True/False)**
8. **Shortness of Breath (True/False)**
9. **Headache (True/False)**

#### C. Other Information:

10. **Known Contact with an Individual Confirmed to have COVID-19 (True/False)**

#### D. COVID Report:

11. **Corona Positive or Negative**

## Insights

# Univariate Analysis

![image](https://github.com/nitinkc1/Covid_19_test_prediction/assets/130339748/3b87bc30-5a42-4e32-96ea-bfec40696f52)

# Major symptoms which are True Based on their Gender

![image](https://github.com/nitinkc1/Covid_19_test_prediction/assets/130339748/a56870d4-98fe-4a76-b29f-fc3a9605cb78)

# Major symptom cause of the people who are corona positive

![image](https://github.com/nitinkc1/Covid_19_test_prediction/assets/130339748/89ef8ee2-5990-470e-aa4a-7509777fa959)

# Major symptoms which are True and how they got corona and through by which type of contact

![image](https://github.com/nitinkc1/Covid_19_test_prediction/assets/130339748/a3005c1d-6b34-4110-b1ed-8ac199a53fa8)

## Machine Learning Models

# The following machine learning models were employed for prediction:

- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- Support Vector Classifier (SVC)
- Evaluation Metrics

  Each model was evaluated using precision, recall, and f1-score, providing a comprehensive understanding of their performance. Additionally, the accuracy score was     
  computed to gauge the overall predictive capability.
  
## Results

# Here's a tabular representation of the above information:

```markdown
| Model                     | Precision | Recall | F1-Score | Accuracy Score |
|---------------------------|-----------|--------|----------|----------------|
| Random Forest Classifier  |   0.00    |  0.47  |   0.00   |     0.9528     |
| Logistic Regression       |   0.00    |  0.59  |   0.00   |     0.9435     |
| Decision Tree Classifier  |   0.00    |  0.47  |   0.00   |     0.9528     |
| Support Vector Classifier  |   0.00    |  0.53  |   0.00   |     0.9392     |
```

This table summarizes the precision, recall, f1-score, and accuracy score for each machine learning model used in the COVID-19 prediction.


## Acknowledgment

We extend our sincere appreciation and gratitude to all individuals and organizations who have contributed to the development and evaluation of the COVID-19 Prediction Project. This collaborative effort would not have been possible without the dedication and expertise of those involved.

## Summary

The proposed project aims to contribute to the understanding of COVID-19 prediction, paving the way for improved medical treatment and screening methodologies. Feel free to explore the code and datasets provided in this repository for a detailed understanding of the models and their evaluations. Your contributions and insights are welcome as we work together to improve COVID-19 prediction methodologies.
