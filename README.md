# Machine Learning Laboratory Repository

---
***Created By:***  
Sanjay.S  
CH.SC.U4CSE23241  
CSE - C *(2023 - 2027)*  
---

### Activity 1: Linear Regression

**Aim:**  
To learn to train a simple Linear Regression Model using Python.

**Activities Done:**  
- **Experience to Salary Prediction Model:**  
  Predicts salary based on years of experience using linear regression.  
  Dataset: [Kaggle Salary Data](https://www.kaggle.com/datasets/amineoumous/salary-data)

- **Chicago Taxi Trip Fare Prediction Model:**  
  Predicts taxi fare amounts from trip data using linear regression.  
  Dataset: [Kaggle Chicago Taxi Trips](https://www.kaggle.com/datasets/adelanseur/taxi-trips-chicago-2024)

### Activity 2: Logistic Regression

**Aim:**  
To understand and implement Logistic Regression models for binary and multiclass classification problems using Python.

**Activities Done:**  

- **Admission Predictor Model:**  
  Predicts graduate school admission based on GRE score, GPA, and undergraduate institution prestige using binary logistic regression.  
  Dataset: [UCLA Binary Logistic Regression Dataset](https://stats.idre.ucla.edu/stat/stata/dae/binary.dta)

- **Purchase Prediction Model:**  
  Predicts the category of product a customer is likely to purchase based on demographic and transactional features using multinomial logistic regression.  
  Dataset: [Kaggle - Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

### Activity 3: Principal Component Analysis (PCA)  

**Aim:**  
To understand and apply Principal Component Analysis for dimensionality reduction and identifying the most important variables influencing the data.  

**Activities Done:**  

- **PCA on Breast Cancer Dataset:**  
  Applied Principal Component Analysis to the Breast Cancer dataset to reduce dimensionality and visualize class separation.  
  Steps included data preprocessing, applying PCA, plotting explained variance, and visualizing the transformed data to highlight how PCA can help distinguish between malignant and benign cases.  

- **Real Estate Insights Model:**  
  Used PCA to analyze why some properties are not selling for more than six months.  
  Steps included preprocessing (handling missing values, scaling), applying PCA, and visualizing components.  
  Dataset: [Kaggle – House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)  

- **Variable Importance Identification:**  
  Performed PCA on a sample dataset with variables like Climate, Housing, Health, Crime, etc.  
  Found the dominant variables for each principal component:  
  - **PC1 → Arts**  
  - **PC2 → Health**  
  - **PC3 → Crime**  

### Activity 4: Decision Tree Classifier  

**Aim:**  
To apply Decision Tree Classification for predicting outcomes and visualizing the decision-making process.  

**Activities Done:**
  
- **Diabetes Classification:**  
  Predicted whether a person has **diabetes** based on medical features such as **Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age**.  
  Steps: preprocessing, train-test split, model training, tree visualization, accuracy & confusion matrix.  
  **Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  

- **House Purchase Prediction:**  
  Predicted whether a person buys a house based on **Age** and **Income**.  
  Steps: preprocessing, train-test split, model training, tree visualization, accuracy & confusion matrix.  
  **Dataset:** [Social Network Ads](https://www.kaggle.com/datasets/rakeshrau/social-network-ads)  

- **Free Time Activity Based on Weather:**  
  Predicted whether a person **goes out** or **stays in** based on **Weather** and **Activity**.  
  Steps: categorical encoding, train-test split, model training, tree visualization, confusion matrix.  
  **Dataset:** [Simple Weather Forecast](https://www.kaggle.com/datasets/dheemanthbhat/simple-weather-forecast)  

### Activity 5: Support Vector Machines

**Aim:**  
To apply Support Vector Machine (SVM) classification for identifying patterns in datasets such as spam emails or quadratic relationships, and to evaluate the models using metrics like accuracy and confusion matrix.

**Activities Done:**
- **Diabetes Classification:**  
  Predicted whether a person has **diabetes** based on medical features such as **Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age**.  
  Steps: preprocessing, train-test split, model training, tree visualization, accuracy & confusion matrix.  
  **Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) 

- **Email Spam Classification:**  
  Classified whether an email is **spam or not spam** based on features like **word frequencies (e.g., ‘the’, ‘to’, ‘and’, etc.)** and **message structure attributes**.  
  Steps: data exploration, feature-target separation, train-test split, model training with linear kernel, prediction, accuracy evaluation, and confusion matrix visualization.  
  **Dataset:** [Email Spam Classification Dataset CSV](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

- **Quadratic Distribution Classification:**  
  Classified points as belonging to one of two classes based on a **quadratic boundary** defined by formulas like **x₁² + x₂² < threshold**.  
  Steps: generating random data points, labeling based on quadratic condition, train-test split, model training with polynomial kernel, prediction, accuracy evaluation, and decision boundary visualization.  
  **Dataset:** Synthetic dataset generated using Python (features: **x₁, x₂**, labels based on quadratic relationship).






