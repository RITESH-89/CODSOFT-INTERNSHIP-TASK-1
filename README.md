# 💼 CODSOFT Internship – Task 1

## 🚢 Titanic Survival Prediction using Machine Learning

This is my submission for **Task 1** of the **CODSOFT Data Science Internship**.  
The goal of this project is to build a supervised machine learning model that predicts whether a Titanic passenger survived or not based on features like age, sex, class, fare, and more.

---

## 📂 Dataset Information

The dataset used is based on the famous [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) dataset.

### 🎯 Features used for prediction:
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age in years
- `SibSp`: # of siblings / spouses aboard
- `Parch`: # of parents / children aboard
- `Fare`: Passenger fare
- `Embarked`: Port of Embarkation

---

## 🛠️ Tools & Libraries Used

- Python (Spyder via Anaconda)
- [Pandas](https://pandas.pydata.org/) – for data handling
- [Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/) – for visualization
- [Scikit-learn](https://scikit-learn.org/) – for machine learning model
- Logistic Regression – for classification

---

## 🔍 Steps Followed

1. **Loaded** the dataset using `pandas`
2. **Cleaned** the data (handled missing values, dropped irrelevant columns)
3. **Encoded** categorical variables (`Sex`, `Embarked`)
4. **Split** the data into training and testing sets (80-20)
5. **Trained** a Logistic Regression model
6. **Predicted** survival on test data
7. **Evaluated** performance using accuracy score, confusion matrix, and classification report

---

## 📊 Model Evaluation

- **Accuracy Achieved**: ~81%
- **Metrics Used**:
  - Accuracy Score
  - Precision, Recall, F1-score
  - Confusion Matrix (Visualized using Seaborn)

---

## 📈 Visualization Example

![Confusion Matrix](https://github.com/yourusername/yourrepo/blob/main/confusion_matrix_example.png)  
*(Replace the URL above with your actual image path if you upload the confusion matrix)*

---

## 📌 Conclusion

This project helped me understand the full ML pipeline including:
- Real-world data preprocessing
- Feature engineering
- Classification modeling
- Model evaluation & improvement

---

## 🧑‍💻 Author

**Ritesh Paithankar**  
_Data Science Intern at CODSOFT_  
📧 riteshpaithankar00@gmail.com

---

## 📎 Related Tags
`#CODSOFT` `#TitanicML` `#LogisticRegression` `#DataScienceInternship` `#MachineLearning` `#PythonProjects`
