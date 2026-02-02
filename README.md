# Q_Skill 

This repository contains multiple **Machine Learning projects** implemented using **Python** and **Jupyter Notebooks**.  
Each project demonstrates core ML concepts such as data preprocessing, model training, evaluation, and prediction.

---

## Repository Structure

Q_Skill/
│
├── Spam Mail Detector.ipynb

├── iris_dataset.ipynb

├── House Price Prediction/

│ └── (House Price Prediction notebook & related files)

└── README.md

---

## Projects Overview

### Spam Mail Detector 

A **machine learning–based spam email classification system** that classifies emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP).

#### Features
- Text preprocessing (cleaning, tokenization)
- Feature extraction using **TF-IDF / Count Vectorizer**
- Machine learning classification
- Model evaluation using accuracy and confusion matrix

#### Algorithms Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

#### Dataset
The dataset is **large**, so it is **not included in this repository**.

**Download from Kaggle**:  
https://www.kaggle.com/datasets

(Use a spam email dataset such as *Spam Email Dataset*)

➡ After downloading:
1. Extract the CSV file  
2. Place it in the same directory as `Spam Mail Detector.ipynb`  
3. Update the file path inside the notebook if required

---

### House Price Prediction 🏠

A **regression-based ML project** that predicts **house prices** based on multiple input features.

#### Features
- Data cleaning and preprocessing
- Handling missing values
- Feature selection
- Train-test split
- Model training and evaluation

#### Algorithms Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

#### Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

### Iris Dataset Classification 

A beginner-friendly **machine learning classification project** using the classic **Iris dataset**.

#### Features
- Dataset loading and exploration
- Data visualization
- Model training
- Prediction and evaluation

#### Algorithms Used
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier

This dataset is small and usually loaded directly from **scikit-learn**, so no external download is required.

---

## Technologies Used

- Python 
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- NLP libraries (for spam detection)

---

## How to Run the Projects

1. Clone the repository:
   ```bash
   git clone https://github.com/jay51211/Q_Skill.git
