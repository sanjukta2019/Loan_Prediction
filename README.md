# Loan_Prediction
Loan Prediction System using Machine Learning & Flask

This project is a Loan Approval Prediction Web Application built using Machine Learning (Logistic Regression) and Flask.
It predicts whether a loan will be approved or not based on applicant details such as income, education, credit history, and property area.

# Project Overview

Financial institutions receive numerous loan applications daily. Manually evaluating each application is time-consuming and error-prone.
This system automates the decision-making process using a trained Logistic Regression model, deployed as a Flask web app.

#Features

Data preprocessing and missing value handling

Categorical encoding for ML compatibility

Logistic Regression model training

Model serialization using pickle

User-friendly Flask web interface

Real-time loan approval prediction


#Tech Stack

Python

Pandas, NumPy

Scikit-learn

Flask

HTML (Jinja Templates)

Pickle

#Project Structure
Loan_Prediction/
│
├── templates/               # HTML templates
│   ├── index.html
│   ├── show.html
│   ├── heading.html
│   ├── heading1.html
│   └── heading2.html
│
├── train.csv                # Training dataset
├── test.csv                 # Testing dataset
├── LoanPrediction.ipynb     # Jupyter Notebook (EDA + Model)
├── scriptsolution.py        # Model training & serialization
├── script.py                # Flask application
├── loan.pkl                 # Trained ML model
├── model.pkl                # (Optional model file)
├── README.md                # Project documentation
└── venv/                    # Virtual environment


📊 Dataset Description

The dataset contains the following features:

Feature	Description
Gender	Male / Female
Married	Yes / No
Dependents	Number of dependents
Education	Graduate / Not Graduate
Self_Employed	Yes / No
ApplicantIncome	Applicant income
CoapplicantIncome	Co-applicant income
LoanAmount	Loan amount
Loan_Amount_Term	Loan repayment term
Credit_History	Credit history (0/1)
Property_Area	Urban / Semiurban / Rural
Loan_Status	Target variable (Y/N)


🧠 Machine Learning Workflow

Data Cleaning

Missing values filled using mode/median

Encoding

Categorical variables mapped to numeric values

Train-Test Split

80% Training, 20% Testing

Model

Logistic Regression

Model Saving

Serialized using pickle as loan.pkl

🌐 Flask Web Application

The Flask app:

Takes user inputs from an HTML form

Converts inputs to numerical format

Uses the trained model to predict loan approval

Displays the prediction result on the UI
_________________________
▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/sanjukta2019/Loan_Prediction.git
cd Loan_Prediction

2️⃣ Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install pandas numpy scikit-learn flask

4️⃣ Train the Model (Optional)
python scriptsolution.py

5️⃣ Run the Flask App
python script.py

6️⃣ Open Browser
http://127.0.0.1:5000/

📈 Prediction Output

1 → Loan Approved

0 → Loan Not Approved

📌 Sample Input Format
[Gender, Married, Education, Self_Employed,
 ApplicantIncome, CoapplicantIncome,
 LoanAmount, Loan_Amount_Term,
 Credit_History, Property_Area, Dependents]
