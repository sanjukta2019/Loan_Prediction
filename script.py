from flask import Flask, render_template, request
import pickle
app=Flask(__name__)

file=open('loan.pkl','rb')
model=pickle.load(file)
file.close()

@app.route('/heading2')
def home2():
	return render_template('heading2.html')


@app.route('/heading1')
def home1():
	return render_template('heading1.html')

@app.route('/heading')
def home():
	return render_template('heading.html')

#'Gender','Married','Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents'
@app.route('/', methods= ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Gender = int(myDict['Gender'])
        Married = int(myDict['Married'])
        Education = int(myDict['Education'])
        Self_Employed = int(myDict['Self_Employed'])
        ApplicantIncome = int(myDict['ApplicantIncome'])
        CoapplicantIncome = int(myDict['CoapplicantIncome'])
        LoanAmount = int(myDict['LoanAmount'])
        Loan_Amount_Term = int(myDict['Loan_Amount_Term'])
        Credit_History = int(myDict['Credit_History'])
        Property_Area = int(myDict['Property_Area'])
        Dependents = int(myDict['Dependents'])
        inputFeatures = [Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Dependents]
        infProb = model.predict([inputFeatures])
        print(infProb)
        return render_template('show.html', inf= int(infProb))
    return render_template('index.html')
    #return 'Hello World' + str(infProb)

if __name__=='__main__':
    app.run(debug=True)