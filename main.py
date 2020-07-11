# from flask import Flask, render_template, request
# from bokeh.models import ColumnDataSource
# import pickle
# import numpy as np
# app=Flask(__name__)
#
# files=open('model.pkl','rb')
# classifier=pickle.load(files)
# files.close()
#
#
# @app.route('/', methods= ["GET","POST"])
# def hello_world():
#     if request.method == "POST":
#         myDict = request.form
#         Gender = int(myDict['Gender'])
#         Married = int(myDict['Married'])
#         #Dependents = int(myDict['Dependents'])
#         Education = int(myDict['Education'])
#         Self_Employed = int(myDict['Self_Employed'])
#         ApplicantIncome = int(myDict['ApplicantIncome'])
#         CoapplicantIncome = int(myDict['CoapplicantIncome'])
#         LoanAmount = int(myDict['LoanAmount'])
#         Loan_Amount_Term = int(myDict['Loan_Amount_Term'])
#         Credit_History = int(myDict['Credit_History'])
#         Property_Area = int(myDict['Property_Area'])
#         inputFeatures = [Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
#         files = open('model.pkl', 'rb')
#         classifier = pickle.load(files)
#         files.close()
#         model = pickle.load(open('model.pkl', 'rb'))
#         infProb = classifier.predict_proba(inputFeatures)
#         print(infProb)
#         #classifier = model.predict_proba(dataset)
#         predictions = [item for sublist in infProb for item in sublist]
#         # colors = ['#1f77b4','#ff7f0e']
#         loan_status = ['Yes', 'No']
#         source = ColumnDataSource(
#             data=dict(loan_status=loan_status, predictions=predictions))
#         return render_template('result.html', predictions=predictions)
#
#     return render_template('index.html')
#     #return 'Hello World' + str(infProb)
#
# if __name__=='__main__':
#     app.run(debug=True)
#
#
#

#importing libraries
# import numpy as np
# import flask
# import pickle
# from flask import Flask, render_template, request
#
# # creating instance of the class
# app = Flask(__name__)
#
#
# #to tell flask what url should trigger the function index()
# @app.route('/')
# @app.route('/index')
# def index():
#     return flask.render_template('index.html')
#     # return "Hello World"
#
#
# # prediction function
# def ValuePredict(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1, 9)
#     loaded_model = pickle.load(open("model.pkl", "rb"))
#     result = loaded_model.predict(to_predict)
#     return result[0]
#
#
# @app.route('/', methods=['GET','POST'])
# def result():
#     if request.method == 'POST':
#         to_predict_list = request.form.to_dict()
#         to_predict_list = list(to_predict_list.values())
#         to_predict_list = list(map(int, to_predict_list))
#         result = ValuePredict(to_predict_list)
#
#         if int(result) == 1:
#             prediction = 'Loan yes'
#         else:
#             prediction = 'No Loan'
#
#         return render_template("result.html", prediction=prediction)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)












from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.embed import components

app=Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Getting the data from the form
    gender = request.form['Gender']
    married = request.form['Married']
    dependents= request.form['Dependents']
    education = request.form['Education']
    self_employed=request.form['Self_Employed']
    applicantincome=request.form['ApplicantIncome']
    coapplicantincome=request.form['CoapplicantIncome']
    loanamount=request.form['LoanAmount']
    loan_amount_term=request.form['Loan_Amount_Term']
    credit_history=request.form['Credit_History']
    property_area=request.form['Property_Area']

    #  creating a json object to hold the data from the form
    input_data=[{
    'gender':gender,
    'married':married,
    'dependents': dependents,
    'education':education,
    'self_employed':self_employed,
    'applicantincome':applicantincome,
    'coapplicantincome':coapplicantincome,
    'loanamount':loanamount,
    'loan_amount_term':loan_amount_term,
    'credit_history':credit_history,
    'property_area':property_area,
    }]


    dataset=pd.DataFrame(input_data)

    dataset=dataset.rename(columns={
                'gender':'Gender',
                'married': 'Married',
                'dependents' : 'Dependents',
                'education':'Education',
                'self_employed':'Self_Employed',
                'applicantincome':'ApplicantIncome',
                'coapplicantincome':'CoapplicantIncome',
                'loanamount':'LoanAmount',
                'loan_amount_term':'Loan_Amount_Term',
                'credit_history':'Credit_History',
                'property_area':'Property_Area',
                })

    dataset[['Credit_History','Loan_Amount_Term','LoanAmount','CoapplicantIncome'
             ]] = dataset[['Credit_History','Loan_Amount_Term','LoanAmount','CoapplicantIncome']].astype(float)

    dataset[['Gender','Married','Dependents','Education', 'Self_Employed','Property_Area']]=dataset[['Gender','Married', 'Dependents', 'Education', 'Self_Employed','Property_Area']].astype('object')

    dataset = dataset[['Gender','Married', 'Dependents', 'Education','Self_Employed',
    'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
    model = pickle.load(open('model.pkl', 'rb'))
    classifier=model.predict_proba(dataset)
    predictions = [item for sublist in classifier for item in sublist]
    # colors = ['#1f77b4','#ff7f0e']
    loan_status = ['Y','N']
    source = ColumnDataSource(
        data=dict(loan_status=loan_status, predictions=predictions))

    # p = figure(x_range=loan_status, plot_height=500,
    #            toolbar_location=None, title="Loan Status", plot_width=800)
    # p.vbar(x='loan_status', top='predictions', width=0.4, source=source, legend="loan_status",
    #        line_color='black', fill_color=factor_cmap('loan_status', palette=colors, factors=loan_status))
    #
    # p.xgrid.grid_line_color = None
    # p.y_range.start = 0.1
    # p.y_range.end = 0.9
    # p.legend.orientation = "horizontal"
    # p.legend.location = "top_center"
    # p.xaxis.axis_label = 'Loan Status'
    # p.yaxis.axis_label = ' Predicted Probabilities'
    # script, div = components(p)
    return render_template('show.html', predictions=predictions)




if __name__=="__main__":
    app.run(debug=False)
