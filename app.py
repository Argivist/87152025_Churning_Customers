#importing modules
import pandas as pd
import numpy as np 
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

#Loading all models and dependencies
##########Dependencies##############
with open('dependencies.pkl','rb') as f:
    dep=pickle.load(f)
########Unpacking the pickle#########
sc=dep['Scaler']
inputs=dep['input']
##########Loading  Models###########
bestmodel=load_model('CustomerChurn_best.h5')
finalmodel=load_model('CustomerChurn_final.h5')

#prediction function
def pred(input,model):
    input_expanded = np.expand_dims(input, axis=0)
    if model==0:
        pred=bestmodel.predict(input_expanded)
    else:
        pred=finalmodel.predict(input_expanded)
    if(pred[0][0]>0.5):
        return "This customer will churn",pred
    else:
        return "This customer will not churn",pred

    


#Application
def main():
    html_temp="""<div style="backgroung-color:cyan; padding:10px">
    <h2 style="color:grey;text-align:center;">Customer Churn</h2>
    </div>
    <style>
    .stCheckbox{
        background-color:tomato;
        border-radius:10px;
        box-shadow:none;
        border:none;
        width:50px;
        height:30px;
    }
    .stCheckbox:checked{
        background-color:green;
    }
    </style>
    """
    unsafe_allow_html=True
    entries=[]
    #heading
    st.markdown(html_temp,unsafe_allow_html=True)
    #inputs
    
    tenure=st.number_input("Tenure",step=10)
    #scaling the tenure
    scaling=pd.DataFrame([[tenure,0,0]],columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    scaled=sc.transform(scaling)
    tenure=scaled[0][0]
    
    InternetService_Fiber_optic=st.checkbox('Uses Fiber optice',key="checkbox")
    os=st.checkbox('Has Online Security',key="checkbox1")
    if os:
        OnlineSecurity_No=False
    else:
        OnlineSecurity_No=True    
    ts=st.checkbox('Has Tech Support',key="checkbox2")
    if ts:
        TechSupport_No=False
    else:
        TechSupport_No=True  
    Contract_Month_to_month=st.checkbox('Has Fiber optic',key="checkbox3")
    Contract_Two_year=st.checkbox('On a 2 year contract',key="checkbox4")
    PaymentMethod_Electronic_check=st.checkbox('Uses Electronic check',key="checkbox5")
    entries=[InternetService_Fiber_optic,OnlineSecurity_No,TechSupport_No,Contract_Month_to_month,Contract_Two_year,PaymentMethod_Electronic_check]
    entry=[]
    entry.append(tenure)
    for i in entries:
        if i: entry.append(1) 
        else: entry.append(0)
    
    ####predicting output
    if(st.button("Predict")):
        pr=pred(entry,1)[0]
        st.write(pr)
        print(pred(entry,1)[1])
    
    
if __name__=='__main__':
    main()
    