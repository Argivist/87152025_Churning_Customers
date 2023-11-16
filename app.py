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
    return (float)(pred[0][0])

     


#Application
def main():
    html_temp="""<div class="header">
    <h2 style="color:grey;text-align:center;">Customer Churn</h2>
    </div>
    <style>
    .stCheckbox{
        border-radius:10px;
        box-shadow:none;
        border:none;
        width:50px;
        height:30px;
    }
    .header{
        background:url('https://www.stockvault.net/data/2021/08/05/287431/preview16.jpg') #B59410;
        color:#B59410;
        backdrop-filter: blur(5px); 
        padding:10px;
        border-radius:10px;
        margin-bottom:15px;
    }

    </style>
    """
    unsafe_allow_html=True
    entries=[]
    #heading
    st.markdown(html_temp,unsafe_allow_html=True)
    #inputs
    
    tenure=st.sidebar.number_input("Tenure",step=10)

    
    
    
    InternetService_Fiber_optic=st.sidebar.checkbox('Uses Fiber optics',key="checkbox")
    os=st.sidebar.checkbox('Has Online Security',key="checkbox1")
    if os:
        OnlineSecurity_No=False
    else:
        OnlineSecurity_No=True    
    ts=st.sidebar.checkbox('Has Tech Support',key="checkbox2")
    if ts:
        TechSupport_No=False
    else:
        TechSupport_No=True  
    Contract=st.sidebar.selectbox("Contract type",["Month_to_Month","Two_years"])
    Contract_Month_to_month=(Contract=="Month_to_Month")
    Contract_Two_year=(Contract=="Two_years")
    Payment=st.sidebar.selectbox("Payment type",["Electronic Check","other"])
    PaymentMethod_Electronic_check=(Payment=="Electronic Check")
    
    ####predicting output
    if(st.sidebar.button("Predict")):
            #scaling the tenure
        scaling=pd.DataFrame([[tenure,0,0]],columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
        scaled=sc.transform(scaling)
        tenure=scaled[0][0]
        #
        entries=[InternetService_Fiber_optic,OnlineSecurity_No,TechSupport_No,Contract_Month_to_month,Contract_Two_year,PaymentMethod_Electronic_check]
        entry=[]
        for i in entries:
            if i: entry.append(1) 
            else: entry.append(0)
        entry.append(tenure)
        entries=[]
        pr=pred(entry,0)
        if pr>0.5:        
            st.success("This customer will churn")
        else:        
            st.error("This customer will not churn")
        st.info('''Model Statistics:
                   \nPrecision: 0.6104417443275452 
                   \nRecall: 0.5352112650871277 
                   \nAccuracy: 0.7829383611679077''')
    
    
if __name__=='__main__':
    main()
    