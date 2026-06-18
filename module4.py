import streamlit as st
import module3 as mod3
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
import traceback
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from xgboost import XGBRegressor,XGBClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import precision_score,recall_score,f1_score
st.title("ML Workbench")
st.header("ML Workbench - Train, Evaluate, Save, and Predict with Tabular Data")
st.subheader("Remember if you switched to my model page your content in this page will be reset so finish work on one page before switching")
def everything(X,feature,file_r,target):
    if feature is not None:
        X = X[feature]
        Y = file_r[target]
        if "field" not in st.session_state:
            st.session_state["field"] = Y.unique().tolist()
        if "value" not in st.session_state:
            st.session_state["value"] = Y.nunique()
        size = st.sidebar.slider("Select test_size",min_value=0.1,max_value=0.8,value=0.8)
        model = st.selectbox("Select Model",[
        DecisionTreeRegressor(random_state=0),RandomForestRegressor(random_state=0),LinearRegression(),XGBRegressor(random_state = 0),
        LogisticRegression(random_state = 0),DecisionTreeClassifier(random_state = 0),RandomForestClassifier(random_state=0),XGBClassifier(random_state = 0)])
        preprocess = mod3.split(X,Y,size) 
        final = mod3.train(preprocess.one_hot_col,preprocess.ord_col,preprocess.imputable_c,preprocess.imputable_n,model,preprocess.X_train,preprocess.Y_train,preprocess.X_valid)
        return preprocess.X_valid,preprocess.Y_valid,final.prediction,final.Pipe,preprocess.X_train.columns

loc = st.sidebar.radio("Choose file Location",["local storage","url"])
t = st.selectbox("Select type of file",["csv","xlsx","json"])
try:
    if loc == "local storage":
        file = st.file_uploader("Upload File from local storage")
        if "file_r" not in st.session_state and file is not None:
            st.session_state["file_r"] = mod3.LoadFile(file,t)
        elif file is not None:
            st.session_state["file_r"] = mod3.LoadFile(file,t)
    elif loc == "url":
        file = st.text_input("Enter url of file")
        try:
            if "file_r" not in st.session_state and file != "":
                st.session_state["file_r"] = mod3.LoadFile(file,t)
            elif file != "":
                st.session_state["file_r"] = mod3.LoadFile(file,t)
        except Exception as e:
            st.write("Wait")
    if st.session_state["file_r"] is not None:
        target = st.selectbox("Select target",[i for i in st.session_state["file_r"].columns])
        file_r = st.session_state["file_r"].dropna(subset=[target])
        X = st.session_state["file_r"].drop(target,axis = 1)
        feature = st.multiselect("Select Features",[i for i in X.columns])
        operation = st.radio("Select operation",["Regression","Classification"])
except Exception as e:
    st.write("All operations are on hold until proper file input")
try:
    X_valid,Y_valid,prediction,pipe,col = everything(X,feature,st.session_state["file_r"],target)
    show = st.button("Show portion of predictions")
    if show:    
        see = mod3.show_portion(X_valid,Y_valid,prediction)
    if operation == "Regression":
        loss = mod3.validation_r(prediction,Y_valid)
    if operation == "Classification":
        if st.session_state["value"] > 2:
            a = st.sidebar.selectbox("Select Average paramter",["micro","weighted","macro"])
            st.write(f"precision_score: {precision_score(prediction,Y_valid,average=a)}")
            st.write(f"Recall_score: {recall_score(prediction,Y_valid,average=a)}")
            st.write(f"f1: {f1_score(prediction,Y_valid,average=a)}")
        elif st.session_state["value"] < 2:
            a = st.sidebar.selectbox("Select pros_lable paramter",[st.session_state["field"]])
            st.write(f"precision_score: {precision_score(prediction,Y_valid,pos_label=a)}")
            st.write(f"Recall_score: {recall_score(prediction,Y_valid,pos_label = a)}")
            st.write(f"f1: {f1_score(prediction,Y_valid,pos_label=a)}")
            st.write("⚠️ Note: Confusion Matrix may take huge space so if you find unconvience in analysing page content then do not open it ")
        show = st.button("See confusion Matrix")
        if show:
            fig,ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                    Y_valid,
                    prediction,
                    ax=ax
                        )
            st.pyplot(fig)
except Exception as e:
    st.write("Always check values and models  you face any issue ")

name = st.text_input("Enter file name if you want to save model: ")
st.write("⚠️Only save model if you are satisfied with it and name the file carefully ")

if name != "":
    try:
        mod3.save(pipe,col,st.session_state["file_r"],name,target)
    except Exception as e:
        st.write("Iy you facing erros or wrong outputs retry be removing any unncessasry feautre and check all inputs you gave")
