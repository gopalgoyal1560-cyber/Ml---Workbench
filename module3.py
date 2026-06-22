import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib 
import os
import io
def LoadFile(file,t):
    df = None
    if file is None:
        st.stop()
    elif t == "csv":
        df = pd.read_csv(file)
    elif t == "json":
        df = pd.read_json(file)
    elif t == "xlsx":
        df = pd.read_excel(file)
    if df is not None:
        try:
            st.write(df.head())
            return df
        except Exception as e:
                st.write("")
            
class split:
    def __init__(self,X,Y,size):
        self.flag = 0
        X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y,test_size = size,random_state = 0)
        self.one_hot_col = [i for i in X_train.columns if X_train[i].dtype == object and X_train[i].nunique() <= 15]
        self.ord_col =  [i for i in X_train.columns if X_train[i].dtype == object and X_train[i].nunique() > 15]
        self.imputable_c = [i for i in X_train.columns if X_train[i].dtype == object and X_train[i].isnull().any()]
        self.imputable_n = [i for i in X_train.columns if (X_train[i].dtype == int or X_train[i].dtype == float) and  X_train[i].isnull().any()] 
        self.X_train,self.X_valid = X_train,X_valid
        self.Y_train,self.Y_valid = Y_train,Y_valid    

class train:
    def __init__(self,one_hot_col,ord_col,imputable_c,imputable_n,model,X_train,Y_train,X_valid):
        Preprocess_imp = Pipeline(steps=[
            ('impute_num',SimpleImputer(strategy='median'))
        ])
        Preprocess_enc_one= Pipeline(steps = [
            ('impute_cat',SimpleImputer(strategy='most_frequent')),
            ('one_hot',OneHotEncoder(handle_unknown='infrequent_if_exist',sparse_output=False)),                                 
        ])
        Preprocess_enc_ord = Pipeline(steps = [
            ('impute_cat',SimpleImputer(strategy='most_frequent')),
            ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
        ])
        
        Preprocess = ColumnTransformer(transformers=[
            ('step1',Preprocess_imp,imputable_n),
            ('step2',Preprocess_enc_one,one_hot_col),
            ('step3',Preprocess_enc_ord,ord_col)
        ])
        self.Pipe = Pipeline(steps = [
            ('preprocess',Preprocess),
            ('model',model)
        ])
        self.Pipe.fit(X_train,Y_train)
        st.success("Training is completed")
        self.prediction = self.Pipe.predict(X_valid)
       

class show_portion:
    def __init__(self,X_valid,Y_valid,prediction):
        tab1,tab2,tab3 = st.tabs(["Features table","Target","Predictions"])
        with tab1:
                st.write(X_valid.head())
        with tab2:
                st.write(Y_valid.head())
        with tab3:
                st.write(prediction[0:5])

class validation_r:
    def __init__(self,y_pred,y_true):
        self.y_pred,self.y_true = y_pred,y_true
        type = st.sidebar.selectbox("Select loss type",["mae","mse","r2","All"])
        if type == "mae":
                loss = mean_absolute_error(y_true,y_pred)
        elif type == "mse":
                loss = mean_squared_error(y_true,y_pred)
        elif type == "r2":
                loss = r2_score(y_pred,y_true)
        elif type == "All":
                loss = ["r2_Score:",r2_score(y_pred,y_true), "MSE:",mean_squared_error(y_true,y_pred),"MAE:" ,mean_absolute_error(y_true,y_pred)]
        if loss is not None:
                if loss != 0:
                    st.header(f"😒 We got some loss : {loss}")
                else:
                    st.header(f"😁 Everything is all right ! loss : {loss}")
         
class save:
    def __init__(self,pipe,col,dataset,model,target):
        try:
            col_d = {}
            for i in col:
                if i != target:
                    col_d[i] = dataset[i].dtype
            artifact = {
                "model" : pipe,
                "col" : col_d
            }
            buffer = io.BytesIO()
            
            result = joblib.dump(artifact,buffer)
            buffer.seek(0)
            final = st.download_button(label = "Download model",data = buffer,file_name = f"{model}.pkl",mime = "application/octet-stream")
            st.write(result)
            if result:
                st.success("Model is downloaded")
                st.balloons()
                st.write(col_d)
        except Exception as e:
            st.write(str(e))
