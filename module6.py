import streamlit as st
from joblib import load
from module3 import LoadFile
import pandas as pd
import numpy as np
import traceback
file = st.file_uploader("Select Model")
try:
    artifact = load(file)
    model = artifact["model"]
    col = artifact["col"]
    
except Exception as e:
    print("Currently nothing is selected")
datatype = st.sidebar.radio("Select dataset type",["None","File","Single Value"])

try:
    if datatype == "File":
        st.write("⚠️ In file loading multiple issues can happen like unicode error or unloadable url or corrunpt file or wrong file type or too large file to load so try to load different files in case you notice a failuer to check system")
        loc = st.radio("Select file location",["None","local storage","url"])
        t = st.selectbox("Select type of file",["csv","xlsx","json"])
        if loc == "local storage":
                file = st.file_uploader("Upload File from local storage")
                if "file_r" not in st.session_state and file is not None:
                    st.session_state["file_r"] = LoadFile(file,t)
                elif "file_r" in st.session_state and file is not None:
                        st.session_state["file_r"] = LoadFile(file,t)
        elif loc == "url":
                file = st.text_input("Enter url of file")
                try:
                    if "file_r" not in st.session_state and file != "":
                        st.session_state["file_r"] = LoadFile(file,t)
                    elif "file_r" in st.session_state and file != "":
                        st.session_state["file_r"] = LoadFile(file,t)
                except Exception as e:
                    st.write("Currently all operation at halt")
        if st.session_state["file_r"] is not None:
            pred = model.predict(st.session_state["file_r"])
            show = st.button("Show a portion")
            save = st.button("Save all predition")
            if show:
                tab1,tab2 = st.tabs(["Features","Predictions"])
                with tab1:
                        st.write(st.session_state["file_r"].head())
                with tab2:
                        st.write(pred[0:5])

            elif save:
                st.session_state["file_r"]["prediction"] = pred
                st.write(st.session_state["file_r"])   
                st.write("Move to cursor over the table you will see a download button on right corner or table click it and download the table") 
    elif datatype == "Single Value":
        try:
            with st.form("save_from"):
                d = {}
                for i in col:
                    if col[i] == np.dtype("int64") or col[i] == np.dtype("float64"):
                        d[i] = st.number_input(f"Enter value {i}: ")
                    elif col[i] == np.dtype("O"):
                        d[i] = st.text_input(f"Enter category {i}")
                if st.form_submit_button("Submit"):
                    st.write("Submitted")
                    data,data_r = [],[]
                    for i in d:
                        data_r.append(d[i])
                    data.append(data_r)
            df = pd.DataFrame(data,columns = list(d.keys()))
            pred = model.predict(df)
            tab1,tab2 = st.tabs(["Data frame","Prediction"])
            with tab1:
                st.write(df)
            with tab2:
                st.write(pred)
        except Exception as e:
            st.write("If any error happens or you see this message click submit button again and check values you entered")
except Exception as e:
     st.write("")