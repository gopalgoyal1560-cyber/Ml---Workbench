import streamlit as st
st.write("")
page_1 = st.Page("module4.py",title="Home")
page_2 = st.Page("module6.py",title = "My models")
pg = st.navigation([page_1,page_2])
pg.run()