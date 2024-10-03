# demo(test)
# import numpy as np
# import pandas as pd
# import streamlit as st

# st.title("Iris")
# st.text("Iris")
# st.write("Iris")
# st.info("Iris")
# st.write("# Iris") #Markdown

# a = np.array([[1,2],[3,4],[5,6]])
# st.write(a)
# b = pd.DataFrame(a)
# st.write(b)
# st.table(a)
# st.table(b)

# cb_red = st.checkbox('RED')
# if cb_red:
#     st.info("RED")
# else:
#     st.info("NOTHING")
# cb_pink = st.checkbox('PINK')
# if cb_red:
#     st.info("PINK")
# else:
#     st.info("NOTHING")

# get_married = st.radio("married?", ["married", "couple", "single"], key="married")
# if get_married == "married":
#     st.info("Married")
# elif get_married == "single":
#     st.info("Single")
# else:
#     st.info("Dating")

# gender = st.radio("gender?", ["man", "woman", "other"], key="sex")
# if gender == "man":
#     st.info("man")
# elif gender == "woman":
#     st.info("woman")
# else:
#     st.info("other")

import streamlit as st
import joblib
X_std = joblib.load("./file/X_std_model.joblib")
knn = joblib.load("./file/knn_model.joblib")
lr = joblib.load("./file/lr_model.joblib")
svm = joblib.load("./file/svm_model.joblib")
rf = joblib.load("./file/rf_model.joblib")

# 頁面
st.title("鳶尾花品種預測")
# 左邊欄選單
clr = st.sidebar.selectbox("## Classifier Selection", ("SVM", "KNN", "Random Forest", "Logistic Regression"))
if clr == "SVM":
    model = svm
elif clr == "KNN":
    model = knn
elif clr == "Logistic Regression":
    model = lr
elif clr == "Random Forest":
    model = rf

# 特徵輸入
s1 = st.slider("花萼長度：", min_value=4.3, max_value=7.9, value=5.0)
s2 = st.slider("花萼寬度：", min_value=2.0, max_value=4.4, value=3.0)
s3 = st.slider("花瓣長度：", min_value=1.0, max_value=6.9, value=4.0)
s4 = st.slider("花瓣寬度：", min_value=0.1, max_value=2.5, value=1.0)
btn = st.button("鳶尾花品種預測")

# model prediction
name = ['setosa', 'versicolor', 'virginica']
if btn:
    y_pred = model.predict([[s1, s2, s3, s4]])
    st.write("## 品種：",name[y_pred[0]])