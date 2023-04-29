import joblib


def predict(data):
    lr = joblib.load('rentpred.sav')
    return lr.predict(data)
    rt = joblib.load('label_encoder.sav')
    return lr.predict(data)
    t = joblib.load('scaler.sav')
    return lr.predict(data)
from sklearn.compose import ColumnTransformer
import pandas as pd
import streamlit as st

# create a ColumnTransformer

X = pd.DataFrame({'location','status'})
ct = ColumnTransformer(transformers=['location','status'])
# transform the input data
X_transformed = ct.fit_transform(X)

# get the output feature names
feature_names_out = ct.get_feature_names_out()

# create a DataFrame with the transformed data and renamed columns
df_transformed = pd.DataFrame(X_transformed, columns=feature_names_out)

# display the transformed data in Streamlit
st.write(df_transformed)

