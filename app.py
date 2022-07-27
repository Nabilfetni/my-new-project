import streamlit as st
import pandas as pd
import numpy as np
import pickle
  
st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

file1 = open('diabetes_prediction.pkl', 'rb')
rf = pickle.load(file1)
file1.close()


data = pd.read_csv("diabete_population.csv")

#pour la normalisation################################
moy_age=data['age'].mean()
std_age=data['age'].std()
moy_grossesses=data['grossesses'].mean()
std_grossesses=data['grossesses'].std()
moy_insuline=data['insuline'].mean()
std_insuline=data['insuline'].std()
#########

print(data)
  
#age = st.number_input("Enter your age") 
age = st.slider("Enter your age",data['age'].min(), data['age'].max())
age= (age-moy_age)/std_age #########

grossesses = st.number_input("Enter your grossesses")
grossesses = (grossesses-moy_grossesses)/std_grossesses ############## 

insuline= st.number_input("Enter your insuline") 
insuline =(insuline-moy_insuline)/std_insuline ############

#moy=data['grossesses', 'age', 'insuline'].mean()
#sig=data['grossesses', 'age', 'insuline'].std()
#tn=(data['grossesses', 'age', 'insuline']-moy)/sig

if(st.button('Predict Diabete')): 
    query = np.array([grossesses, age, insuline])

    query = query.reshape(1, 3)
    print(query)
    prediction = rf.predict(query)[0]
    st.title("Predicted value " +
             str(prediction)) 


