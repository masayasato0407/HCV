#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import pickle


# In[ ]:


rsf = pickle.load(open("rfmodel.sav", 'rb'))


# In[ ]:


st.title('Prediction model for post-SVR HCC (SMART model)') 


# In[ ]:


st.markdown("Enter the following items to display the predicted HCC risk")


# In[ ]:


with st.form('user_inputs'): 
  gender = st.selectbox('gender', options=['female', 'male']) 
  age=st.number_input('age (year)', min_value=0) 
  BMI=st.number_input('Body mass index', min_value=10.0) 
  alc60 = st.selectbox('Daily alcoholic consumption', options=['Less than 60g', '60g or more']) 
  DM= st.selectbox('Diabetes', options=['absent', 'present']) 
  PLT=st.number_input('Platelet count (×10^4/µL)', min_value=0.0)
  AFP=st.number_input('AFP (ng/mL)', min_value=0.0) 
  ALB=st.number_input('Albumin (g/dL)', min_value=0.0) 
  AST=st.number_input('AST (IU/L)', min_value=0)
  ALT=st.number_input('ALT (IU/L)', min_value=0)
  GGT=st.number_input('GGT (IU/L)', min_value=0)
  TBil=st.number_input('Total bilirubin (mg/dL)', min_value=0.0)
  st.form_submit_button() 


# In[ ]:


if gender == 'male': 
  gender = 1

elif gender == 'female':
  gender = 0 


# In[ ]:


if alc60 == '60g or more': 
  alc60 = 1 

elif alc60 == 'Less than 60g': 
  alc60 = 0 


# In[ ]:


if DM == 'present': 
  DM = 1 

elif DM == 'absent': 
  DM = 0 


# In[ ]:


surv = rsf.predict_survival_function(pd.DataFrame(
    data={'gender': [gender], 
          'age': [age],
          'BMI': [BMI],
          'alc60': [alc60],
          'PLT': [PLT],
          'AFP': [AFP],
          'ALB': [ALB],
          'TBil': [TBil],
          'AST': [AST],
          'ALT': [ALT],
          'GGT': [GGT],
          'DM': [DM]
         }
), return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.xlim(0,10)
plt.ylim(0,1)
plt.ylabel("predicted HCC development")
plt.xlabel("years")
plt.grid(True)

plt.gca().invert_yaxis()

plt.yticks([0.0, 0.2, 0.4,0.6,0.8,1.0],
            ['100%', '80%', '60%', '40%', '20%', '0%'])
plt.savefig("img.png")


# In[ ]:


st.subheader("HCC risk for submitted patient")


# In[ ]:


st.image ("img.png")

