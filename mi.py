from numpy.core.numeric import NaN
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from bs4 import BeautifulSoup as bs # type: ignore
import requests
from selenium import webdriver # type: ignore
from selenium.webdriver.chrome.options import Options # type: ignore
import time
import json
import base64
import math
import random
from scipy.stats import norm  # type: ignore
from scipy.stats import bernoulli  # type: ignore
from scipy.stats import uniform # type: ignore   
import math

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Demo Data Analysis App')

#---------------------------------#
st.write("""
# Multiple Imputation Analyze Binary Distribution response App
""")

a = []
a2 = []
for trial in range(1, 101):
    for trt in range(0, 2):
        for subj in range(1, 61): 

            # 	Covariates 
            gender = bernoulli.rvs(p=0.5) 
            race = math.ceil(uniform.rvs(loc=0)*5)
            score = norm.rvs(2,1) 								    # a continuous baseline score
            if score > 3.5 : score_c = 2 
            else : score_c = 1		                            	# a baseline score category (1=low, 2=high)

            # Adverse event 	 
            miss_ae = bernoulli.rvs(p=0.2) 							# miss_ae is missingness indicator for subjects
            if miss_ae==1 : mm_ae=random.randint(1,6)         	    # mm_ae is missingness indicator for months - adverse event
            elif miss_ae==0 : mm_ae=NaN

            # Intercurrent event 	 
            miss_ie = bernoulli.rvs(p=0.1)							# miss_ie is missingness indicator for subjects
            if miss_ie==1 : mm_ie=random.randint(1,6)      	        # mm_ie is missingness indicator for months - intercurrent event
            elif miss_ie==0 : mm_ie=NaN        

            for vis in range(7): 
                if trt == 0 : resp = bernoulli.rvs(p=0.8)            # placebo values
                else : resp = bernoulli.rvs(p=0.9)                   # active values

                # Lack of Efficacy			
                if score < 0  : miss_ef = 1	
                if vis == 0 : miss_ef = NaN	

                if (vis >= mm_ae and mm_ae != NaN) or (vis >= mm_ie and mm_ie != NaN) or miss_ef == 1 : resp_ = NaN  
                else : resp_ = resp
 
                a.append([trial,trt,subj,gender,race,score,score_c,vis,mm_ae,mm_ie,miss_ef,resp])
                a2.append([trial,trt,subj,gender,race,score,score_c,vis,mm_ae,mm_ie,miss_ef,resp_])



df = pd.DataFrame(a, columns=["trial", "trt", "subj", "gender", "race", "score", "score_c", "vis","mm_ae", "mm_ie", "miss_ef", "resp"])
df2 = pd.DataFrame(a2, columns=["trial", "trt", "subj", "gender", "race", "score", "score_c", "vis","mm_ae", "mm_ie", "miss_ef", "resp"])

st.write("""
# 
Full Data
""")
st.write(df)

st.write("""
# 
Data With Missing value
""")

st.write(df2)





#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Input your parameter')



#---------------------------------#
# Plot Function
def boxplot(x, y, group):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(y, 'by', x)
    # sns.boxplot(x="Randomization", y="SurgeryTime", palette="husl", data=df)
    ax = sns.boxplot(x=x, y=y, hue=group, palette="husl", data=df)
    plt.setp(ax.get_xticklabels(), rotation=30)
    st.pyplot()

def pointplot(x, y, group):
    st.write(y, 'by', x)
    ax = sns.pointplot(x=x, y=y, hue=group, err_style="bars", ci=95, data=df, dodge=0.4, join=True)
    plt.setp(ax.get_xticklabels(), rotation=30)
    st.pyplot()


#---------------------------------#
# Download Function
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SurgeryAnalysis.csv">Download CSV File</a>'
    return href

#---------------------------------#
# Main panel

    


