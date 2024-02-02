#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span><ul class="toc-item"><li><span><a href="#Data-Upload" data-toc-modified-id="Data-Upload-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Data Upload</a></span></li></ul></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Data-Transformation" data-toc-modified-id="Data-Transformation-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data Transformation</a></span><ul class="toc-item"><li><span><a href="#Age" data-toc-modified-id="Age-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Age</a></span></li><li><span><a href="#Attrition" data-toc-modified-id="Attrition-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Attrition</a></span></li><li><span><a href="#Business-Travel" data-toc-modified-id="Business-Travel-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Business Travel</a></span></li><li><span><a href="#DailyRate" data-toc-modified-id="DailyRate-2.1.4"><span class="toc-item-num">2.1.4&nbsp;&nbsp;</span>DailyRate</a></span></li><li><span><a href="#Company-Department" data-toc-modified-id="Company-Department-2.1.5"><span class="toc-item-num">2.1.5&nbsp;&nbsp;</span>Company Department</a></span></li><li><span><a href="#Distance-from-Home" data-toc-modified-id="Distance-from-Home-2.1.6"><span class="toc-item-num">2.1.6&nbsp;&nbsp;</span>Distance from Home</a></span></li><li><span><a href="#Level-of-Education" data-toc-modified-id="Level-of-Education-2.1.7"><span class="toc-item-num">2.1.7&nbsp;&nbsp;</span>Level of Education</a></span></li><li><span><a href="#Education-Field" data-toc-modified-id="Education-Field-2.1.8"><span class="toc-item-num">2.1.8&nbsp;&nbsp;</span>Education Field</a></span></li><li><span><a href="#Environment-Satisfaction" data-toc-modified-id="Environment-Satisfaction-2.1.9"><span class="toc-item-num">2.1.9&nbsp;&nbsp;</span>Environment Satisfaction</a></span></li><li><span><a href="#Gender" data-toc-modified-id="Gender-2.1.10"><span class="toc-item-num">2.1.10&nbsp;&nbsp;</span>Gender</a></span></li><li><span><a href="#Hourly-Rate" data-toc-modified-id="Hourly-Rate-2.1.11"><span class="toc-item-num">2.1.11&nbsp;&nbsp;</span>Hourly Rate</a></span></li><li><span><a href="#Job-Involvement" data-toc-modified-id="Job-Involvement-2.1.12"><span class="toc-item-num">2.1.12&nbsp;&nbsp;</span>Job Involvement</a></span></li><li><span><a href="#Job-Level" data-toc-modified-id="Job-Level-2.1.13"><span class="toc-item-num">2.1.13&nbsp;&nbsp;</span>Job Level</a></span></li><li><span><a href="#Job-Role" data-toc-modified-id="Job-Role-2.1.14"><span class="toc-item-num">2.1.14&nbsp;&nbsp;</span>Job Role</a></span></li><li><span><a href="#Job-Satisfaction" data-toc-modified-id="Job-Satisfaction-2.1.15"><span class="toc-item-num">2.1.15&nbsp;&nbsp;</span>Job Satisfaction</a></span></li><li><span><a href="#MaritalStatus" data-toc-modified-id="MaritalStatus-2.1.16"><span class="toc-item-num">2.1.16&nbsp;&nbsp;</span>MaritalStatus</a></span></li><li><span><a href="#MonthlyIncome" data-toc-modified-id="MonthlyIncome-2.1.17"><span class="toc-item-num">2.1.17&nbsp;&nbsp;</span>MonthlyIncome</a></span></li><li><span><a href="#MonthlyRate" data-toc-modified-id="MonthlyRate-2.1.18"><span class="toc-item-num">2.1.18&nbsp;&nbsp;</span>MonthlyRate</a></span></li><li><span><a href="#NumCompaniesWorked" data-toc-modified-id="NumCompaniesWorked-2.1.19"><span class="toc-item-num">2.1.19&nbsp;&nbsp;</span>NumCompaniesWorked</a></span></li><li><span><a href="#OverTime" data-toc-modified-id="OverTime-2.1.20"><span class="toc-item-num">2.1.20&nbsp;&nbsp;</span>OverTime</a></span></li><li><span><a href="#PercentSalaryHike" data-toc-modified-id="PercentSalaryHike-2.1.21"><span class="toc-item-num">2.1.21&nbsp;&nbsp;</span>PercentSalaryHike</a></span></li><li><span><a href="#PerformanceRating" data-toc-modified-id="PerformanceRating-2.1.22"><span class="toc-item-num">2.1.22&nbsp;&nbsp;</span>PerformanceRating</a></span></li><li><span><a href="#RelationshipSatisfaction" data-toc-modified-id="RelationshipSatisfaction-2.1.23"><span class="toc-item-num">2.1.23&nbsp;&nbsp;</span>RelationshipSatisfaction</a></span></li><li><span><a href="#StockOptionLevel" data-toc-modified-id="StockOptionLevel-2.1.24"><span class="toc-item-num">2.1.24&nbsp;&nbsp;</span>StockOptionLevel</a></span></li><li><span><a href="#TotalWorkingYears" data-toc-modified-id="TotalWorkingYears-2.1.25"><span class="toc-item-num">2.1.25&nbsp;&nbsp;</span>TotalWorkingYears</a></span></li><li><span><a href="#TrainingTimesLastYear" data-toc-modified-id="TrainingTimesLastYear-2.1.26"><span class="toc-item-num">2.1.26&nbsp;&nbsp;</span>TrainingTimesLastYear</a></span></li><li><span><a href="#WorkLifeBalance" data-toc-modified-id="WorkLifeBalance-2.1.27"><span class="toc-item-num">2.1.27&nbsp;&nbsp;</span>WorkLifeBalance</a></span></li><li><span><a href="#YearsAtCompany" data-toc-modified-id="YearsAtCompany-2.1.28"><span class="toc-item-num">2.1.28&nbsp;&nbsp;</span>YearsAtCompany</a></span></li><li><span><a href="#YearsInCurrentRole" data-toc-modified-id="YearsInCurrentRole-2.1.29"><span class="toc-item-num">2.1.29&nbsp;&nbsp;</span>YearsInCurrentRole</a></span></li><li><span><a href="#YearsSinceLastPromotion" data-toc-modified-id="YearsSinceLastPromotion-2.1.30"><span class="toc-item-num">2.1.30&nbsp;&nbsp;</span>YearsSinceLastPromotion</a></span></li><li><span><a href="#YearsWithCurrManager" data-toc-modified-id="YearsWithCurrManager-2.1.31"><span class="toc-item-num">2.1.31&nbsp;&nbsp;</span>YearsWithCurrManager</a></span></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-2.1.32"><span class="toc-item-num">2.1.32&nbsp;&nbsp;</span>Feature Engineering</a></span></li><li><span><a href="#Variable-Transformation" data-toc-modified-id="Variable-Transformation-2.1.33"><span class="toc-item-num">2.1.33&nbsp;&nbsp;</span>Variable Transformation</a></span></li></ul></li></ul></li><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Outliers-Analysis" data-toc-modified-id="Outliers-Analysis-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Outliers Analysis</a></span><ul class="toc-item"><li><span><a href="#Income-outliers-by-department" data-toc-modified-id="Income-outliers-by-department-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Income outliers by department</a></span></li><li><span><a href="#Income-outliers-by-JobRole" data-toc-modified-id="Income-outliers-by-JobRole-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Income outliers by JobRole</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Lower-outliers-analysis" data-toc-modified-id="Lower-outliers-analysis-4.2.0.1"><span class="toc-item-num">4.2.0.1&nbsp;&nbsp;</span>Lower outliers analysis</a></span></li></ul></li></ul></li><li><span><a href="#Income-outliers-by-EducationField-and-level" data-toc-modified-id="Income-outliers-by-EducationField-and-level-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Income outliers by EducationField and level</a></span></li></ul></li><li><span><a href="#Data-Transformation" data-toc-modified-id="Data-Transformation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data Transformation</a></span><ul class="toc-item"><li><span><a href="#Standardization" data-toc-modified-id="Standardization-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Standardization</a></span></li><li><span><a href="#Normalization" data-toc-modified-id="Normalization-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Normalization</a></span></li></ul></li><li><span><a href="#Data-Modelling---Clustering" data-toc-modified-id="Data-Modelling---Clustering-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Data Modelling - Clustering</a></span><ul class="toc-item"><li><span><a href="#Employee-Segmentation" data-toc-modified-id="Employee-Segmentation-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Employee Segmentation</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.1.2"><span class="toc-item-num">6.1.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.1.3"><span class="toc-item-num">6.1.3&nbsp;&nbsp;</span>K-Means</a></span></li><li><span><a href="#Improved-Approach" data-toc-modified-id="Improved-Approach-6.1.4"><span class="toc-item-num">6.1.4&nbsp;&nbsp;</span>Improved Approach</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.1.4.1"><span class="toc-item-num">6.1.4.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.1.4.2"><span class="toc-item-num">6.1.4.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.1.4.3"><span class="toc-item-num">6.1.4.3&nbsp;&nbsp;</span>K-Means</a></span></li></ul></li></ul></li><li><span><a href="#Job-Position-Segmentation" data-toc-modified-id="Job-Position-Segmentation-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Job Position Segmentation</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>K-Means</a></span></li><li><span><a href="#Improved-Approach" data-toc-modified-id="Improved-Approach-6.2.4"><span class="toc-item-num">6.2.4&nbsp;&nbsp;</span>Improved Approach</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.2.4.1"><span class="toc-item-num">6.2.4.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.2.4.2"><span class="toc-item-num">6.2.4.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.2.4.3"><span class="toc-item-num">6.2.4.3&nbsp;&nbsp;</span>K-Means</a></span></li></ul></li></ul></li><li><span><a href="#Historic-Segmentation" data-toc-modified-id="Historic-Segmentation-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Historic Segmentation</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.3.3"><span class="toc-item-num">6.3.3&nbsp;&nbsp;</span>K-Means</a></span></li><li><span><a href="#Improved-Approach" data-toc-modified-id="Improved-Approach-6.3.4"><span class="toc-item-num">6.3.4&nbsp;&nbsp;</span>Improved Approach</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.3.4.1"><span class="toc-item-num">6.3.4.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.3.4.2"><span class="toc-item-num">6.3.4.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.3.4.3"><span class="toc-item-num">6.3.4.3&nbsp;&nbsp;</span>K-Means</a></span></li></ul></li></ul></li><li><span><a href="#Ongoing-Segmentation" data-toc-modified-id="Ongoing-Segmentation-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Ongoing Segmentation</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.4.3"><span class="toc-item-num">6.4.3&nbsp;&nbsp;</span>K-Means</a></span></li><li><span><a href="#Improved-Approach" data-toc-modified-id="Improved-Approach-6.4.4"><span class="toc-item-num">6.4.4&nbsp;&nbsp;</span>Improved Approach</a></span><ul class="toc-item"><li><span><a href="#Correlation-Analysis" data-toc-modified-id="Correlation-Analysis-6.4.4.1"><span class="toc-item-num">6.4.4.1&nbsp;&nbsp;</span>Correlation Analysis</a></span></li><li><span><a href="#Hierarchical-Clustering" data-toc-modified-id="Hierarchical-Clustering-6.4.4.2"><span class="toc-item-num">6.4.4.2&nbsp;&nbsp;</span>Hierarchical Clustering</a></span></li><li><span><a href="#K-Means" data-toc-modified-id="K-Means-6.4.4.3"><span class="toc-item-num">6.4.4.3&nbsp;&nbsp;</span>K-Means</a></span></li></ul></li></ul></li><li><span><a href="#Concluding-Remarks---Clustering" data-toc-modified-id="Concluding-Remarks---Clustering-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Concluding Remarks - Clustering</a></span></li></ul></li><li><span><a href="#Model-Building" data-toc-modified-id="Model-Building-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Model Building</a></span><ul class="toc-item"><li><span><a href="#Recursive-Feature-Elimination" data-toc-modified-id="Recursive-Feature-Elimination-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Recursive Feature Elimination</a></span><ul class="toc-item"><li><span><a href="#One-Hot-Encoding-of-Categorical-Variables" data-toc-modified-id="One-Hot-Encoding-of-Categorical-Variables-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>One Hot Encoding of Categorical Variables</a></span></li><li><span><a href="#Removal-of-non-numeric-variables" data-toc-modified-id="Removal-of-non-numeric-variables-7.1.2"><span class="toc-item-num">7.1.2&nbsp;&nbsp;</span>Removal of non numeric variables</a></span></li><li><span><a href="#RFE-Function-and-Analysis" data-toc-modified-id="RFE-Function-and-Analysis-7.1.3"><span class="toc-item-num">7.1.3&nbsp;&nbsp;</span>RFE Function and Analysis</a></span></li><li><span><a href="#Data-as-is---data-before-it-is-normalized" data-toc-modified-id="Data-as-is---data-before-it-is-normalized-7.1.4"><span class="toc-item-num">7.1.4&nbsp;&nbsp;</span>Data <em>as is</em> - data before it is normalized</a></span></li><li><span><a href="#Data-Robust-Scaled" data-toc-modified-id="Data-Robust-Scaled-7.1.5"><span class="toc-item-num">7.1.5&nbsp;&nbsp;</span>Data Robust Scaled</a></span></li><li><span><a href="#Data-MinMax-Scaled" data-toc-modified-id="Data-MinMax-Scaled-7.1.6"><span class="toc-item-num">7.1.6&nbsp;&nbsp;</span>Data MinMax Scaled</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-7.1.7"><span class="toc-item-num">7.1.7&nbsp;&nbsp;</span>Summary</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-7.1.7.1"><span class="toc-item-num">7.1.7.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-7.1.7.2"><span class="toc-item-num">7.1.7.2&nbsp;&nbsp;</span>Random Forest</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-7.1.7.3"><span class="toc-item-num">7.1.7.3&nbsp;&nbsp;</span>Decision Tree</a></span></li></ul></li></ul></li><li><span><a href="#Variable-Selection" data-toc-modified-id="Variable-Selection-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Variable Selection</a></span></li><li><span><a href="#Model-Definition" data-toc-modified-id="Model-Definition-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Model Definition</a></span><ul class="toc-item"><li><span><a href="#Naive-Bayes:" data-toc-modified-id="Naive-Bayes:-7.3.1"><span class="toc-item-num">7.3.1&nbsp;&nbsp;</span>Naive Bayes:</a></span></li><li><span><a href="#Random-Forest-Classifier" data-toc-modified-id="Random-Forest-Classifier-7.3.2"><span class="toc-item-num">7.3.2&nbsp;&nbsp;</span>Random Forest Classifier</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-7.3.3"><span class="toc-item-num">7.3.3&nbsp;&nbsp;</span>Decision Tree</a></span></li><li><span><a href="#Neural-Networks" data-toc-modified-id="Neural-Networks-7.3.4"><span class="toc-item-num">7.3.4&nbsp;&nbsp;</span>Neural Networks</a></span></li></ul></li><li><span><a href="#Model-Evaluation" data-toc-modified-id="Model-Evaluation-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Model Evaluation</a></span></li><li><span><a href="#Fine-Tuning" data-toc-modified-id="Fine-Tuning-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>Fine Tuning</a></span><ul class="toc-item"><li><span><a href="#Gradient-Boosting" data-toc-modified-id="Gradient-Boosting-7.5.1"><span class="toc-item-num">7.5.1&nbsp;&nbsp;</span>Gradient Boosting</a></span></li><li><span><a href="#Neural-Networks" data-toc-modified-id="Neural-Networks-7.5.2"><span class="toc-item-num">7.5.2&nbsp;&nbsp;</span>Neural Networks</a></span></li><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-7.5.3"><span class="toc-item-num">7.5.3&nbsp;&nbsp;</span>Logistic Regression</a></span></li></ul></li></ul></li><li><span><a href="#Model:-1-Year-in-Advance" data-toc-modified-id="Model:-1-Year-in-Advance-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Model: 1 Year in Advance</a></span><ul class="toc-item"><li><span><a href="#Defining-the-New-Dataset" data-toc-modified-id="Defining-the-New-Dataset-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Defining the New Dataset</a></span></li><li><span><a href="#Correlation-Analysis---1-year-back-model" data-toc-modified-id="Correlation-Analysis---1-year-back-model-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Correlation Analysis - 1 year back model</a></span></li><li><span><a href="#Recursive-Feature-Elimination" data-toc-modified-id="Recursive-Feature-Elimination-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Recursive Feature Elimination</a></span></li><li><span><a href="#Variable-Selection" data-toc-modified-id="Variable-Selection-8.4"><span class="toc-item-num">8.4&nbsp;&nbsp;</span>Variable Selection</a></span></li><li><span><a href="#Model-Definition-and-Fine-Tunning" data-toc-modified-id="Model-Definition-and-Fine-Tunning-8.5"><span class="toc-item-num">8.5&nbsp;&nbsp;</span>Model Definition and Fine Tunning</a></span></li><li><span><a href="#Model-Evaluation-and-Selection" data-toc-modified-id="Model-Evaluation-and-Selection-8.6"><span class="toc-item-num">8.6&nbsp;&nbsp;</span>Model Evaluation and Selection</a></span><ul class="toc-item"><li><span><a href="#Final-Model-Selection" data-toc-modified-id="Final-Model-Selection-8.6.1"><span class="toc-item-num">8.6.1&nbsp;&nbsp;</span>Final Model Selection</a></span></li></ul></li></ul></li></ul></div>

# # Imports

# In[148]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
import datetime as dt
import itertools
import random

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot 

import plotly.express as px
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy import stats
import seaborn as sns

from numpy import unique
from numpy import where
import pylab as pl
import time


from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import spearmanr
from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Upload

# In[2]:


hr0 = pd.read_csv(r'HR_DS.csv')
hr0.head(5)


# # Exploratory Data Analysis

# In[3]:


# To describe dataset structure - nº of observations(rows) and features(columns)
hr0.shape


# In[4]:


# To verify uniqueness of records
hr0['EmployeeNumber'].duplicated().sum()


# In[5]:


hr0.set_index(keys = hr0['EmployeeNumber'], inplace = True)


# In[6]:


# To identify dataset features (columns)
hr0.columns


# In[7]:


# To check dtypes and missing values
hr0.info()


# In[8]:


# To describe the main statistical informartion about the numerical features

# The display.max_columns option controls the number of columns to be printed
pd.set_option('display.max_columns', None)

hr0.describe().T


# - `EmployeeCount` is always 1 (std = 0) - it will be deleted
# - `StandardHours` is always 80 (std = 0) - it will be deleted
#    
# No Missing values detected

# In[9]:


# To drop 'Overt18' feature from the dataset
hr0 = hr0.drop(['EmployeeCount', 'StandardHours'], axis = 1)

# To validate the features
hr0.shape


# In[10]:


# To describe the main statistical informartion about the categorical features
hr0.describe(include=['O']).T


# - `Over18` is always Y - it will be deleted
#   
# No Missing values detected

# In[11]:


# To drop 'Overt18' feature from the dataset
hr0 = hr0.drop(['Over18'], axis = 1)

# To validate the features
hr0.shape


# ## Data Transformation

# ### Age 

# #Nota: colocar o numero de colaboradores em cima de cada coluna

# In[12]:


# Group employees per Age
employee_TotalWorkingYears = hr0.groupby('Age')['EmployeeNumber'].count().reset_index()

plt.figure(figsize=(15,6))
ax = sns.barplot(x = 'Age', y = 'EmployeeNumber', data = employee_TotalWorkingYears)

ax.set_xlabel('Age')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Age')
sns.despine()
plt.show()


# In[13]:


x = hr0['Age'].values
plt.hist(x, bins = 15)
plt.show()


# ### Attrition

# In[14]:


#Describe the variable 'Attrition'
a = hr0['Attrition'].value_counts().sort_index()
b = hr0['Attrition'].value_counts(normalize=True) * 100
b = b.map('{:,.2f} %'.format).sort_index()
hr0_temp = pd.concat([a,b],axis = 1)
hr0_temp.columns = ['Nr of Employees', '%']
display(hr0_temp)

Attrition_Employee = hr0.groupby(['Attrition'])['Attrition'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
Attrition_Employee.plot(kind='pie', labels = ['No', 'Yes'],subplots=True, figsize=(5, 5),explode = (0.15 , 0),startangle = 90,autopct = '%1.f%%')
plt.title("Attrition")
plt.ylabel('')
#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# ### Business Travel

# In[15]:


#Describe the variable 'Business Travel'
c = hr0['BusinessTravel'].value_counts().sort_index()
d = hr0['BusinessTravel'].value_counts(normalize=True) * 100
d = d.map('{:,.2f} %'.format).sort_index()
hr0_temp1 = pd.concat([c,d],axis = 1)
hr0_temp1.columns = ['Nr of Employees', '%']
display(hr0_temp1)

BusinessTravel = hr0.groupby(['BusinessTravel'])['BusinessTravel'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
BusinessTravel.plot(kind='pie',subplots=True, figsize=(5, 5), autopct = '%1.f%%')
                    #explode = (0.15 , 0),startangle = 90,autopct = '%1.f%%')
plt.title("BusinessTravel")

#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# ### DailyRate

# In[16]:


x = hr0['DailyRate'].values
plt.hist(x, bins = 25)
plt.show()


# ### Company Department

# In[17]:


#Describe the variable 'Department'
a = hr0['Department'].value_counts().sort_index()
b = hr0['Department'].value_counts(normalize=True) * 100
b = b.map('{:,.1f} %'.format).sort_index()
hr0_temp2 = pd.concat([a,b],axis = 1)
hr0_temp2.columns = ['Nr of Employees', '%']
display(hr0_temp2)

Department = hr0.groupby(['Department'])['Department'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
Department.plot(kind='pie',subplots=True, figsize=(5, 5), autopct = '%1.1f%%')
                    #explode = (0.15 , 0),startangle = 90,autopct = '%1.1f%%')

plt.ylabel('')
plt.title("Department")

#tight_layout: adjusts subplot to fit in to the figure are
plt.show()


# ### Distance from Home

# In[18]:


x = hr0['DistanceFromHome'].values
plt.hist(x, bins = 20)
plt.show()


# NOTA - Criar ranges de distância???

# ### Level of Education

# In[19]:


#Describe the variable 'Level of Education'
e = hr0['Education'].value_counts().sort_index()
f = hr0['Education'].value_counts(normalize=True) * 100
f = f.map('{:,.2f} %'.format).sort_index()
hr0_temp3 = pd.concat([e,f],axis = 1)
hr0_temp3.columns = ['Nr of Employees', '%']
display(hr0_temp3)

# Group employees per level of Education
employee_educ = hr0.groupby('Education')['EmployeeNumber'].count().reset_index()

#Plot the distribituion of utilization by time
plt.figure(figsize = (5, 5))

ax = sns.barplot(x = 'Education', y = 'EmployeeNumber', data = employee_educ)
ax.set_xlabel('Education Level')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per Level of Education')
sns.despine()
plt.show()


# ### Education Field

# In[20]:


#Describe the variable 'Level of Education'
g = hr0['EducationField'].value_counts().sort_index()
h = hr0['EducationField'].value_counts(normalize=True) * 100
h = h.map('{:,.1f} %'.format).sort_index()
hr0_temp4 = pd.concat([g,h],axis = 1)
hr0_temp4.columns = ['Nr of Employees', '%']
display(hr0_temp4)

# Group employees per level of Education
employee_EducationField = hr0.groupby('EducationField')['EmployeeNumber'].count().reset_index()

#Plot the distribituion of utilization by time
plt.figure(figsize = (10, 5))

ax = sns.barplot(x = 'EducationField', y = 'EmployeeNumber', data = employee_EducationField)
ax.set_xlabel('EducationField')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per EducationField')
sns.despine()
plt.show()


# ### Environment Satisfaction

# In[21]:


#Describe the variable 'EnvironmentSatisfaction'
i = hr0['EnvironmentSatisfaction'].value_counts().sort_index()
j = hr0['EnvironmentSatisfaction'].value_counts(normalize=True) * 100
j = j.map('{:,.1f} %'.format).sort_index()
hr0_temp5 = pd.concat([i,j],axis = 1)
hr0_temp5.columns = ['Nr of Employees', '%']
display(hr0_temp5)


# Group employees per EnvironmentSatisfaction
employee_EnvironmentSatisfaction = hr0.groupby('EnvironmentSatisfaction')['EmployeeNumber'].count().reset_index()

#Plot the distribituion of utilization by time
plt.figure(figsize = (10, 5))

ax = sns.barplot(x = 'EnvironmentSatisfaction', y = 'EmployeeNumber', data = employee_EnvironmentSatisfaction)
ax.set_xlabel('EnvironmentSatisfaction')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per EnvironmentSatisfaction')
sns.despine()
plt.show()


# ### Gender

# In[22]:


#Describe the variable Gender
a = hr0['Gender'].value_counts().sort_index()
b = hr0['Gender'].value_counts(normalize=True) * 100
b = b.map('{:,.2f} %'.format).sort_index()
hr0_temp6 = pd.concat([a,b],axis = 1)
hr0_temp6.columns = ['Counts', '%']
display(hr0_temp6)

Gender_Employee = hr0.groupby(['Gender'])['Gender'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
Gender_Employee.plot(kind='pie',subplots=True, figsize=(5, 5),explode = (0.15 , 0),startangle = 90,autopct = '%1.f%%')
plt.ylabel('')
plt.title("Gender")

#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# ### Hourly Rate

# In[23]:


fig, axes= plt.subplots(figsize=(15,5))
sns.distplot(hr0['HourlyRate'], kde=False)
plt.show()


# ### Job Involvement

# In[24]:


#Describe the variable 'JobInvolvement'
k = hr0['JobInvolvement'].value_counts().sort_index()
l = hr0['JobInvolvement'].value_counts(normalize=True) * 100
l = l.map('{:,.1f} %'.format).sort_index()
hr0_temp7 = pd.concat([k,l],axis = 1)
hr0_temp7.columns = ['Nr of Employees', '%']
display(hr0_temp7)

# Group employees per JobInvolvement
employee_JobInvolvement = hr0.groupby('JobInvolvement')['EmployeeNumber'].count().reset_index()

#Plot the distribituion of utilization by time
plt.figure(figsize = (10, 5))

ax = sns.barplot(x = 'JobInvolvement', y = 'EmployeeNumber', data = employee_JobInvolvement)
ax.set_xlabel('JobInvolvement')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per JobInvolvement Level')
sns.despine()
plt.show()


# ### Job Level

# In[25]:


#Describe the variable 'JobLevel'
k = hr0['JobLevel'].value_counts().sort_index()
l = hr0['JobLevel'].value_counts(normalize=True) * 100
l = l.map('{:,.1f} %'.format).sort_index()
hr0_temp7 = pd.concat([k,l],axis = 1)
hr0_temp7.columns = ['Nr of Employees', '%']
display(hr0_temp7)

# Group employees per JobLevel
employee_JobLevel = hr0.groupby('JobLevel')['EmployeeNumber'].count().reset_index()

#Plot the distribituion of utilization by time
plt.figure(figsize = (10, 5))

ax = sns.barplot(x = 'JobLevel', y = 'EmployeeNumber', data = employee_JobLevel)
ax.set_xlabel('JobLevel')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per JobLevel Level')
sns.despine()
plt.show()


# ### Job Role

# In[26]:


#Describe the variable 'JobRole'
k = hr0['JobRole'].value_counts().sort_index()
l = hr0['JobRole'].value_counts(normalize=True) * 100
l = l.map('{:,.1f} %'.format).sort_index()
hr0_temp7 = pd.concat([k,l],axis = 1)
hr0_temp7.columns = ['Nr of Employees', '%']
display(hr0_temp7)

# Group employees per JobRole
employee_JobRole = hr0.groupby('JobRole')['EmployeeNumber'].count().reset_index()

#Plot the distribituion of utilization by time
plt.figure(figsize = (20, 5))

ax = sns.barplot(x = 'JobRole', y = 'EmployeeNumber', data = employee_JobRole)
ax.set_xlabel('JobRole')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per JobRole Level')
sns.despine()
plt.show()


# ### Job Satisfaction

# In[27]:


#Describe the variable 'Business Travel'
o = hr0['JobSatisfaction'].value_counts().sort_index()
p = hr0['JobSatisfaction'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp8 = pd.concat([o,p],axis = 1)
hr0_temp8.columns = ['Nr of Employees', '%']
display(hr0_temp8)

# Group employees per JobRole
employee_JobSatisfaction = hr0.groupby('JobSatisfaction')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'JobSatisfaction', y = 'EmployeeNumber', data = employee_JobSatisfaction)
ax.set_xlabel('JobSatisfaction')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('Distribution per JobSatisfaction Level')
sns.despine()
plt.show()


# ### MaritalStatus

# In[28]:


#Describe the variable 'Marital Status'
q = hr0['MaritalStatus'].value_counts().sort_index()
r = hr0['MaritalStatus'].value_counts(normalize=True) * 100
r = r.map('{:,.1f} %'.format).sort_index()
hr0_temp9 = pd.concat([q,r],axis = 1)
hr0_temp9.columns = ['Nr of Employees', '%']
display(hr0_temp9)

MaritalStatus = hr0.groupby(['MaritalStatus'])['MaritalStatus'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
MaritalStatus.plot(kind='pie',subplots=True, figsize=(5, 5), autopct = '%1.1f%%')
                    #explode = (0.15 , 0),startangle = 90,autopct = '%1.1f%%')

plt.ylabel('')
plt.title("MaritalStatus")

#tight_layout: adjusts subplot to fit in to the figure are
plt.show()


# ### MonthlyIncome

# In[29]:


x = hr0['MonthlyIncome'].values
plt.hist(x, bins = 20)
plt.show()


# ### MonthlyRate

# In[30]:


x = hr0['MonthlyRate'].values
plt.hist(x, bins = 20)
plt.show()


# ### NumCompaniesWorked

# In[31]:


#Describe the variable 'NumCompaniesWorked'
s = hr0['NumCompaniesWorked'].value_counts().sort_index()
t = hr0['NumCompaniesWorked'].value_counts(normalize=True) * 100
t = t.map('{:,.1f} %'.format).sort_index()
hr0_temp10 = pd.concat([s,t],axis = 1)
hr0_temp10.columns = ['Nr of Employees', '%']
display(hr0_temp10)

# Group employees per NumCompaniesWorked
employee_NumCompaniesWorked = hr0.groupby('NumCompaniesWorked')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'NumCompaniesWorked', y = 'EmployeeNumber', data = employee_NumCompaniesWorked)
ax.set_xlabel('NumCompaniesWorked')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('NumCompaniesWorked')
sns.despine()
plt.show()


# ### OverTime

# In[32]:


#Describe the variable 'OverTime'
u = hr0['OverTime'].value_counts().sort_index()
v = hr0['OverTime'].value_counts(normalize=True) * 100
v = v.map('{:,.1f} %'.format).sort_index()
hr0_temp11 = pd.concat([u,v],axis = 1)
hr0_temp11.columns = ['Nr of Employees', '%']
display(hr0_temp11)

OverTime_Employee = hr0.groupby(['OverTime'])['OverTime'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
OverTime_Employee.plot(kind='pie', labels = ['No', 'Yes'],subplots=True, figsize=(5, 5),explode = (0.15 , 0),startangle = 90,autopct = '%1.f%%')
plt.title("OverTime")
plt.ylabel('')
#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# ### PercentSalaryHike

# In[33]:


#Describe the variable 'PercentSalaryHike'
o = hr0['PercentSalaryHike'].value_counts().sort_index()
p = hr0['PercentSalaryHike'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp12 = pd.concat([o,p],axis = 1)
hr0_temp12.columns = ['Nr of Employees', '%']
display(hr0_temp12)

# Group employees per PercentSalaryHike
employee_PercentSalaryHike = hr0.groupby('PercentSalaryHike')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'PercentSalaryHike', y = 'EmployeeNumber', data = employee_PercentSalaryHike)
ax.set_xlabel('PercentSalaryHike')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('PercentSalaryHike')
sns.despine()
plt.show()


# In[34]:


#PercentSalaryHike in decade clusters
bins = [11, 15, 19, 23, 27]
labels = ['11-14', '15-18', '19-22', '23+']
hr0['PercentSalaryHike_Bracket']= pd.cut(hr0['PercentSalaryHike'], bins, labels = labels,include_lowest = True)

#Describe the variable 'PercentSalaryHike' by brackets
o = hr0['PercentSalaryHike_Bracket'].value_counts().sort_index()
p = hr0['PercentSalaryHike_Bracket'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp13 = pd.concat([o,p],axis = 1)
hr0_temp13.columns = ['Nr of Employees', '%']
display(hr0_temp13)

# Group employees per PercentSalaryHike by brackets
employee_PercentSalaryHike_Bracket = hr0.groupby('PercentSalaryHike_Bracket')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'PercentSalaryHike_Bracket', y = 'EmployeeNumber', data = employee_PercentSalaryHike_Bracket)
ax.set_xlabel('PercentSalaryHike_Bracket')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('PercentSalaryHike_Bracket')
sns.despine()
plt.show()


# ### PerformanceRating

# In[35]:


#Describe the variable 'PerformanceRating'
u = hr0['PerformanceRating'].value_counts().sort_index()
v = hr0['PerformanceRating'].value_counts(normalize=True) * 100
v = v.map('{:,.1f} %'.format).sort_index()
hr0_temp14 = pd.concat([u,v],axis = 1)
hr0_temp14.columns = ['Nr of Employees', '%']
display(hr0_temp14)

PerformanceRating_Employee = hr0.groupby(['PerformanceRating'])['PerformanceRating'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
OverTime_Employee.plot(kind='pie', subplots=True, figsize=(4, 3),explode = (0.15 , 0),startangle = 90,autopct = '%1.f%%')
plt.title("PerformanceRating")
plt.ylabel('')
#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# ### RelationshipSatisfaction 

# In[36]:


#Describe the variable 'RelationshipSatisfaction' 
o = hr0['RelationshipSatisfaction'].value_counts().sort_index()
p = hr0['RelationshipSatisfaction'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp15 = pd.concat([o,p],axis = 1)
hr0_temp15.columns = ['Nr of Employees', '%']
display(hr0_temp15)

# Group employees per employee_PerformanceRating by brackets
employee_PerformanceRating = hr0.groupby('RelationshipSatisfaction')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'RelationshipSatisfaction', y = 'EmployeeNumber', data = employee_PerformanceRating)
ax.set_xlabel('RelationshipSatisfaction')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('RelationshipSatisfaction')
sns.despine()
plt.show()

PerformanceRating_Employee = hr0.groupby(['RelationshipSatisfaction'])['RelationshipSatisfaction'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
PerformanceRating_Employee.plot(kind='pie', subplots=True, figsize=(4, 3), startangle = 90,autopct = '%1.f%%')
plt.title("RelationshipSatisfaction")
plt.ylabel('')
#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# ### StockOptionLevel

# In[37]:


#Describe the variable 'StockOptionLevel' 
o = hr0['StockOptionLevel'].value_counts().sort_index()
p = hr0['StockOptionLevel'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp16 = pd.concat([o,p],axis = 1)
hr0_temp16.columns = ['Nr of Employees', '%']
display(hr0_temp16)

# Group employees per employee_StockOptionLevel 
employee_StockOptionLevel = hr0.groupby('StockOptionLevel')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'StockOptionLevel', y = 'EmployeeNumber', data = employee_StockOptionLevel)
ax.set_xlabel('StockOptionLevel')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('StockOptionLevel')
sns.despine()
plt.show()

StockOptionLevel_Employee = hr0.groupby(['StockOptionLevel'])['StockOptionLevel'].count()

# Create a pie chart (with one slide exploded out, start angle at 90% and with the percent listed as a fraction)
StockOptionLevel_Employee.plot(kind='pie', subplots=True, figsize=(4, 3), startangle = 90,autopct = '%1.f%%')
plt.title("StockOptionLevel")
plt.ylabel('')
#tight_layout: adjusts subplot to fit in to the figure are
plt.tight_layout()
plt.show()


# #Nota: na tabela seguinte colocar por ordem e se for rápido com cor gradiente

# ### TotalWorkingYears

# In[38]:


#Describe the variable 'TotalWorkingYears'
o = hr0['TotalWorkingYears'].value_counts().sort_index()
p = hr0['TotalWorkingYears'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp17 = pd.concat([o,p],axis = 1)
hr0_temp17.columns = ['Nr of Employees', '%']

display(hr0_temp17)

# Group employees per TotalWorkingYears
employee_TotalWorkingYears = hr0.groupby('TotalWorkingYears')['EmployeeNumber'].count().reset_index()

plt.figure(figsize=(12,4))
ax = sns.barplot(x = 'TotalWorkingYears', y = 'EmployeeNumber', data = employee_TotalWorkingYears)

ax.set_xlabel('TotalWorkingYears')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('TotalWorkingYears')
sns.despine()
plt.show()


# In[39]:


#TotalWorkingYears_Bracket in the main clusters
bins = [0,5, 10, 15, 20, 25, 30, 35, 40, 45]
labels = ['0-5','5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40+']
hr0['TotalWorkingYears_Bracket']= pd.cut(hr0['TotalWorkingYears'], bins, labels = labels,include_lowest = True)


#Describe the variable 'TotalWorkingYears' by brackets
o = hr0['TotalWorkingYears_Bracket'].value_counts().sort_index()
p = hr0['TotalWorkingYears_Bracket'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp18 = pd.concat([o,p],axis = 1)
hr0_temp18.columns = ['Nr of Employees', '%']
display(hr0_temp18)

# Group employees per TotalWorkingYears by brackets
employee_PercentSalaryHike_Bracket = hr0.groupby('TotalWorkingYears_Bracket')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'TotalWorkingYears_Bracket', y = 'EmployeeNumber', data = employee_PercentSalaryHike_Bracket)
ax.set_xlabel('TotalWorkingYears_Bracket')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('TotalWorkingYears_Bracket')
sns.despine()
plt.show()


# ### TrainingTimesLastYear

# In[40]:


#Describe the variable 'TrainingTimesLastYear'
o = hr0['TrainingTimesLastYear'].value_counts().sort_index()
p = hr0['TrainingTimesLastYear'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp19 = pd.concat([o,p],axis = 1)
hr0_temp19.columns = ['Nr of Employees', '%']

display(hr0_temp19)

# Group employees per 'TrainingTimesLastYear'
employee_TrainingTimesLastYear = hr0.groupby('TrainingTimesLastYear')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'TrainingTimesLastYear', y = 'EmployeeNumber', data = employee_TrainingTimesLastYear)
ax.set_xlabel('TrainingTimesLastYear')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('TrainingTimesLastYear')
sns.despine()
plt.show()


# ### WorkLifeBalance 

# In[41]:


#Describe the variable 'WorkLifeBalance'
o = hr0['WorkLifeBalance'].value_counts().sort_index()
p = hr0['WorkLifeBalance'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp20 = pd.concat([o,p],axis = 1)
hr0_temp20.columns = ['Nr of Employees', '%']
display(hr0_temp20)

# Group employees per 'WorkLifeBalance'
employee_WorkLifeBalance = hr0.groupby('WorkLifeBalance')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'WorkLifeBalance', y = 'EmployeeNumber', data = employee_WorkLifeBalance)
ax.set_xlabel('WorkLifeBalance')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('WorkLifeBalance')
sns.despine()
plt.show()


# ### YearsAtCompany 

# In[42]:


#Describe the variable 'YearsAtCompany'
o = hr0['YearsAtCompany'].value_counts().sort_index()
p = hr0['YearsAtCompany'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp21 = pd.concat([o,p],axis = 1)
hr0_temp21.columns = ['Nr of Employees', '%']
display(hr0_temp21)

# Group employees per 'YearsAtCompany'
employee_YearsAtCompany = hr0.groupby('YearsAtCompany')['EmployeeNumber'].count().reset_index()

plt.figure(figsize=(12,4))

ax = sns.barplot(x = 'YearsAtCompany', y = 'EmployeeNumber', data = employee_YearsAtCompany)
ax.set_xlabel('YearsAtCompany')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('YearsAtCompany')
sns.despine()
plt.show()


# In[43]:


#TotalWorkingYears_Bracket in the main clusters
bins = [0,5, 10, 15, 20, 25, 30, 35, 40, 45]
labels = ['0-5','5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40+']
hr0['YearsAtCompany_Bracket']= pd.cut(hr0['YearsAtCompany'], bins, labels = labels,include_lowest = True)


#Describe the variable 'YearsAtCompany' by brackets
o = hr0['YearsAtCompany_Bracket'].value_counts().sort_index()
p = hr0['YearsAtCompany_Bracket'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp22 = pd.concat([o,p],axis = 1)
hr0_temp22.columns = ['Nr of Employees', '%']
display(hr0_temp22)

# Group employees per TotalWorkingYears by brackets
employee_YearsAtCompany_Bracket = hr0.groupby('YearsAtCompany_Bracket')['EmployeeNumber'].count().reset_index()

ax = sns.barplot(x = 'YearsAtCompany_Bracket', y = 'EmployeeNumber', data = employee_YearsAtCompany_Bracket)
ax.set_xlabel('YearsAtCompany_Bracket')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('YearsAtCompany_Bracket')
sns.despine()
plt.show()


# ### YearsInCurrentRole 

# In[44]:


#Describe the variable 'YearsInCurrentRole'
o = hr0['YearsInCurrentRole'].value_counts().sort_index()
p = hr0['YearsInCurrentRole'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp23 = pd.concat([o,p],axis = 1)
hr0_temp23.columns = ['Nr of Employees', '%']
display(hr0_temp23)

# Group employees per 'YearsInCurrentRole'
employee_YearsInCurrentRole = hr0.groupby('YearsInCurrentRole')['EmployeeNumber'].count().reset_index()

#plt.figure(figsize=(12,4))

ax = sns.barplot(x = 'YearsInCurrentRole', y = 'EmployeeNumber', data = employee_YearsInCurrentRole)
ax.set_xlabel('YearsInCurrentRole')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('YearsInCurrentRole')
sns.despine()
plt.show()


# ### YearsSinceLastPromotion

# In[45]:


#Describe the variable 'YearsSinceLastPromotion'
o = hr0['YearsSinceLastPromotion'].value_counts().sort_index()
p = hr0['YearsSinceLastPromotion'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp24 = pd.concat([o,p],axis = 1)
hr0_temp24.columns = ['Nr of Employees', '%']
display(hr0_temp24)

# Group employees per 'YearsSinceLastPromotion'
employee_YearsSinceLastPromotion = hr0.groupby('YearsSinceLastPromotion')['EmployeeNumber'].count().reset_index()

#plt.figure(figsize=(12,4))

ax = sns.barplot(x = 'YearsSinceLastPromotion', y = 'EmployeeNumber', data = employee_YearsSinceLastPromotion)
ax.set_xlabel('YearsSinceLastPromotion')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('YearsSinceLastPromotion')
sns.despine()
plt.show()


# ### YearsWithCurrManager 

# In[46]:


#Describe the variable 'YearsWithCurrManager'
o = hr0['YearsWithCurrManager'].value_counts().sort_index()
p = hr0['YearsWithCurrManager'].value_counts(normalize=True) * 100
p = p.map('{:,.1f} %'.format).sort_index()
hr0_temp25 = pd.concat([o,p],axis = 1)
hr0_temp25.columns = ['Nr of Employees', '%']
display(hr0_temp25)

# Group employees per 'YearsWithCurrManager'
employee_YearsWithCurrManager = hr0.groupby('YearsWithCurrManager')['EmployeeNumber'].count().reset_index()

#plt.figure(figsize=(12,4))

ax = sns.barplot(x = 'YearsWithCurrManager', y = 'EmployeeNumber', data = employee_YearsWithCurrManager)
ax.set_xlabel('YearsWithCurrManager')
ax.set_ylabel('Nr of Employees', fontsize = 14)
plt.title('YearsWithCurrManager')
sns.despine()
plt.show()


# ### Feature Engineering

# In[47]:


# New variables

# Age of entry in company
hr0['Age_Entry']=hr0['Age']-hr0['YearsAtCompany']

# Age of entry in the workforce
hr0['Age_Workforce']=hr0['Age']-hr0['TotalWorkingYears']

# Binary variable indicating if employee has worked solemny at this company
hr0['FLG_1stjob']=np.where(hr0['TotalWorkingYears']==hr0['YearsAtCompany'],1,0)

#Percentage of work lifetime in the company.
hr0['Perc_lftm_company']=np.where(hr0['TotalWorkingYears']!=0,(hr0['YearsAtCompany']/hr0['TotalWorkingYears'])*100,0)

#Perc of time in the company with current manager
hr0['Perc_tmcompany_cur_manager']=np.where(hr0['YearsAtCompany']!=0,(hr0['YearsWithCurrManager']/hr0['YearsAtCompany'])*100,0)

#PercentSalaryHike in the main clusters
bins = [10, 20, 30, 40, 50, 60, 70, 100]
labels = ['<20','20-29', '30-39', '40-49', '50-59', '60-69', '70+']
hr0['Age_Bracket']= pd.cut(hr0['Age'], bins, labels = labels,include_lowest = True)

hr0.head()


# In[48]:


#confirms if new variables have ranges that make sense (validates info in the dataset)
hr0[['Age_Bracket','Age_Entry','Age_Workforce','FLG_1stjob','Perc_lftm_company','Perc_tmcompany_cur_manager']].describe().T


# In[49]:


# Average time a worked worked at other companies
hr0['Avg_prev_worktime']=np.where(hr0['NumCompaniesWorked']!=0,(
    hr0['TotalWorkingYears']-hr0['YearsAtCompany'])/hr0['NumCompaniesWorked'],0)

#Income to daily rate ratio
hr0['IncRateRatio']=hr0['MonthlyIncome']/hr0['DailyRate']

#Income to years at company ratio
hr0['IncYearsRatio']=np.where(hr0['YearsAtCompany']!=0,(hr0['MonthlyIncome']/hr0['YearsAtCompany']),0)

#Ratio Monthly Rate
hr0['IncMonthlyRate']=np.where(hr0['MonthlyIncome']!=0,(hr0['MonthlyRate']/hr0['MonthlyIncome']),0)

#Hourly Rate ratio
hr0['IncHourlyRate']=np.where(hr0['HourlyRate']!=0,(hr0['MonthlyIncome']/hr0['HourlyRate']),0)

#Hours in Daily Rate
hr0['HrDailyRate']=np.where(hr0['HourlyRate']!=0,(hr0['DailyRate']/hr0['HourlyRate']),0)


# ### Variable Transformation

# - `Attrition`: binary variable - 1_YES and 0_NO
# - `Gender`: binary variable - 0_MALE and 1_FEMALE
# - `OverTime`: binary variable - 1_YES and 0_NO
# - `MaritalStatus`: binary variable - 1_Married and 0_Single/Divorced

# In[50]:


# To convert selected features into binary variables 
hr0['Attrition'] = hr0['Attrition'].replace({'No': 0, 'Yes': 1})
hr0['Gender'] = hr0['Gender'].replace({'Male': 0, 'Female': 1})
hr0['OverTime'] = hr0['OverTime'].replace({'No': 0, 'Yes': 1})

sph = {'Divorced':0,'Married':1,'Single':0}
hr0['MS'] = hr0['MaritalStatus'].map(sph)


# To validate the previous operations
hr0[['Attrition','Gender','OverTime']].describe().T


# In[51]:


bt = {'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}
hr0['BT'] = hr0['BusinessTravel'].map(bt)


# # Correlation Analysis

# In[234]:


#Prepares dataset for correlation. Some variables won't make sense, so we'll start by removing them.
#drop(['Over18'], axis = 1)
hr0_corr = hr0.drop(['Attrition','EmployeeNumber','OverTime','Gender','MS','MonthlyRate','HourlyRate'], axis=1).copy()
corr = hr0_corr.corr()
figure = plt.figure(figsize=(16,10))
sns.heatmap(corr, annot=True, fmt = '.1g')


# In[236]:


#Previous correlation graph was not very readable. Extra variables with weak correlations are removed.
hr0_corrp = hr0.drop(['Attrition','EmployeeNumber','OverTime','Gender','MS','MonthlyRate','HourlyRate',
                      'StockOptionLevel','RelationshipSatisfaction','JobInvolvement','EnvironmentSatisfaction',
                     'DistanceFromHome','PerformanceRating','PercentSalaryHike','WorkLifeBalance',
                      'TrainingTimesLastYear','JobSatisfaction','BT'], axis=1).copy()
corrp = hr0_corrp.corr()
figure = plt.figure(figsize=(10,8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corrp, annot=True, fmt = '.1g',cmap=cmap)


# In[54]:


#Prepares dataset for Spearman correlation.
#drop(['Over18'], axis = 1)
hr0_corr = hr0.drop(['Attrition','EmployeeNumber','OverTime','Gender','MS','MonthlyRate','HourlyRate'], axis=1).copy()
#Spearmen correlation
figure = plt.figure(figsize=(16,16))
cor_spearman = hr0_corr.corr(method ='spearman')
sns.heatmap(cor_spearman, annot=True, fmt = '.1g')


# In[55]:


#Defines a funtion to detect highly correlated features (aka above 0.8 correlation) to subsequently remove them in the 
#feature selection phase
def correlated_features(corr_matrix):
    corr_featureS = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colnamei = corr_matrix.columns[i]
                corr_featureS.add(colnamei)
                colnamej = corr_matrix.columns[j]
                print('Feature\033[1m', colnamei, '\033[0mis correlated to\033[1m', colnamej, 
                      '\033[0mwith a correlation coefficient of:\033[1m', corr_matrix.iloc[i, j],'\033[0m')

    return corr_featureS


# In[56]:


#Run function on pearson correlation
correlated_features(corr)


# In[57]:


#Run function on spearman correlation
correlated_features(cor_spearman)


# # Outliers Analysis

# In[58]:


OutColumns=hr0.drop(['Attrition','EmployeeNumber','OverTime','Gender','BusinessTravel','Department',
                    'EducationField','JobRole','MaritalStatus','PercentSalaryHike_Bracket',
                    'TotalWorkingYears_Bracket','YearsAtCompany_Bracket','Age_Bracket','FLG_1stjob','PerformanceRating',
                    'MS','BT'], axis=1).columns
range(len(OutColumns))


# In[59]:


fig, axes= plt.subplots(figsize=(16,40))
whiskers=dict()
for o in range(len(OutColumns)):
    plt.subplot(9, 4, o+1)
    C1=plt.boxplot(x = OutColumns[o], data = hr0)
    plt.title(OutColumns[o])
    w=[item.get_ydata() for item in C1['whiskers']]
    whiskers[OutColumns[o]] = (w[0][1],w[1][1])


# **Note:** apart from the Income and Income related ratios, the outliers in the variables of the dataset don't hold values too far from the median - most of them are in the same order of magnitude. Because all of the values make sense within the context, we won't be removing outliers in the dataset - but we will proceed with an extra analysis of the Income outliers.

# ## Income outliers by department

# In[60]:


to_graph=hr0[['EmployeeNumber','Department','MonthlyIncome']]


# In[61]:


#
fig, axes= plt.subplots(figsize=(16,16))
o=0
whiskers=dict()
for label, group in to_graph.groupby(["Department"]):
    plt.subplot(4, 4, o+1)
    C1=plt.boxplot(x = group['MonthlyIncome'], data = group)
    plt.title(label)
    w=[item.get_ydata() for item in C1['whiskers']]
    whiskers[label] = (w[0][1],w[1][1])
    o=o+1


# ## Income outliers by JobRole

# In[62]:


to_graph=hr0[['EmployeeNumber','JobRole','MonthlyIncome']]
#
fig, axes= plt.subplots(figsize=(16,16))
o=0
whiskers=dict()
for label, group in to_graph.groupby(["JobRole"]):
    plt.subplot(4, 4, o+1)
    C1=plt.boxplot(x = group['MonthlyIncome'], data = group)
    plt.title(label)
    w=[item.get_ydata() for item in C1['whiskers']]
    whiskers[label] = (w[0][1],w[1][1])
    o=o+1


# #### Lower outliers analysis
# The positions of manager and sales representative have lower bound outliers. Is there greater attrition in these outliers? Is there dissatisfaction if comparatively, peoples' income is lower than their company peers?

# In[63]:


whiskers['Manager'][0]


# In[64]:


display(hr0[['Attrition','EmployeeNumber','JobRole','MonthlyIncome']].loc[(
    (hr0['MonthlyIncome'] < 12504.0) & (hr0['JobRole'] == 'Manager'))].describe())


# In[65]:


display(hr0[['Attrition','EmployeeNumber','JobRole','MonthlyIncome']].loc[(
    (hr0['MonthlyIncome'] < whiskers['Sales Representative'][0]) & (hr0['JobRole'] == 'Sales Representative'))].describe())


# There seems to be greater attrition in  lower outliers in the Sales representative job role (60% leave - 3 out of 5).

# ## Income outliers by EducationField and level

# In[66]:


to_graph=hr0[['EmployeeNumber','EducationField','MonthlyIncome']]
#
fig, axes= plt.subplots(figsize=(16,16))
o=0
whiskers=dict()
for label, group in to_graph.groupby(["EducationField"]):
    plt.subplot(4, 4, o+1)
    C1=plt.boxplot(x = group['MonthlyIncome'], data = group)
    plt.title(label)
    w=[item.get_ydata() for item in C1['whiskers']]
    whiskers[label] = (w[0][1],w[1][1])
    o=o+1


# In[67]:


to_graph=hr0[['EmployeeNumber','Education','MonthlyIncome']]
#
fig, axes= plt.subplots(figsize=(16,16))
o=0
whiskers=dict()
for label, group in to_graph.groupby(["Education"]):
    plt.subplot(4, 4, o+1)
    C1=plt.boxplot(x = group['MonthlyIncome'], data = group)
    plt.title(label)
    w=[item.get_ydata() for item in C1['whiskers']]
    whiskers[label] = (w[0][1],w[1][1])
    o=o+1


# No further lower bound outliers are observed.

# # Data Transformation

# Data Transformation involves important steps of numerical feature engineering, since the range of values of data varies widely. In some machine learning algorithms, there are functions that will not work properly without data scaling. This scaling can be achieved by standardizing or normalizing numerical input features.
# 
# Standardization and normalization are both used to build features that have similar ranges to each other.
# To gather better insight about the data, we decided to apply both techniques, so as to perceive their effect on the features. While standardization is applied to multivariate analysis on data of different units, this approach has the assumption that the data follows a Gaussian distribution. Also, while normalization shrinks the range of the data such that it is fixed between 0 and 1, standardization does not have a bounding range, which means that even if outliers are present in the data, the data will not be much affected by these outliers with this method. However, since we cannot at this point assume that all features are equally important, and considering that most of our data is not normally distributed, normalization is expected to be more indicated for further procedures in our case. As such, we will apply both scaling methodologies, but we anticipate to rely mainly on the normalized dataset to draw our conclusions.

# In[57]:


#Transform remaining categorical variables by one hot encoding:
hr1.hr0.copy()

# BusinessTravel (if both have 0 then 'travels rarely')
list_bin = [1 if value == 'Non-Travel' else 0 for value in eval('hr1[' + '\'' + 'BusinessTravel' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'BT_Non-Travel' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Travel_Frequently' else 0 for value in eval('hr1[' + '\'' + 'BusinessTravel' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'BT_Travel_Frequently' + '\'' + '] = list_bin')


# Department (if both have 0 then 'Research and Development')
list_bin = [1 if value == 'Sales' else 0 for value in eval('hr1[' + '\'' + 'Department' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Dep_Sales' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Human Resources' else 0 for value in eval('hr1[' + '\'' + 'Department' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Dep_HR' + '\'' + '] = list_bin')


# EducationField (if all have 0 then 'Other')
list_bin = [1 if value == 'Human Resources' else 0 for value in eval('hr1[' + '\'' + 'EducationField' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Educ_HR' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Life Sciences' else 0 for value in eval('hr1[' + '\'' + 'EducationField' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Educ_LifeSciences' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Marketing' else 0 for value in eval('hr1[' + '\'' + 'EducationField' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Educ_Marketing' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Medical' else 0 for value in eval('hr1[' + '\'' + 'EducationField' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Educ_Medical' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Technical Degree' else 0 for value in eval('hr1[' + '\'' + 'EducationField' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'Educ_TechDegree' + '\'' + '] = list_bin')


# JobRole (if all have 0 then 'Manager')

list_bin = [1 if value == 'Healthcare Representative' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_HealthcareRep' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Human Resources' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_HR' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Laboratory Technician' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_LabTech' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Manufacturing Director' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_ManufactDirec' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Research Director' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_ResearchDirec' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Research Scientist' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_ResearchScientist' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Sales Executive' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_SalesExec' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Sales Representative' else 0 for value in eval('hr1[' + '\'' + 'JobRole' + '\'' + ']')]
exec('hr1.loc[:, ' + '\'' + 'JR_SalesRep' + '\'' + '] = list_bin')


# In[64]:


#remove all non-numerical and bracket columns and the dependent variable (attrition) as
#we won't need for this uspervised learning analysis (clustering)

temp_df = hr1.copy()

for column in temp_df.columns:

    # Drop all non-numerical and brackets columns.
    if type(temp_df[column][0]) != np.int64 and type(temp_df[column][0]) != np.int32 and type(temp_df[column][0]) != np.float64     or temp_df[column].name == 'Attrition' or temp_df[column].name == 'EmployeeNumber': #and all the newer
        temp_df.drop(columns=column, inplace=True)
      
    df = temp_df.copy()
df


# ## Standardization

# In[65]:


# Create an instance of the StandardScaler class.
standard_scaler = StandardScaler()

# Standardize our dataset (returns an array).
standardized_df = standard_scaler.fit_transform(df)

# Convert the array above back into a pandas DataFrame.
standardized_df = pd.DataFrame(data=standardized_df, columns=temp_df.columns)
standardized_df 


# ## Normalization

# In[66]:


# Create an instance of the MinMaxScaler class.
min_max_scaler = MinMaxScaler()

# Normalize our dataset (returns an array).
normalized_df = min_max_scaler.fit_transform(df)

# Convert the array above back into a pandas DataFrame.
normalized_df = pd.DataFrame(data=normalized_df, columns=temp_df.columns)
normalized_df


# In[67]:


datasets = {'original': df, 'standardized': standardized_df, 'normalized': normalized_df}
datasets


# # Data Modelling - Clustering

# We need to group the records in different segments, using similarities as measurement. The features are properly divided, in order to promote the optimization of the variables involved. So our data-driven approach is to split the data in three main perspectives: employee, job position, historic and ongoing. 
# 
# **- Employee segmentation** in regards of specific attributes of each employees such as their age, gender, marital status, education, etc.
# 
# **- Job Position segmentation** relates to specifics of an employee's job position in the company. Here we analyze what's their job position in which department and what's their level.
# 
# **- Historic segmentation** in regards of an employee's past, meaning we are going to group variables like the age they entered IBM,the number of companies they worked, total years at this company, in current role and with current manager, etc.
# 
# **- Ongoing segmentation** relates to specifics of an employee's present situation in the company. Here we analyze what's their job envolvement and environment satisfaction, salary, work/life balance, number of years since last promotion, etc.

# ## Employee Segmentation

# In[68]:


# Create a dictionary to hold the employee perspective for each dataset (original, standardized, and normalized).
employee_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the employee segmentation perspective.
    employee_columns = ['Age',
                                        'DistanceFromHome',
                                        'Education',
                                        'Gender',
                                        'MS',
                                        'Educ_HR',
                                        'Educ_LifeSciences',
                                        'Educ_Marketing',
                                        'Educ_Medical',
                                        'Educ_TechDegree'
                                       ]

    # Store perspectives into general dictionary.
    employee_info[key] = sub_df[employee_columns]


# ### Correlation Analysis

# In order to strengthen a model, the natural next step is to identify and reduce the features that are highly correlated, since they can skew the output due to  the possibility of having the same information. In this sense, a possible way to achieve this is by visualizing these correlations as a **heatmap**. This graphical representation of correlations compares magnitudes of negative versus positive values and highlights the areas where there are high positive and negative correlations.

# In[69]:


for key in employee_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(employee_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 10},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('Employee Segmentation:\n' + key.capitalize() + ' Data', size=30, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90, size=15)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=15)
    ax.tick_params(axis='both', bottom=False, left=False)


# Regarding this whole segmentation,  we observe no significant difference between the original, standardized and normalized data, and we have low correlated variables in general (ignoring correlation between the dummy variables which don't make sense). Thus, we are going to keep all variables and keep on with the analysis by reaching an optimal number of clusters by using two techinques: Hierarchical clustering and k-means clustering.

# ### Hierarchical Clustering

# Through this kind of diagrams that show the hierarchical relationship between observations it is possible to allocate them to clusters. By drawing a horizontal line through the dendrogram, we can figure out the possible ideal number of clusters and confront them with other steps of algorithms, like k-means.

# In[70]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(employee_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=employee_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('Employee Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# From observing the above dendrograms, we found the following number of clusters (spearated by different colors):
# <br> 
#  - original data: 3
# <br>
#  - standardized data: 5
# <br> 
#  - normalized data: 3

# ### K-Means

# K-means is a great solution when solving clustering tasks to get an idea of the dataset structure. Through the number of predetermined clusters each data point is iteratively assigned to one of the clusters based on feature similarity. This step is crucial for the market segmentation, where we try to find employees that are similar to each other (high intra-class similarity) but different between clusters (low inter-class similarity).
# 
# The main advantages of this method, comparing to the hierarchical ones, are the easier implementation and its ability to reassemble elements that previously belonged to another cluster. Consequently, an erroneous classification of an individual is very unlikely.

# In[71]:


# Select the range of clusters we wan to attempt (let's choose n+1 = 11)
nclusters = range(1, 11)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(employee_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(employee_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('Employee Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# Another method to determine the optimal number of clusters is the **elbow method**, which is based on the sum of squared distance (SSE) between data points and their assigned clusters’ centroids. To find this number of clusters, we observe the elbow graphs and confront the analysis with the interpretations of the dendrograms.

# In[72]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 3, 'standardized': 5, 'normalized': 3}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(employee_info[key])

    # Get the cluster to which each observation belongs to.
    employee_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[73]:


# Get descriptive statistics for each feature of each cluster.
for key in employee_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in employee_info[key].columns[:-1]:
        print(employee_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[74]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in employee_info.keys()}

for key in employee_info.keys():
    
    # Get number of clusters.
    n_clusters = employee_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = employee_info[key][employee_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# We will now compare the distribution of each feature in each cluster with the distribution of the features in the whole sample.

# In[75]:


for column in employee_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution for the whole employee base.
        sns.distplot(employee_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('employee Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# We will now explore the relationship of each feature pairwise and their segmentation into clusters via a scatterplot. This could give some insight as to the nature of which features best segment our data.

# In[76]:


for c, combination in enumerate(itertools.combinations(employee_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=employee_info[key][combination[0]],
                                y=employee_info[key][combination[1]],
                                hue=employee_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][employee_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][employee_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('Employee Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# In[77]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(employee_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = employee_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')

    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('Employee Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

for n, cluster in enumerate(cluster_info['normalized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

plt.show()
plt.close()


# From the above comparison of each feature mean for each cluster, we can clearly see, in the standardized and normalized cases, that the education field flags are dominating the segmentation in a non-informative way. We now plan to remove these features from the segmentation process and re-run the segmentation pipeline, adjusting the number of cluster accordingly.

# <a class="anchor" id="8.1.2-improved-approach"></a>
# ### Improved Approach

# <a class="anchor" id="8.1.2.1-correlation-analysis"></a>
# #### Correlation Analysis

# In[78]:


# Create a dictionary to hold the employee perspective for each dataset (original, standardized, and normalized).
employee_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the employee perspective.
    employee_columns = ['Age',
                                        'DistanceFromHome',
                                        'Education',
                                        'Gender',
                                        'MS'
                                       ]

    # Store perspectives into general dictionary.
    employee_info[key] = sub_df[employee_columns]


# In[79]:


for key in employee_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(18, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(employee_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 15},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('employee Segmentation:\n' + key.capitalize() + ' Data', size=26, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, size=20)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=20)
    ax.tick_params(axis='both', bottom=False, left=False)


# <a class="anchor" id="8.1.2.2-hierarchical-clustering"></a>
# #### Hierarchical Clustering

# In[80]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(employee_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=employee_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('employee Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# As before, we try to understand the effect of the distinct scaling methods through the employee segment. 
# 
# From observing the above dendrograms, we found the following number of clusters:
# <br> 
#  - original data: 3
# <br>
#  - standardized data: 3
# <br> 
#  - normalized data: 3

# <a class="anchor" id="8.1.2.3-k-means"></a>
# #### K-Means

# In[81]:


# Select the range of clusters we wan to attempt.
nclusters = range(1, 6)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(employee_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(employee_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('employee Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[82]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 3, 'standardized': 3, 'normalized': 3}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(employee_info[key])

    # Get the cluster to which each observation belongs to.
    employee_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[83]:


# Get descriptive statistics for each feature of each cluster.
for key in employee_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in employee_info[key].columns[:-1]:
        print(employee_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[84]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in employee_info.keys()}

for key in employee_info.keys():
    
    # Get number of clusters.
    n_clusters = employee_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = employee_info[key][employee_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# In[85]:


for column in employee_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution for the whole employee base.
        sns.distplot(employee_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('employee Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# In[86]:


for c, combination in enumerate(itertools.combinations(employee_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=employee_info[key][combination[0]],
                                y=employee_info[key][combination[1]],
                                hue=employee_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][employee_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][employee_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('employee Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# By reducing the number of clusters and removing uninformative features, we can now observe a much better cluster resolution across all scaling datasets.

# In[87]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(employee_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = employee_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')
    
    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('employee Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

for n, cluster in enumerate(cluster_info['original']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

plt.show()

plt.close()


# From comparing this clustering result with the previous one, we can clearly observe that the variables `Age` and `DistanceFromHome` are the most relevant in the original data. Therefore, as it is the most informative and clear when comparing to the normalized and the standardized dataset, we are going to choose this dataset segmentation.
# <br>**NOTE:** The cluster numbering can differ between the following text and that represented in the figure above, due to the fact that distinct initial seeds can lead to different cluster assignment. Nevertheless, the characterizations that follow are always consistent with the segmentations obtained.
# 
# <br>Based on the observations made from the segmentation of original samples, we can describe the following employee segments:
# 
# <br>**Cluster0 (Oldest)**: This cluster comprises the oldest segment with 29% (n=426) of employees. These employees have no distinction in terms of gender, education and marital status but they are the oldest ones and they usually live close to the company.
# 
# <br>**Cluster1 (Medium aged)**: This cluster comprises the middle-ahged segment with 23% (n=331) of employees that also live far away from their work.
# 
# <br>**Cluster2 (Youngest)**: This cluster comprises the youngest segment with 49% (n=713) of employees. These employees have no distinction in terms of gender, education and marital status but they are the youngest ones (with a mean of about 31 years) and they usually live close to the company just like the first group.

# ## Job Position Segmentation

# In[88]:


# Create a dictionary to hold the job role perspective for each dataset (original, standardized, and normalized).
job_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the employee segmentation perspective.
    job_columns = ['FLG_1stjob',
                                        'BT_Non-Travel',
                                        'BT_Travel_Frequently',
                                        'Dep_Sales',
                                        'Dep_HR',
                                        'JobLevel',
                                        'JR_HealthcareRep',
                                        'JR_HR',
                                        'JR_LabTech',
                                        'JR_ManufactDirec',
                                        'JR_ResearchDirec',
                                        'JR_ResearchScientist',
                                        'JR_SalesExec',
                                        'JR_SalesRep'
                                       ]

    # Store perspectives into general dictionary.
    job_info[key] = sub_df[job_columns]


# ### Correlation Analysis

# In[89]:


for key in job_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(job_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 10},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('Job Position Segmentation:\n' + key.capitalize() + ' Data', size=30, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90, size=15)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=15)
    ax.tick_params(axis='both', bottom=False, left=False)


# Regarding this whole segmentation,  we observe no significant difference between the original, standardized and normalized data, and we have low correlated variables in general (ignoring correlation between the dummy variables which don't make sense). We have some exceptions related to the correlation between some departments and some job roles (for instance in the HR sector) in which we see high positive correlations because they are connected and coeherent (for example in the dep HR, we only have job roles of HR or managers). We also see some interesting medium correlations between some variables and 'Job Level', with some job roles having positive relationships and others negative, meaning that usually roles like a research scientist, for instance, have a lower job level then directors or executives, which make sense,

# ### Hierarchical Clustering

# In[90]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(job_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=job_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('Job Position Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# From observing the above dendrograms, we found the following number of clusters (spearated by different colors):
# <br> 
#  - original data: 2
# <br>
#  - standardized data: 9
# <br> 
#  - normalized data: 2

# ### K-Means

# In[91]:


# Select the range of clusters we wan to attempt (let's choose n+1 = 15)
nclusters = range(1, 15)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(job_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(job_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('Job Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[92]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 4, 'standardized': 8, 'normalized': 4}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(job_info[key])

    # Get the cluster to which each observation belongs to.
    job_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[93]:


# Get descriptive statistics for each feature of each cluster.
for key in job_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in job_info[key].columns[:-1]:
        print(job_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[94]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in job_info.keys()}

for key in job_info.keys():
    
    # Get number of clusters.
    n_clusters = job_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = job_info[key][job_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# We will now compare the distribution of each feature in each cluster with the distribution of the features in the whole sample.

# In[95]:


for column in job_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution
        sns.distplot(job_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('Employee Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# We will now explore the relationship of each feature pairwise and their segmentation into clusters via a scatterplot. This could give some insight as to the nature of which features best segment our data.

# In[96]:


for c, combination in enumerate(itertools.combinations(job_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=job_info[key][combination[0]],
                                y=job_info[key][combination[1]],
                                hue=job_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][job_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][job_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('job Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# From the observations of the scatterplots, we conclude that for the standardized data, 5 clusters is too high a number of clusters, since in almost all cases the majority of the cluster centroids are very close to each other, and many of the observations are overlapping, rather than occupying discrete regions of space. For the onormlized data, we can reason the same although we only have 3 centroids, therefore the best yielded results in this case are the 3 clusters from the **original data**, which are the most well defined and separated from each other.

# In[97]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(job_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = job_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')

    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('job Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

for n, cluster in enumerate(cluster_info['normalized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

plt.show()
plt.close()


# From the above comparison of each feature mean for each cluster, we can clearly see, we have too many variables that are biasing some results in a non-informative way. What we can do to improve is remove the **Job Role flags**, because some are already redundant with the department and thus, it would be clearer to distinguish groups by department and not per job role.We now plan to remove these features from the segmentation process and re-run the segmentation pipeline, adjusting the number of cluster accordingly.

# <a class="anchor" id="8.1.2-improved-approach"></a>
# ### Improved Approach

# <a class="anchor" id="8.1.2.1-correlation-analysis"></a>
# #### Correlation Analysis

# In[98]:


# Create a dictionary to hold the job perspective for each dataset (original, standardized, and normalized).
job_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the job perspective.
    job_columns = ['FLG_1stjob',
                                        'BT_Non-Travel',
                                        'BT_Travel_Frequently',
                                        'Dep_Sales',
                                        'Dep_HR',
                                        'JobLevel'
                                       ]

    # Store perspectives into general dictionary.
    job_info[key] = sub_df[job_columns]


# In[99]:


for key in job_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(18, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(job_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 15},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('job Segmentation:\n' + key.capitalize() + ' Data', size=26, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, size=20)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=20)
    ax.tick_params(axis='both', bottom=False, left=False)


# <a class="anchor" id="8.1.2.2-hierarchical-clustering"></a>
# #### Hierarchical Clustering

# In[100]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(job_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=job_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    # (without any leaf labels - given the fact that we have 2500 elements, this greatly increases plotting speed: from ~20mins to ~20secs).
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('job Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# As before, we try to understand the effect of the distinct scaling methods through the job segment. 
# 
# From observing the above dendrograms, we found the following number of clusters:
# <br> 
#  - original data: 2
# <br>
#  - standardized data: 5
# <br> 
#  - normalized data: 5

# <a class="anchor" id="8.1.2.3-k-means"></a>
# #### K-Means

# In[101]:


# Select the range of clusters we wan to attempt.
nclusters = range(1, 7)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(job_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(job_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('job Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[102]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 3, 'standardized': 3, 'normalized': 3}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(job_info[key])

    # Get the cluster to which each observation belongs to.
    job_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[103]:


# Get descriptive statistics for each feature of each cluster.
for key in job_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in job_info[key].columns[:-1]:
        print(job_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[104]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in job_info.keys()}

for key in job_info.keys():
    
    # Get number of clusters.
    n_clusters = job_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = job_info[key][job_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# In[105]:


for column in job_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution for the whole job base.
        sns.distplot(job_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('job Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# In[106]:


for c, combination in enumerate(itertools.combinations(job_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=job_info[key][combination[0]],
                                y=job_info[key][combination[1]],
                                hue=job_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][job_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][job_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('job Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# By removing uninformative features, we can now observe a much better cluster resolution across all scaling datasets.

# In[107]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(job_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = job_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')
    
    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('job Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

for n, cluster in enumerate(cluster_info['normalized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

plt.show()

plt.close()


# From this results, we are going to choose the **normalized dataset** fo the Job Position segmentation groups, as the distinction between variables seem the most informative. Here the variables `FLG_1stjob`, `JobLevel` and the departments are the most relevant.
# <br>**NOTE:** The cluster numbering can differ between the following text and that represented in the figure above, due to the fact that distinct initial seeds can lead to different cluster assignment. Nevertheless, the characterizations that follow are always consistent with the segmentations obtained.
# 
# <br>Based on the observations made from the segmentation of normalized samples, we can describe the following employee segments:
# 
# <br>**Cluster0 (Experienced Researcher)**: This cluster comprises the majority segment with 48% (n=701) of employees. This group is composed majority with workers on R&D and some on HR, but none ate sales. Their job level is medium to high and all of them already worked on at least 1 other company before. 
# 
# <br>**Cluster1 (Rookie Researchers)**: This cluster comprises all the rookies (meaning that this is their first job) segment with 22% (n=323) of employees that are majority researchers and some HR, just like the previous group. As we could expect, while being their 1st job, this group's job leve is tipically the lowest of all.
# 
# <br>**Cluster2 (Sales workers)**: This cluster comprises all the sales' employees segment with 30% (n=446) of employees. These workers tend to have the highest job level as well and its mixed in terms of people who always have been working in this company and the others that already had previous careers.

# ## Historic Segmentation

# In[108]:


# Create a dictionary to hold the historic perspective for each dataset (original, standardized, and normalized).
historic_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the historic segmentation perspective.
    historic_columns = ['Age_Entry',
                                        'Age_Workforce',
                                        'Avg_prev_worktime',
                                        'NumCompaniesWorked',
                                        'TotalWorkingYears',
                                        'YearsAtCompany',
                                        'YearsInCurrentRole',
                                        'YearsWithCurrManager',
                                        'Perc_lftm_company',
                                        'Perc_tmcompany_cur_manager',
                                       ]

    # Store perspectives into general dictionary.
    historic_info[key] = sub_df[historic_columns]


# ### Correlation Analysis

# In[109]:


for key in historic_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(historic_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 10},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('Historic Segmentation:\n' + key.capitalize() + ' Data', size=30, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90, size=15)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=15)
    ax.tick_params(axis='both', bottom=False, left=False)


# Regarding this whole segmentation,  we observe no significant difference between the original, standardized and normalized data, and we have medium to high correlated variables. Some variables are ratios from others and thus it is fine to see higher correlation between them, for instance 'avg_prev_worktime' with Perc_lftm_company', which both have into account the total working years and the number of years at this company.<br>
# We will keep on with this whole anayzis and, if needed, we shall remove unnecessary features like previous situations.
# 

# ### Hierarchical Clustering

# In[110]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(historic_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=historic_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('Historic Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# From observing the above dendrograms, we found the following number of clusters (spearated by different colors):
# <br> 
#  - original data: 3
# <br>
#  - standardized data: 3
# <br> 
#  - normalized data: 2

# ### K-Means

# In[111]:


# Select the range of clusters we wan to attempt (let's choose n+1 = 11)
nclusters = range(1, 11)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(historic_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(historic_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[112]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 4, 'standardized': 4, 'normalized': 3}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(historic_info[key])

    # Get the cluster to which each observation belongs to.
    historic_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[113]:


# Get descriptive statistics for each feature of each cluster.
for key in historic_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in historic_info[key].columns[:-1]:
        print(historic_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[114]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in historic_info.keys()}

for key in historic_info.keys():
    
    # Get number of clusters.
    n_clusters = historic_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = historic_info[key][historic_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# In[115]:


for column in historic_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution
        sns.distplot(historic_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('Employee Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# We will now explore the relationship of each feature pairwise and their segmentation into clusters via a scatterplot. This could give some insight as to the nature of which features best segment our data.

# In[116]:


for c, combination in enumerate(itertools.combinations(historic_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=historic_info[key][combination[0]],
                                y=historic_info[key][combination[1]],
                                hue=historic_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][historic_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][historic_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# In[117]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(historic_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = historic_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')

    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

for n, cluster in enumerate(cluster_info['normalized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

plt.show()
plt.close()


# As seen, the clearer segments and the most well balanced are within the normalized data. However, here and in the other datasets as well, the last two percentage variables can lead to some confusion and they overwhelm the majority of analyzis. Thus, we will improve this approach by solely removing these two variables.

# <a class="anchor" id="8.1.2-improved-approach"></a>
# ### Improved Approach

# <a class="anchor" id="8.1.2.1-correlation-analysis"></a>
# #### Correlation Analysis

# In[118]:


# Create a dictionary to hold the historic perspective for each dataset (original, standardized, and normalized).
historic_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the historic perspective.
    historic_columns = ['Age_Entry',
                                        'Age_Workforce',
                                        'Avg_prev_worktime',
                                        'NumCompaniesWorked',
                                        'TotalWorkingYears',
                                        'YearsAtCompany',
                                        'YearsInCurrentRole',
                                        'YearsWithCurrManager'
                                       ]

    # Store perspectives into general dictionary.
    historic_info[key] = sub_df[historic_columns]


# In[119]:


for key in historic_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(18, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(historic_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 15},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('historic Segmentation:\n' + key.capitalize() + ' Data', size=26, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, size=20)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=20)
    ax.tick_params(axis='both', bottom=False, left=False)


# <a class="anchor" id="8.1.2.2-hierarchical-clustering"></a>
# #### Hierarchical Clustering

# In[120]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(historic_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=historic_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    # (without any leaf labels - given the fact that we have 2500 elements, this greatly increases plotting speed: from ~20mins to ~20secs).
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# As before, we try to understand the effect of the distinct scaling methods through the historic segment. 
# 
# From observing the above dendrograms, we found the following number of clusters:
# <br> 
#  - original data: 2
# <br>
#  - standardized data: 3
# <br> 
#  - normalized data: 3

# <a class="anchor" id="8.1.2.3-k-means"></a>
# #### K-Means

# In[121]:


# Select the range of clusters we wan to attempt.
nclusters = range(1, 9)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(historic_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(historic_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[122]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 3, 'standardized': 3, 'normalized': 3}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(historic_info[key])

    # Get the cluster to which each observation belongs to.
    historic_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[123]:


# Get descriptive statistics for each feature of each cluster.
for key in historic_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in historic_info[key].columns[:-1]:
        print(historic_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[124]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in historic_info.keys()}

for key in historic_info.keys():
    
    # Get number of clusters.
    n_clusters = historic_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = historic_info[key][historic_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# In[125]:


for column in historic_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution for the whole historic base.
        sns.distplot(historic_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('historic Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# In[126]:


for c, combination in enumerate(itertools.combinations(historic_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=historic_info[key][combination[0]],
                                y=historic_info[key][combination[1]],
                                hue=historic_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][historic_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][historic_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# In[127]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(historic_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = historic_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')
    
    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('historic Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

for n, cluster in enumerate(cluster_info['normalized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

plt.show()

plt.close()


# From comparing this clustering result with the previous one, we can clearly observe that both removed variables do not help in segmenting the historic base.
# 
# <br>**NOTE:** The cluster numbering can differ between the following text and that represented in the figure above, due to the fact that distinct initial seeds can lead to different cluster assignment. Nevertheless, the characterizations that follow are always consistent with the segmentations obtained.
# 
# <br>Based on the observations made from the segmentation of normalized samples, we can describe the following historic segments:
# 
# <br>**Cluster0 (Senior loyal workers)**: This cluster represents 31% (n=456) of employees. People here, entered the company at a lower age and also have the highest working time at this company and in general as well. The total number of companies they worked is medium to low as expected and they also experience the majority of their working years in the current role and with the current manager.
# 
# <br>**Cluster1 (Frequent job changers)**: This cluster represents 32% (n=475) of employees. In this segment, workers have already worked in several companies before and their total working time is high as well. So we get a older workbase which have entered the current company at an older age, thus, being the total years at the company, the current role and with the current manager relatively low when compared to the Cluster0.
# 
# <br>**Cluster2 (Working Life beginners)**: This cluster comprises about 37% (n=539) of employees. In this segment, workers are younger and so, they have lower total working years in general and within this company. However, their entrying age is still, on average, a little bit higher than the Cluster0 group. So, the majority of these employees started to work here and stayed in the same role and with the same manager (most likely they have characteristics in common with the 1st flag job workers).

# ## Ongoing Segmentation

# In[128]:


# Create a dictionary to hold the ongoing perspective for each dataset (original, standardized, and normalized).
ongoing_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the employee segmentation perspective.
    ongoing_columns = ['EnvironmentSatisfaction',
                                        'JobInvolvement',
                                        'JobSatisfaction',
                                        'MonthlyIncome',
                                        'OverTime',
                                        'PercentSalaryHike',
                                        'PerformanceRating',
                                        'RelationshipSatisfaction',
                                        'StockOptionLevel',
                                        'TrainingTimesLastYear',
                                        'WorkLifeBalance',
                                        'YearsSinceLastPromotion',
                                        'DailyRate',
                                        'HourlyRate',
                                        'MonthlyRate',
                                        'IncYearsRatio',
                                        'IncRateRatio'
                                       ]

    # Store perspectives into general dictionary.
    ongoing_info[key] = sub_df[ongoing_columns]


# ### Correlation Analysis

# In[129]:


for key in ongoing_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(ongoing_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 10},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', size=30, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90, size=15)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=15)
    ax.tick_params(axis='both', bottom=False, left=False)


# Regarding this whole segmentation,  we observe no significant difference between the original, standardized and normalized data, and we have low correlated variables in general with exception of our new transformed variable 'IncRateRatio' with 'MonthlyIncome' and 'DailyRate', as they are highly correlated because the ratio variable includes these latter two. We will go on to see if clustering so many variables produces relevant results.

# ### Hierarchical Clustering

# In[130]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(ongoing_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=ongoing_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# From observing the above dendrograms, we found the following number of clusters (spearated by different colors):
# <br> 
#  - original data: 2
# <br>
#  - standardized data: 3
# <br> 
#  - normalized data: 3

# ### K-Means

# In[131]:


# Select the range of clusters we wan to attempt (let's choose n+1 = 18)
nclusters = range(1, 18)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(ongoing_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(ongoing_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[132]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 4, 'standardized': 6, 'normalized': 4}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(ongoing_info[key])

    # Get the cluster to which each observation belongs to.
    ongoing_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[133]:


# Get descriptive statistics for each feature of each cluster.
for key in ongoing_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in ongoing_info[key].columns[:-1]:
        print(ongoing_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[134]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in ongoing_info.keys()}

for key in ongoing_info.keys():
    
    # Get number of clusters.
    n_clusters = ongoing_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = ongoing_info[key][ongoing_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# We will now compare the distribution of each feature in each cluster with the distribution of the features in the whole sample.

# In[135]:


for column in ongoing_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution
        sns.distplot(ongoing_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('Employee Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# We will now explore the relationship of each feature pairwise and their segmentation into clusters via a scatterplot. This could give some insight as to the nature of which features best segment our data.

# In[136]:


for c, combination in enumerate(itertools.combinations(ongoing_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=ongoing_info[key][combination[0]],
                                y=ongoing_info[key][combination[1]],
                                hue=ongoing_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][ongoing_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][ongoing_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# In[137]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(ongoing_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = ongoing_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')

    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

for n, cluster in enumerate(cluster_info['normalized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

plt.show()
plt.close()


# As seen, the clearer segments and the most well balanced are within the normalized data. The original dataset can't be used in this situation as we mix different magnitude variables such as 'MontlyIncome' with others, for instance, therefore we cannot read all variables in each cluster for this dataset. As all other approaches, we still see many variables that are not informative and can bias the groups for this segment, thus, we are going to do an improved analyzis without these features. Features like: 'EnvironmentSatisfaction', 'Overtime', 'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TrainingTimesLastYear', 'DailyRate', 'HourlyRate', 'MonthlyRate' and the two transformed ratio variables can be seen as uninformative for our problem (like we can see from the graph and with previous descriptive work) and so we are going to cluster again without them.

# <a class="anchor" id="8.1.2-improved-approach"></a>
# ### Improved Approach

# <a class="anchor" id="8.1.2.1-correlation-analysis"></a>
# #### Correlation Analysis

# In[138]:


# Create a dictionary to hold the ongoing perspective for each dataset (original, standardized, and normalized).
ongoing_info = {}

for key, dataset in datasets.items():
    
    sub_df = dataset.copy()

    # Select columns for the ongoing perspective.
    ongoing_columns = ['JobInvolvement',
                                        'JobSatisfaction',
                                        'RelationshipSatisfaction',
                                        'MonthlyIncome',
                                        'YearsSinceLastPromotion',
                                        'WorkLifeBalance'
                                       ]

    # Store perspectives into general dictionary.
    ongoing_info[key] = sub_df[ongoing_columns]


# In[139]:


for key in ongoing_info.keys():
        
    # Initialize a new figure.
    fig, ax = plt.subplots(figsize=(18, 10))

    # Draw heatmap with correlation coefficients, set scale, and square cells. Because many of our features have non-normal distributions,
    # and in order to get a more accurate correlation metric, we will use the the non-parametric Spearman correlation coefficient.
    sns.heatmap(ongoing_info[key].corr(method='spearman'),
                         annot=True,
                         annot_kws={'fontsize': 15},
                         vmin=-1,
                         vmax=1,
                         square=True,
                         fmt='.0%'
                        )

    # Format axis elements.
    ax.set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', size=26, weight='bold')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, size=20)
    ax.set_xticks(ax.get_xticks()-0.2)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0, size=20)
    ax.tick_params(axis='both', bottom=False, left=False)


# <a class="anchor" id="8.1.2.2-hierarchical-clustering"></a>
# #### Hierarchical Clustering

# In[140]:


# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

for n, key in enumerate(ongoing_info.keys()):

    # Calculate the Z matrix for Ward distance using Euclidean distance.
    z_matrix = linkage(y=ongoing_info[key], method='ward', metric='euclidean')

    # Draw dendrogram.
    # (without any leaf labels - given the fact that we have 2500 elements, this greatly increases plotting speed: from ~20mins to ~20secs).
    dendrogram(z_matrix, no_labels=True, ax=axes[n])

    # Format axis elements.
    axes[n].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
    axes[0].set_ylabel('Distance (Ward)', size=14)
    
plt.show()
plt.close()


# As before, we try to understand the effect of the distinct scaling methods through the ongoing segment. 
# 
# From observing the above dendrograms, we found the following number of clusters:
# <br> 
#  - original data: 2
# <br>
#  - standardized data: 3
# <br> 
#  - normalized data: 4

# <a class="anchor" id="8.1.2.3-k-means"></a>
# #### K-Means

# In[141]:


# Select the range of clusters we wan to attempt.
nclusters = range(1, 7)

# Initialize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for n, key in enumerate(ongoing_info.keys()):

    inertias = []

    for k in nclusters:

        # Define the type of model we want and feed it a cluster number.
        model = KMeans(n_clusters=k, random_state=28)

        # Fit the model to our data.
        model.fit(ongoing_info[key])

        # Save the model's inertia (aka, within-cluster sum-of-squares) for the selected amount of clusters.
        inertias.append(model.inertia_)

    # Plot inertias against number of clusters.
    axes[n].plot(nclusters, inertias)
    
    # Format axis elements.
    axes[n].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=16, weight='bold')
    axes[n].set_xlabel('Number of Clusters', size=14)
    axes[n].set_ylabel('Inertia', size=14)
    axes[n].set_xticks(nclusters)

plt.show()
plt.close()


# In[142]:


# Define the number of clusters for K-Means implementation with each dataset.
cluster_number = {'original': 2, 'standardized': 3, 'normalized': 3}

# Set a dictionary to save the coordinates of the centroid for all clusters in each dataset.
cluster_centroids = {'original': 0, 'standardized': 0, 'normalized': 0}

for key in cluster_number.keys():

    model = KMeans(n_clusters=cluster_number[key], random_state=28)

    model.fit(ongoing_info[key])

    # Get the cluster to which each observation belongs to.
    ongoing_info[key]['cluster'] = model.labels_

    # Get the centroid coordinates of each cluster.
    cluster_centroids[key] = model.cluster_centers_


# In[143]:


# Get descriptive statistics for each feature of each cluster.
for key in ongoing_info.keys():
    print('\n', key.capitalize() + ' Data', '\n')
    for column in ongoing_info[key].columns[:-1]:
        print(ongoing_info[key][[column, 'cluster']].groupby(['cluster']).describe().transpose(), '\n')


# In[144]:


# Separate description table above based on its clusters.
cluster_info = {key : [] for key in ongoing_info.keys()}

for key in ongoing_info.keys():
    
    # Get number of clusters.
    n_clusters = ongoing_info[key]['cluster'].nunique()

    for cluster_number in range(n_clusters):
        
        # Get observations belonging to a given cluster.
        cluster = ongoing_info[key][ongoing_info[key]['cluster']==cluster_number]

        # Save those cluster-specific obervations in our dictionary.
        cluster_info[key].append(cluster)


# In[145]:


for column in ongoing_columns:
    
    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    
    for k, key in enumerate(cluster_info.keys()):
        
        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='RdYlGn', n_colors=n_clusters)

        for n in range(n_clusters):
            
            # Draw current feature distribution for each cluster.
            sns.distplot(cluster_info[key][n][[column, 'cluster']][cluster_info[key][n]['cluster']==n],
                               color=current_palette[n],
                               label=column + ' - Cluster ' + str(n),
                               kde=False,
                               ax=axes[k]
                              )

        # Draw current feature distribution for the whole ongoing base.
        sns.distplot(ongoing_info[key][column],
                           color='skyblue',
                           label=column + ' - Sample',
                           kde=False,
                           ax=axes[k]
                          )
        
        # Format axis elements.
        axes[k].set_title(key.capitalize() + ' Data', y=1.02, size=20)
        axes[k].set_xlabel('')
        legend_labels = ['Cluster ' + str(i) for i in range(n_clusters)]
        legend_labels.append('Sample')
        axes[k].legend(labels=legend_labels, bbox_to_anchor=(1.01, 1.0))

    # Format more axis elements.
    axes[0].set_ylabel('ongoing Count', size=16)
    plt.suptitle(column, y=1.05, size=32, weight='bold')
    
    plt.show()
    plt.close()


# In[146]:


for c, combination in enumerate(itertools.combinations(ongoing_columns, 2)):

    # Initialize a new figure.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    for k, key in enumerate(cluster_info.keys()):

        # Get number of clusters.
        n_clusters = len(cluster_info[key])
        
        # Set current color palette.
        current_palette = sns.color_palette(palette='tab10', n_colors=n_clusters)

        # Draw scatterplot of the observations for the current pair of features.
        sns.scatterplot(x=ongoing_info[key][combination[0]],
                                y=ongoing_info[key][combination[1]],
                                hue=ongoing_info[key]['cluster'],
                                palette=current_palette,
                                ax=axes[k]
                               )
        
        # Draw the centroid of each cluster.
        for n in range(n_clusters):
            sns.scatterplot(x=[cluster_centroids[key][n][ongoing_columns.index(combination[0])]],
                                    y=[cluster_centroids[key][n][ongoing_columns.index(combination[1])]],
                                    marker='X',
                                    color=current_palette[n],
                                    edgecolor='black',
                                    linewidth=2,
                                    s=200,
                                    ax=axes[k]
                                   )

        # Format axis elements.
        axes[k].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', y=1.00, size=20, weight='bold')
        axes[k].set_xlabel(combination[0], size=16)
        axes[k].set_ylabel(combination[1], size=16)

    plt.show()
    plt.close()


# By reducing the number of clusters and removing uninformative features, we can now observe a much better cluster resolution across all scaling datasets.

# In[147]:


# Initiallize a new figure.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

for n, key in enumerate(ongoing_info.keys()):

    # Aggregate our dataset by the mean of each feature per cluster.
    cluster_groups = ongoing_info[key].groupby(by='cluster', as_index=False).mean()

    # Melt the DataFrame so we can compare, for each cluster, the mean value of each feature.
    cluster_melt = pd.melt(frame=cluster_groups, id_vars='cluster',  var_name='features', value_name='value')
    
    # Draw barplots for all features of each cluster.
    sns.barplot(x='cluster',
                      y='value',
                      hue='features',
                      data=cluster_melt,
                      palette='Set2',
                      ax=axes[n]
                     )

    # Format axis elements.
    axes[n].set_title('ongoing Segmentation:\n' + key.capitalize() + ' Data', size=20, weight='bold')
    axes[n].set_xlabel('Cluster', size=18)
    axes[n].set_ylabel('')

for n, cluster in enumerate(cluster_info['standardized']):
    cluster_size = len(cluster)
    cluster_proportion = round(cluster_size / len(df) * 100, 2)
    print('Cluster ' + str(n) + ': ' + str(cluster_size) + '(' + str(cluster_proportion) + '%)')

# Format more axis elements.
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend(bbox_to_anchor=(1, 1.0))

plt.show()

plt.close()


# This time, the cluster dataset which yields better groups is the standardized one (with z-score normalization), as we can distinguish better groups and have better perspectives at some key features like the 'MonthlyIncome'.
# 
# <br>**NOTE:** The cluster numbering can differ between the following text and that represented in the figure above, due to the fact that distinct initial seeds can lead to different cluster assignment. Nevertheless, the characterizations that follow are always consistent with the segmentations obtained.
# 
# <br>Based on the observations made from the segmentation of standardized samples, we can describe the following ongoing segments:
# 
# <br>**Cluster0 (Wealthy Stable)**: This cluster is composed with 259 (18%) employees. It's the least populated group and it includes the workers that have the highest income per month with a medium relationship with their colleagues. Its work life balance its a lit bit better than the average but the highest differentiating element is the years that they stayed in the same job role. This makes sense as the majority of this segment is director or has a higher position in the hierarchy and stayed in the same position for the last years, leading to a higher income on average.
# 
# <br>**Cluster1 (Poor but nice)**: This group represents 49% (n=714) of the workers from the sample and it is characterized as being the ones with the lowest income (close to the next cluster). The strongest point here is the relationship with coworkers (measured by 'RelationshipSatisfaction') which is clearly above average. They have the lowest time in years since last promotion as well, so probably this is the group with the youngest workforce.
# 
# <br>**Cluster2 (Unpleasant Coworkers)**: This cluster is composed with 34% (n=497) employees. This group comprises workers with the least satisfaction in terms of relationship with team mates. In terms of income, they, on average, earn a little bit more than the previous group but still not much. In terms of years since last promotion, it's defined to lower values as well, so probably we see various types of employees in this group that are included in other mixed clusters.

# <a class="anchor" id="concluding-remarks"></a>
# ## Concluding Remarks - Clustering

# In[148]:


# Create a DataFrame with cluster information from all four perspectives for each employee.
cluster_df = pd.concat([employee_info['original']['cluster'], job_info['normalized']['cluster'],                         historic_info['normalized']['cluster'], ongoing_info['standardized']['cluster']], axis=1)
cluster_df.columns = ['employee_cluster', 'job_cluster', 'historic_cluster', 'ongoing_cluster']
cluster_df


# In[149]:


#Now merge with the original dataset to have the Attrition column
cluster_df_total = cluster_df.join(hr0['Attrition'])
cluster_df_total


# In[150]:


# Initialiaze a new figure.
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

# Set labels for heatmap ticks.
labels = {'employee': {0: 'Old', 1: 'Medium-Aged', 2: 'Young'},
               'job': {0: 'Experienced Researcher', 1: 'Rookie Researcher', 2: 'Sales'},
               'historic': {0: 'Senior Loyal', 1: 'Frequent Changing', 2: 'Work Beginners'},
               'ongoing': {0: 'Rich Stable', 1: 'Empathetic', 2: 'Unpleasant'}
              }

# Set list of perspective sto iterate through.
perspective_list = [name.split('_')[0] for name in cluster_df.columns]
for n, combination in enumerate(itertools.combinations(perspective_list, 2)):
    # Create an empty numpy array
    perspectives_array = np.zeros(shape=(cluster_df[combination[1] + '_cluster'].nunique(), cluster_df[combination[0] + '_cluster'].nunique()))

    # Get count of customers that fall within each cluster combination and store that count in our empty array.
    for p in range(cluster_df[combination[0] + '_cluster'].nunique()):
        for c in range(cluster_df[combination[1] + '_cluster'].nunique()):
            temp_df = cluster_df.query(combination[0] + '_cluster == ' + str(p) + ' and ' + combination[1] + '_cluster == ' + str(c))
            n_observations = len(temp_df)
            perspectives_array[c][p] = n_observations

    perspectives_array = perspectives_array.astype(int)
    

    # Plot heatmap representing the proportion of workers that fall within each cluster combination.

    sns.heatmap(perspectives_array,
                         annot=True,
                         annot_kws={'fontsize': 15},
                         fmt='.0f',
                         cmap='RdYlGn_r',
                         square=True,
                         linewidths=0.5,
                         cbar=False,
                         ax=axes[int(n/3),n%3]
                        )

    # Format axis elements.
    xlabel = combination[0].capitalize() + ' Group'
    axes[int(n/3),n%3].set_xlabel(xlabel, size=12, labelpad=15, weight='bold')
    axes[int(n/3),n%3].set_xticklabels(pd.Series(cluster_df[combination[0] + '_cluster'].unique()).index.map(labels[combination[0]]), size=9, rotation=90)
    ylabel = combination[1].capitalize() + ' Group'
    axes[int(n/3),n%3].set_ylabel(ylabel, size=12, labelpad=15, weight ='bold')
    axes[int(n/3),n%3].set_yticklabels(pd.Series(cluster_df[combination[1] + '_cluster'].unique()).index.map(labels[combination[1]]), size=9, rotation=0)
    axes[int(n/3),n%3].tick_params(axis='both', length=0)
          
plt.suptitle('Cluster Relevance', size=24, weight='bold')

# Adjust subplot spacing.
plt.subplots_adjust(wspace=0.5, hspace=0.6)

plt.show()
plt.close()


# In[151]:


#Now let's cluster the most important groups for employees that left (attrition = 1)

cluster_df_total_attrition = cluster_df_total[cluster_df_total['Attrition']==1]
cluster_df_attrition = cluster_df_total_attrition.drop(columns=['Attrition'])
cluster_df_attrition


# In[152]:


#see the most frequent of all cluster combinations
df1 = cluster_df_attrition.groupby(["employee_cluster", "job_cluster","historic_cluster","ongoing_cluster"]).size().reset_index(name='count')
df1.sort_values('count', ascending = False)


# In[153]:


# Initialiaze a new figure.
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

# Set labels for heatmap ticks.
labels = {'employee': {0: 'Old', 1: 'Medium-Aged', 2: 'Young'},
               'job': {0: 'Experienced Researcher', 1: 'Rookie Researcher', 2: 'Sales'},
               'historic': {0: 'Senior Loyal', 1: 'Frequent Changing', 2: 'Work Beginners'},
               'ongoing': {0: 'Rich Stable', 1: 'Empathetic', 2: 'Unpleasant'}
              }

# Set list of perspective sto iterate through.
perspective_list = [name.split('_')[0] for name in cluster_df_attrition.columns]
for n, combination in enumerate(itertools.combinations(perspective_list, 2)):
    # Create an empty numpy array
    perspectives_array = np.zeros(shape=(cluster_df_attrition[combination[1] + '_cluster'].nunique(), cluster_df_attrition[combination[0] + '_cluster'].nunique()))

    # Get count of customers that fall within each cluster combination and store that count in our empty array.
    for p in range(cluster_df_attrition[combination[0] + '_cluster'].nunique()):
        for c in range(cluster_df_attrition[combination[1] + '_cluster'].nunique()):
            temp_df = cluster_df_attrition.query(combination[0] + '_cluster == ' + str(p) + ' and ' + combination[1] + '_cluster == ' + str(c))
            n_observations = len(temp_df)
            perspectives_array[c][p] = n_observations

    perspectives_array = perspectives_array.astype(int)
    

    # Plot heatmap representing the proportion of workers that fall within each cluster combination.

    sns.heatmap(perspectives_array,
                         annot=True,
                         annot_kws={'fontsize': 15},
                         fmt='.0f',
                         cmap='RdYlGn_r',
                         square=True,
                         linewidths=0.5,
                         cbar=False,
                         ax=axes[int(n/3),n%3]
                        )

    # Format axis elements.
    xlabel = combination[0].capitalize() + ' Group'
    axes[int(n/3),n%3].set_xlabel(xlabel, size=12, labelpad=15, weight='bold')
    axes[int(n/3),n%3].set_xticklabels(pd.Series(cluster_df_attrition[combination[0] + '_cluster'].unique()).index.map(labels[combination[0]]), size=9, rotation=90)
    ylabel = combination[1].capitalize() + ' Group'
    axes[int(n/3),n%3].set_ylabel(ylabel, size=12, labelpad=15, weight ='bold')
    axes[int(n/3),n%3].set_yticklabels(pd.Series(cluster_df_attrition[combination[1] + '_cluster'].unique()).index.map(labels[combination[1]]), size=9, rotation=0)
    axes[int(n/3),n%3].tick_params(axis='both', length=0)
          
plt.suptitle('Cluster Relevance', size=24, weight='bold')

# Adjust subplot spacing.
plt.subplots_adjust(wspace=0.5, hspace=0.6)

plt.show()
plt.close()


# For the most important segments who have are leaving the company, we have the following most relevant groups:
# 
# <br>**Beginners** - This segment is composed by the youngest employees, which tipically live close to medium distances to the company. It's mainly their first job and therefore, its monthly income is tipically low. In terms of relationship with colleagues it's actually positive for the majority of cases, however, they usually see this job as a beginning of career only.
# 
# <br>**High Turnover Researcher** - This group of workers belong mainly to the Research & Development department and it's a group that already worked in previous jobs before as they are frequent changers. They are older and experienced already in this role with medium monthly income. Their total years at this company is low as well as with the current manager, meaning that they like change and it's more difficult to retain them.
# 
# <br>**Tough Personality** - This is the group that is most likely to leave due to their inadaptability to the team leading to bad relationships between peers (and probably the manager as well). They usually fit in an younger region, they live relatively close to work and we see more sales persons. However, their frequent turnover historic and their unempathetic attitude towards others is what will lead them to head out this firm.
# 
# <br>**Remaining employees** - This is the "unkwon" group meaning that is composed by employees that will leave the company but they don't relate to any of the previous profiles.

# In[ ]:





# # Model Building

# ## Recursive Feature Elimination

# ### One Hot Encoding of Categorical Variables

# In[84]:


department=pd.get_dummies(hr0["Department"],prefix='Dep_',drop_first=True)


# In[85]:


EducationField=pd.get_dummies(hr0["EducationField"],prefix='Educ_',drop_first=True)


# In[86]:


JobRole=pd.get_dummies(hr0["JobRole"],prefix='JobRole_',drop_first=True)


# In[87]:


MS3=pd.get_dummies(hr0["MaritalStatus"],prefix='MS_',drop_first=True)


# In[88]:


hr0=hr0.drop(['EmployeeNumber'], axis = 1)


# In[89]:


hr0.head(1)


# In[90]:


hr0=pd.merge(department, hr0, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')
hr0=pd.merge(JobRole, hr0, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')
hr0=pd.merge(EducationField, hr0, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')
hr0=pd.merge(MS3, hr0, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')


# ### Removal of non numeric variables

# In[91]:


# Selection of numerical columns
hr0sel = hr0.drop(['Attrition', 'BusinessTravel','Department','EducationField','JobRole',
                   'MaritalStatus','PercentSalaryHike_Bracket','TotalWorkingYears_Bracket',
                  'YearsAtCompany_Bracket','Age_Bracket','YearsAtCompany','JobLevel','IncHourlyRate'], axis = 1).copy()

cols = hr0sel.columns.tolist()

len(cols)


# ### RFE Function and Analysis

# In[92]:


#Function definition to present the number of features that returns the best accuracy for a given model
def RFE_optimalnof(model,ncols,dados,target, tsize):
    nof_list=np.arange(1,ncols)
    high_score=0
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    for n in range(len(nof_list)):

        X_train, X_test, y_train, y_test = train_test_split(dados,target, test_size = tsize, random_state = 0)    
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)

        score = model.score(X_test_rfe,y_test)
        score_list.append(score)

        if(score>high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features of model\033[1m", type(model).__name__ , "\033[0m, test_size\033[1m", tsize,"\033[0m: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    return nof, high_score


# In[93]:


#Initializing the models
ncols=len(cols)
modelLR=LogisticRegression()
modelRFT=RandomForestClassifier()
modelDT=DecisionTreeClassifier()


# In[94]:


#Sets multiples cut-oof percentages to divide between train and test datasets
lista_test_sizes=[0.1,0.2,0.3]


# In[95]:


#Separates between features and target datasets
dados=hr0sel[cols]
target=hr0.Attrition


# ### Data _as is_ - data before it is normalized

# In[239]:


get_ipython().run_cell_magic('time', '', 'numbLR=[]\nfor a in lista_test_sizes:\n    numbLR.append(RFE_optimalnof(modelLR,ncols,dados,target,a))')


# In[240]:


get_ipython().run_cell_magic('time', '', 'numbRFT=[]\nfor a in lista_test_sizes:\n    numbRFT.append(RFE_optimalnof(modelRFT,ncols,dados,target,a))')


# In[241]:


get_ipython().run_cell_magic('time', '', 'numbDT=[]\nfor a in lista_test_sizes:\n    numbDT.append(RFE_optimalnof(modelDT,ncols,dados,target,a))')


# ### Data Robust Scaled

# In[242]:


## Scaled Dataset
robust = RobustScaler().fit(dados)
dados_Robust=robust.transform(dados)
dados_Robust = pd.DataFrame(dados_Robust, columns = dados.columns) 
numbLRR=[]


# In[243]:


get_ipython().run_cell_magic('time', '', 'for a in lista_test_sizes:\n    numbLRR.append(RFE_optimalnof(modelLR,ncols,dados_Robust,target,a))')


# In[244]:


get_ipython().run_cell_magic('time', '', 'numbRFTR=[]\nfor a in lista_test_sizes:\n    numbRFTR.append(RFE_optimalnof(modelRFT,ncols,dados_Robust,target,a))')


# In[245]:


get_ipython().run_cell_magic('time', '', 'numbDTR=[]\nfor a in lista_test_sizes:\n    numbDTR.append(RFE_optimalnof(modelDT,ncols,dados_Robust,target,a))')


# ### Data MinMax Scaled 

# In[246]:


## Scaled Dataset
scaled = MinMaxScaler(feature_range=(-1,1)).fit(dados)
dados_Scaled=scaled.transform(dados)
dados_Scaled = pd.DataFrame(dados_Scaled, columns = dados.columns) 


# In[247]:


get_ipython().run_cell_magic('time', '', 'numbLRS=[]\nfor a in lista_test_sizes:\n    numbLRS.append(RFE_optimalnof(modelLR,ncols,dados_Scaled,target,a))')


# In[248]:


get_ipython().run_cell_magic('time', '', 'numbRFTS=[]\nfor a in lista_test_sizes:\n    numbRFTS.append(RFE_optimalnof(modelRFT,ncols,dados_Scaled,target,a))')


# In[249]:


get_ipython().run_cell_magic('time', '', 'numbDTS=[]\nfor a in lista_test_sizes:\n    numbDTS.append(RFE_optimalnof(modelDT,ncols,dados_Scaled,target,a))')


# ### Summary
# 
# #### Logistic Regression

# In[250]:


get_ipython().run_cell_magic('time', '', "dictLR = {'AsIs': numbLR, 'RobustScale': numbLRR, 'MinMaxScale': numbLRS} \n \ndf_LogisticReg = pd.DataFrame(dictLR, index = ['0.1','0.2','0.3'])    \ndf_LogisticReg")


# #### Random Forest

# In[251]:


dictRF = {'AsIs': numbRFT, 'RobustScale': numbRFTR, 'MinMaxScale': numbRFTS} 
 
df_RandomForest = pd.DataFrame(dictRF, index = ['0.1','0.2','0.3'])    
df_RandomForest


# #### Decision Tree

# In[252]:


dictDT = {'AsIs': numbDT, 'RobustScale': numbDTR, 'MinMaxScale': numbDTS} 
 
df_DecisionTree = pd.DataFrame(dictDT, index = ['0.1','0.2','0.3'])    
df_DecisionTree


# In[253]:


#Defining function to plot the most important features for a decision tree model.
#This function allows us to see waht the actual features are, by order of importance, not just the number of ideal features
def DT_optimalnof(model, dados,target,tsize):
    X_train, X_test, y_train, y_test = train_test_split(dados, target, test_size = tsize, random_state = 0, stratify = target)
    n_features = X_train.shape[1]
    model.fit(X_train, y_train)
    plt.figure(figsize=(20,10))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


# In[254]:


#Applying previously defined function to a dataset with 10% test size
DT_optimalnof(modelDT,dados,target,0.1)


# ## Variable Selection

# Based on the previous analysis, we will generate 3 types of datasets, two normalized with a Robust Scaler and one with a MinMax Scaler, with varying number of features, to be tested against multiple models. In this phase we are not able to commit to a set number of features - sizes of 47, 28 and 7 features were selected.

# In[112]:


#Dataset with the best 47 features, with Robust Scaled features
rfe = RFE(estimator = modelLR, n_features_to_select = 47)
X_rfe = rfe.fit_transform(X = dados_Robust, y = target) 
modelLR.fit(X = X_rfe,y = target)
temp = pd.Series(rfe.support_, index = dados_Robust.columns)
selected_features_14= temp[temp==True].index
selected_features_14


# In[113]:


dr=dados_Robust[selected_features_14].copy()
dr.head()


# In[114]:


#Dataset with the best 28 features, with MinMax Scaled features
rfe = RFE(estimator = modelDT, n_features_to_select = 28)
X_rfe = rfe.fit_transform(X = dados_Scaled, y = target) 
modelRFT.fit(X = X_rfe,y = target)
temp = pd.Series(rfe.support_, index = dados_Scaled.columns)
selected_features_mm= temp[temp==True].index
selected_features_mm


# In[115]:


ds=dados_Scaled[selected_features_mm].copy()
ds.head()


# In[116]:


#Dataset with the best 7 features, with no normalization
rfe = RFE(estimator = modelDT, n_features_to_select = 7)
X_rfe = rfe.fit_transform(X = dados, y = target) 
modelDT.fit(X = X_rfe,y = target)
temp = pd.Series(rfe.support_, index = dados.columns)
selected_features_6= temp[temp==True].index
selected_features_6


# In[117]:


das=dados[selected_features_6].copy()
das.head()


# ## Model Definition
# 

# In this step, our models will be created and defined. We will define the parameters of some models by optimizing them to our chosen datasets. For that, we first have to split them into training and test datasets, so we can measure the accuracy of the models and choose the best parameters based on that.

# In[118]:


X14_train, X14_test, y14_train, y14_test = train_test_split(
    dr,target,test_size = 0.3, random_state = 0, shuffle = True, stratify = target)


# In[119]:


Xs12_train, Xs12_test, ys12_train, ys12_test = train_test_split(
    ds,target,test_size = 0.3, random_state = 0, shuffle = True, stratify = target)


# In[120]:


Xas_train, Xas_test, yas_train, yas_test = train_test_split(
    das,target,test_size = 0.1, random_state = 0, shuffle = True, stratify = target)


# ### Naive Bayes:

# In[121]:


#Naive Bayes
modelNB=GaussianNB()


# ### Random Forest Classifier

# In[122]:


modelRFT1=RandomForestClassifier()


# ### Decision Tree
# 
# Three decision tree models will be defined, with parameters optimized for each of the three datasets, with the help of the function GridSearchCv.

# In[123]:


parameter_space_DT = {
    'criterion': ['gini','entropy'],
    'splitter': ['random', 'best'],
    'max_depth': [2,4,8],
    'min_samples_split': [2,10,100],
    'min_samples_leaf': [2,10,100],
    'min_weight_fraction_leaf': [0,0.15],
    'max_leaf_nodes': [5,10,10000],
    'min_impurity_decrease':[0,0.02]
}


# In[124]:


model = DecisionTreeClassifier()
DTC_param6 = GridSearchCV(model, parameter_space_DT)


# In[125]:


DTC_param6.fit(Xas_train, yas_train)


# In[126]:


params6=DTC_param6.best_params_
params6


# In[127]:


model_DTCas=DecisionTreeClassifier(criterion= 'gini',
                                max_depth= 2,
                                max_leaf_nodes= 10,
                                min_impurity_decrease= 0,
                                min_samples_leaf= 10,
                                min_samples_split= 10,
                                min_weight_fraction_leaf= 0,
                                splitter= 'random')


# In[128]:


DTC_param12 = GridSearchCV(model, parameter_space_DT)
DTC_param12.fit(Xs12_train, ys12_train)
params12=DTC_param12.best_params_
params12


# In[129]:


#parâmetros selecionados
model_DTCs=DecisionTreeClassifier(criterion= 'gini',
                                max_depth= 2,
                                max_leaf_nodes= 10,
                                min_impurity_decrease= 0,
                                min_samples_leaf= 2,
                                min_samples_split= 10,
                                min_weight_fraction_leaf= 0,
                                splitter= 'random')


# In[130]:


DTC_param14 = GridSearchCV(model, parameter_space_DT)
DTC_param14.fit(X14_train, y14_train)
params14=DTC_param14.best_params_
params14


# In[131]:


model_DTCr=DecisionTreeClassifier(criterion= 'entropy',
                                max_depth= 10,
                                max_leaf_nodes= 5,
                                min_impurity_decrease= 0,
                                min_samples_leaf= 10,
                                min_samples_split= 2,
                                min_weight_fraction_leaf= 0,
                                splitter= 'best')


# ### Neural Networks
# 
# Three neural networks models will be defined, with parameters optimized for each of the three datasets, with the help of the function GridSearchCv.

# In[132]:


parameter_space = {
    'hidden_layer_sizes': [(25,25,25), (50,),(10,)],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['lbfgs','sgd', 'adam'],
    'learning_rate_init': (0.00001,0.001,0.1),
    'learning_rate': ['constant','adaptive']
}


# In[133]:


modelNNC=MLPClassifier()


# In[134]:


BP_NNC14= GridSearchCV(modelNNC, parameter_space)


# In[255]:


BP_NNC14.fit(X14_train, y14_train)
params14_NNC=BP_NNC14.best_params_
params14_NNC


# In[136]:


model_NNr=MLPClassifier(activation= 'tanh',
                        hidden_layer_sizes= (10),
                        learning_rate= 'constant',
                        learning_rate_init= 0.001,
                        solver= 'adam')


# In[137]:


BP_NNC12= GridSearchCV(modelNNC, parameter_space)


# In[256]:


BP_NNC12.fit(Xs12_train, ys12_train)
params12_NNC=BP_NNC12.best_params_
params12_NNC


# In[139]:


model_NNs=MLPClassifier(activation= 'logistic',
                        hidden_layer_sizes= (50,),
                        learning_rate= 'adaptive',
                        learning_rate_init= 0.1,
                        solver= 'sgd')


# ## Model Evaluation

# In[141]:


#Defining a function to score the models with the following parameters: confusion matrix, accuracy score,
#precision, recall, F1-score, and a report that summarizes these values per the predictions

def score_model(model,X_train,y_train, X_val, y_val):
    model=model.fit(X_train,y_train)
    y_pred = model.predict(X_val)
    output_dict=dict()
    
    cm=confusion_matrix(y_val, y_pred)
    acc_score= accuracy_score(y_val, y_pred)
    precision=precision_score(y_val, y_pred)
    recall=recall_score(y_val, y_pred)
    f1=f1_score(y_val, y_pred)
    report=classification_report(y_val, y_pred)
    output_dict['cm']=cm
    output_dict['acc_score']=acc_score
    output_dict['precision']= precision
    output_dict['recall']= recall
    output_dict['f1']= f1
    output_dict['report']= report
    return output_dict


# In[142]:


#Defining a function to print the ROC curve and calculate the area under it.

def AUC_ROC_curve(model, X_train, y_train, X_val, y_val,target):
    model.fit(X_train, y_train)

    lr_probs = model.predict_proba(X_val)
    ns_probs = [0 for _ in range(len(y_val))]
    lr_probs = lr_probs[:, 1]

    ns_auc = roc_auc_score(y_val, ns_probs)
    lr_auc = roc_auc_score(y_val, lr_probs)
    
    print(type(model).__name__,': ROC AUC=%.3f' % (lr_auc))
    ns_fpr, ns_tpr, _ = roc_curve(y_val, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_val, lr_probs)

    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sem modelo')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label=type(model).__name__)
    pyplot.xlabel('FPR')
    pyplot.ylabel('TPR')
    pyplot.legend()
    pyplot.show()


# In[149]:


das.name = 'Dataset não normalizado'
ds.name = 'Dataset normalizado com MinMaxScaler'
dr.name = 'Dataset  normalizado com RobustScaler'
datasets=[das,ds,dr]
results=[]

#Selected models
models=[modelLR,model_DTCas,model_DTCs,model_DTCr,model_NNr,model_NNs,modelNB,modelRFT1,gradient_booster]
dict_score=dict()
for a in datasets:
    for m in models:
        X_train, X_val, y_train, y_val = train_test_split(a,target,test_size = 0.3, random_state = 0, shuffle = True, stratify = target)
        print('\033[1mCurrent dataset\033[0m:', a.name)
        print('\033[1mCurrent model\033[0m:', m,'\n\n')
        dict_score=score_model(m,X_train,y_train, X_val, y_val)
        results.append(dict_score)
        AUC_ROC_curve(m, X_train, y_train, X_val, y_val,target)


# Based on the ROC curve results, we're selecting the following models: Gradient Boosting, Neural Networks (MLPClassifier) and the Logistic Regression.

# ## Fine Tuning
# 

# In[49]:


from sklearn.metrics import confusion_matrix,classification_report

#Defining function to determine model scores (accuracy, recall, precision and confusion matrix)
def plot_cm(model,X_train,y_train, X_val, y_val):
    model=model.fit(X_train,y_train)
    y_pred = model.predict(X_val)
    #confusion matrix
    cm = confusion_matrix(y_val, y_pred) 
    cm_df = pd.DataFrame(cm, index = ['No Attrition','Attrition'], columns = ['No Attrition','Attrition'])
    
    # Plot
    plt.figure(figsize=(4,4))
    ax = sns.heatmap(cm_df, annot=True, cmap = 'Reds', fmt='g', cbar=False, linewidth = 1, annot_kws={"fontsize":13}) 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.yticks(va="center")
    plt.show()
    
    print(classification_report(y_val, y_pred, target_names=['No Attrition','Attrition']))


# Now, we will try to improve our 3 best models (Neural Networks, Logistic Regression and Gradient Boosting) by fine tuning each model with the best hyperparameters for each of them, in order to better predict attrition without being overfitted to our data.<br>
# We will use only the Robust scaled dataset for each one of them, as it yielded the best results for now.

# ### Gradient Boosting

# In[43]:


import numpy as np 
np.random.seed(0)


# In[111]:



parameters_gb = {
    "loss":["deviance"],
    "learning_rate": [0.1],
    "min_samples_split": range(1,10,1),
    "min_samples_leaf": [1],
    "max_depth":range(1,10,1),
    "max_features":["log2"],
    "criterion": ["friedman_mse"],
    "subsample":[0.5, 0.7, 0.9, 1.0],
    "n_estimators":range(60,200,20)
    }


# In[112]:


modelGB = GradientBoostingClassifier()
GB_param13 = GridSearchCV(modelGB, parameters_gb)


# In[114]:


GB_param13.fit(X14_train, y14_train) 


# In[115]:


# Best parameter set
print('------------------------------------------------------------------------------------------------------------------------')
print('Best parameters found:\n', GB_param13.best_params_)
print('------------------------------------------------------------------------------------------------------------------------')

# All results
means = GB_param13.cv_results_['mean_test_score']
stds = GB_param13.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, GB_param13.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std , params))


# In[175]:


#best tuned model
modelGB2 = GradientBoostingClassifier(criterion = 'friedman_mse',
 learning_rate = 0.1,
 loss = 'deviance',
 max_depth = 2,
 max_features = 'log2',
 min_samples_leaf = 2,
 min_samples_split = 10,
 n_estimators = 120,
 subsample = 0.9,
 random_state = 15) 


# In[176]:


dict_score=dict()
X_train, X_val, y_train, y_val = train_test_split(dr,target,test_size = 0.3, random_state = 0, shuffle = True, stratify = target)
dict_score=score_model(modelGB2,X_train,y_train, X_val, y_val)
results.append(dict_score)
AUC_ROC_curve(modelGB2, X_train, y_train, X_val, y_val,target)
plot_cm(modelGB2, X_train, y_train, X_val, y_val)


# ### Neural Networks

# In[187]:


import numpy as np 
np.random.seed(0)


# In[368]:



parameters_nn = {
    'hidden_layer_sizes': [(10,10,10), (50,50), (100,100), (100,)],
    'activation': ['tanh', 'relu'],
    'max_iter': [100, 200],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate_init': list(np.linspace(0.00001,0.1,5)),
    'learning_rate': ['constant','adaptive']
}


# In[369]:


modelNN = MLPClassifier()
NN_param13 = GridSearchCV(modelNN, parameters_nn)


# In[370]:


NN_param13.fit(X14_train, y14_train) 


# In[371]:


# Best parameter set
print('------------------------------------------------------------------------------------------------------------------------')
print('Best parameters found:\n', NN_param13.best_params_)
print('------------------------------------------------------------------------------------------------------------------------')

# All results
means = NN_param13.cv_results_['mean_test_score']
stds = NN_param13.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, NN_param13.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std , params))


# In[518]:


#best tuned model
modelNN2 = MLPClassifier(
activation = 'logistic',
hidden_layer_sizes = (100,),
learning_rate = 'constant',
solver = 'sgd',
max_iter = 100,
learning_rate_init = 0.025,
momentum = 1,
random_state = 15
                    ) 


# In[519]:


dict_score=dict()
X_train, X_val, y_train, y_val = train_test_split(dr,target,test_size = 0.3, random_state = 0, shuffle = True, stratify = target)
dict_score=score_model(modelNN2,X_train,y_train, X_val, y_val)
results.append(dict_score)
AUC_ROC_curve(modelNN2, X_train, y_train, X_val, y_val,target)
plot_cm(modelNN2, X_train, y_train, X_val, y_val)


# ### Logistic Regression

# In[43]:


import numpy as np 
np.random.seed(0)


# In[358]:



parameters_lr = {
    'penalty': ['l1','l2','elasticnet','none'],
    'C': np.logspace(0, 4, 10),
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }


# In[359]:


modelLR = LogisticRegression()
LR_param13 = GridSearchCV(modelLR, parameters_lr)


# In[360]:


LR_param13.fit(X14_train, y14_train) 


# In[361]:


# Best parameter set
print('------------------------------------------------------------------------------------------------------------------------')
print('Best parameters found:\n', LR_param13.best_params_)
print('------------------------------------------------------------------------------------------------------------------------')

# All results
means = LR_param13.cv_results_['mean_test_score']
stds = LR_param13.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, LR_param13.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std , params))


# In[366]:


#best tuned model
modelLR2 = LogisticRegression(
C = 0.001,
penalty = 'none',   
solver = 'saga',   
random_state = 0
) 


# In[367]:


dict_score=dict()
X_train, X_val, y_train, y_val = train_test_split(dr,target,test_size = 0.3, random_state = 0, shuffle = True, stratify = target)
dict_score=score_model(modelLR2,X_train,y_train, X_val, y_val)
results.append(dict_score)
AUC_ROC_curve(modelLR2, X_train, y_train, X_val, y_val,target)
plot_cm(modelLR2, X_train, y_train, X_val, y_val)


# # Model: 1 Year in Advance

# In this next step, we will be trying to predict attrition with one year in advance. Because of this, we have to remove variables which values might have been different one year ago.
# All the year related variables can be used - we just have to remove one year from each.

# ## Defining the New Dataset

# In[150]:


hr_1y=hr0[['Age','Attrition','Department','JobRole','EducationField','Gender','NumCompaniesWorked',
           'TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager',
           'Age_Entry','Age_Workforce','FLG_1stjob']].copy()


# In[151]:


#Remove employees who have been at the company for less than one year.
hr_1y.describe()


# In[152]:


#Remove employees who have been at the company for less than one year.
hr_1y=hr_1y.drop(hr_1y[(hr_1y['YearsAtCompany'] == 0.0)].index)


# In[153]:


hr_1y.describe()


# In[154]:


#Remove employees with YearsSinceLastPromotion=0 - we don't know the reality the year prior.
hr_1y=hr_1y.drop(hr_1y[(hr_1y['YearsSinceLastPromotion'] == 0.0)].index)


# In[155]:


hr_1y.describe()


# In[156]:


#Remove employees with YearsInCurrentRole=0 and YearsWithCurrManager=0 - we don't know the reality the year prior.
hr_1y=hr_1y.drop(hr_1y[(hr_1y['YearsInCurrentRole'] == 0.0) | (hr_1y['YearsWithCurrManager'] == 0.0)].index)


# In[157]:


hr_1y.describe()


# In[158]:


hr_1y.head(3)


# In[159]:


hr_1yb=hr_1y.copy()


# In[160]:


hr_1yb[['Age','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','Age_Entry','Age_Workforce']]=hr_1yb[['Age','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','Age_Entry','Age_Workforce']]-1


# In[161]:


hr_1yb.head()


# In[162]:


#One Hot Encoding for this dataset
department=pd.get_dummies(hr_1yb["Department"],prefix='Dep_',drop_first=True)

EducationField=pd.get_dummies(hr_1yb["EducationField"],prefix='Educ_',drop_first=True)

JobRole=pd.get_dummies(hr_1yb["JobRole"],prefix='JobRole_',drop_first=True)


# In[163]:


hr_1yb=pd.merge(department, hr_1yb, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')
hr_1yb=pd.merge(JobRole, hr_1yb, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')
hr_1yb=pd.merge(EducationField, hr_1yb, how='inner', left_on='EmployeeNumber', right_on='EmployeeNumber')


# In[164]:


#Drop non numerical varibles
hr_1yt = hr_1yb.drop(['Attrition','Department','EducationField','JobRole'], axis = 1).copy()


# ## Correlation Analysis - 1 year back model

# In[165]:


#Pearson Correlation
corrp = hr_1yt.corr()
figure = plt.figure(figsize=(14,14))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corrp, annot=True, fmt = '.1g',cmap=cmap)


# In[166]:


#Spearmen correlation
figure = plt.figure(figsize=(14,14))
cor_spearman = corrp.corr(method ='spearman')
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor_spearman, annot=True, fmt = '.1g', cmap=cmap)


# In[167]:


correlated_features(corrp)


# In[168]:


correlated_features(cor_spearman)


# In[169]:


#We will be removing the highly correlated variables 
hr_1yt = hr_1yt.drop(['Dep__Sales'], axis = 1).copy()


# In[170]:


#We will be removing the highly correlated variables 
hr_2yt=hr_1yt.drop(['Age_Entry', 'Dep__Research & Development', 'FLG_1stjob', 'TotalWorkingYears','YearsInCurrentRole', 
                    'YearsSinceLastPromotion','YearsWithCurrManager'], axis=1).copy()


# In[171]:


hr_2yt.info()


# ## Recursive Feature Elimination

# In[172]:


#Save number of variables of the 1 year back model dataset
ycols=hr_2yt.columns.tolist()
yncols=len(ycols)


#Initialize models for Recursive Feature Engineering
ymodelLR=LogisticRegression()
ymodelRFT=RandomForestClassifier()
ymodelDT=DecisionTreeClassifier()

#Separate input and target data
ydados=hr_1yb[ycols]
target_1y=hr_1yb.Attrition


# In[174]:


## Robust Scaled Dataset
robust = RobustScaler().fit(ydados)
ydados_Robust=robust.transform(ydados)
ydados_Robust = pd.DataFrame(ydados_Robust, columns = ydados.columns) 

ynumbLRR2=[]
for a in lista_test_sizes:
    ynumbLRR2.append(RFE_optimalnof(ymodelLR,yncols,ydados_Robust,target_1y,a))
    

ynumbRFTR2=[]
for a in lista_test_sizes:
    ynumbRFTR2.append(RFE_optimalnof(ymodelRFT,yncols,ydados_Robust,target_1y,a))
    
ynumbDTR2=[]
for a in lista_test_sizes:
    ynumbDTR2.append(RFE_optimalnof(ymodelDT,yncols,ydados_Robust,target_1y,a))


# Since the logistic regression always suggest 1 variable for the final dataset, we won't be considering it's scores. The scores presented are based on accuracy - this means that if the model predicts everything with 0 (non-attriction), since the zeros represent about 87% of the dataset, the accuracy will be 87% - but the model will be useless.

# In[175]:


## MinMAx Scaled Dataset
minmax = MinMaxScaler().fit(ydados)
ydadosS=minmax.transform(ydados)
ydadosS = pd.DataFrame(ydadosS, columns = ydados.columns) 

ynumbRFTS=[]
for a in lista_test_sizes:
    ynumbRFTS.append(RFE_optimalnof(ymodelRFT,yncols,ydadosS,target_1y,a))
    
ynumbDTS=[]
for a in lista_test_sizes:
    ynumbDTS.append(RFE_optimalnof(ymodelDT,yncols,ydadosS,target_1y,a))


# In[185]:


#dataset não normalizado, teste com 20% dos registos
DT_optimalnof(ymodelDT,ydados_Robust,target_1y,0.2)


# ## Variable Selection

# In[190]:


#RobustScales, with 13 variables
rfe = RFE(estimator = ymodelLR, n_features_to_select = 13)
X_rfe = rfe.fit_transform(X = ydados_Robust, y = target_1y) 
ymodelLR.fit(X = X_rfe,y = target_1y)
temp = pd.Series(rfe.support_, index = ydados_Robust.columns)
selected_features_1= temp[temp==True].index
selected_features_1


# In[191]:


yd1=ydados_Robust[selected_features_1].copy()
yd1.head()


# In[192]:


#RobustScaled, with 3 variables
rfe = RFE(estimator = ymodelLR, n_features_to_select = 3)
X_rfe = rfe.fit_transform(X = ydados_Robust, y = target_1y) 
ymodelLR.fit(X = X_rfe,y = target_1y)
temp = pd.Series(rfe.support_, index = ydados_Robust.columns)
selected_features_lg= temp[temp==True].index
selected_features_lg


# In[193]:


ydr=ydados_Robust[selected_features_lg].copy()
ydr.head()


# In[194]:


#MinMaxScaled, with 12 variables
rfe = RFE(estimator = ymodelDT, n_features_to_select = 12)
X_rfe = rfe.fit_transform(X = ydadosS, y = target_1y) 
ymodelRFT.fit(X = X_rfe,y = target_1y)
temp = pd.Series(rfe.support_, index = ydadosS.columns)
selected_features_mm= temp[temp==True].index
selected_features_mm


# In[195]:


yds=ydadosS[selected_features_mm].copy()
yds.head()


# In[196]:


#MinMaxScaled, with 6 variables
rfe = RFE(estimator = ymodelDT, n_features_to_select = 6)
X_rfe = rfe.fit_transform(X = ydadosS, y = target_1y) 
ymodelRFT.fit(X = X_rfe,y = target_1y)
temp = pd.Series(rfe.support_, index = ydadosS.columns)
selected_features_r= temp[temp==True].index
selected_features_r


# In[199]:


yds2=ydadosS[selected_features_r].copy()
yds2.head()


# ## Model Definition and Fine Tunning

# In[200]:


X1_train, X1_test, y1_train, y1_test = train_test_split(
    ydr,target_1y,test_size = 0.3, random_state = 0, shuffle = True, stratify = target_1y)


# In[201]:


X2_train, X2_test, y2_train, y2_test = train_test_split(
    yds,target_1y,test_size = 0.3, random_state = 0, shuffle = True, stratify = target_1y)


# In[202]:


X3_train, X3_test, y3_train, y3_test = train_test_split(
    yds2,target_1y,test_size = 0.3, random_state = 0, shuffle = True, stratify = target_1y)


# In[203]:


#Naive Bayes
ymodelNB=GaussianNB()

#RF
ymodelRFT1=RandomForestClassifier()

#Decision Tree
ymodelDT = DecisionTreeClassifier()

#NN
ymodelNNC=MLPClassifier()


# In[204]:


DTC_param = GridSearchCV(ymodelDT, parameter_space_DT,scoring='f1')

DTC_param.fit(X1_train, y1_train)

DTC_param.best_params_, DTC_param.best_score_


# In[205]:


DTC_param = GridSearchCV(ymodelDT, parameter_space_DT,scoring='f1')

DTC_param.fit(X3_train, y3_train)

DTC_param.best_params_, DTC_param.best_score_


# In[206]:


ymodel_DTC1=DecisionTreeClassifier(criterion= 'gini',
                                max_depth= 2,
                                max_leaf_nodes= 5,
                                min_impurity_decrease= 0,
                                min_samples_leaf= 2,
                                min_samples_split= 2,
                                min_weight_fraction_leaf= 0,
                                splitter= 'random')
ymodel_DTC2=DecisionTreeClassifier(criterion= 'entropy',
                                max_depth= 8,
                                max_leaf_nodes= 10000,
                                min_impurity_decrease= 0,
                                min_samples_leaf= 2,
                                min_samples_split= 10,
                                min_weight_fraction_leaf= 0,
                                splitter= 'random')


# In[207]:


BP_NNC= GridSearchCV(ymodelNNC, parameter_space, scoring='f1')
BP_NNC.fit(X1_train, y1_train)
BP_NNC.best_params_, BP_NNC.best_score_


# In[208]:


BP_NNC= GridSearchCV(ymodelNNC, parameter_space,scoring='f1')
BP_NNC.fit(X3_train, y3_train)
BP_NNC.best_params_, BP_NNC.best_score_


# In[211]:


ymodel_NN1=MLPClassifier(activation= 'tanh',
                        hidden_layer_sizes= (50,),
                        learning_rate= 'constant',
                        learning_rate_init= 0.00001,
                        solver= 'sgd')
ymodel_NN2=MLPClassifier(activation= 'relu',
                        hidden_layer_sizes= (50),
                        learning_rate= 'constant',
                        learning_rate_init= 0.00001,
                        solver= 'adam')


# In[212]:


ygb = GradientBoostingClassifier()


# In[213]:


#https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

param_test2 = {'max_depth':range(2,16,4), 'min_samples_split':range(10,200,5)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2,n_jobs=4,iid=False, cv=5, scoring='f1')
                        #param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X3_train,y3_train)
gsearch2.best_params_, gsearch2.best_score_


# In[216]:


param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=10,min_samples_split=45, min_samples_leaf=60, subsample=0.8, random_state=10),
param_grid = param_test5, scoring='f1',n_jobs=4,iid=False, cv=5)
gsearch5.fit(X3_train,y3_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[217]:


ygb=GradientBoostingClassifier(learning_rate=0.1, max_depth=10,min_samples_split=45, min_samples_leaf=60, 
                              subsample=0.6, random_state=10)


# ## Model Evaluation and Selection
# 
# We will be testing all the previously defined models with the 4 normalized datasets. A first observation of the ROC curves and AUC values is performed. After, a selection of the best models is used to ascertain the model with the best recall - that is, the model that'll predict correctly the highest number of employee who leave (even if that means more false positives).

# In[218]:


yd1.name= 'Dataset com 13 variáveis, RobustScaler'
ydr.name = 'Dataset com 3 variáveis RobustScaler'
yds.name = 'Datasetcom 12 variáveis MinMaxScaler'
yds2.name = 'Dataset com 6 variáveis MinMaxScaler'
datasets=[yd1,ydr,yds,yds2]
results=[]

#modelos selecionados
models=[ymodelLR,ymodel_DTC1,ymodel_DTC2,ymodel_NN1,ymodel_NN2,ymodelNB,ymodelRFT1,ygb]
dict_score=dict()
for a in datasets:
    for m in models:
        X_train, X_val, y_train, y_val = train_test_split(a,target_1y,test_size = 0.3, random_state = 0, shuffle = True, stratify = target_1y)
        print('\033[1mCurrent dataset\033[0m:', a.name)
        print('\033[1mCurrent model\033[0m:', m,'\n\n')
        dict_score=score_model(m,X_train,y_train, X_val, y_val)
        results.append(dict_score)
        AUC_ROC_curve(m, X_train, y_train, X_val, y_val,target)


# The model with the best AUC is the Gradient Boosting - for more that one dataset, it has an area under curve above 0.7.   
# However, will continue our performance testing with more models: logistic regression, one of the decision trees (the first one, typically with the highest AUC), the random forest, the Naive Bayes, the MLPClassifier (the second one, typically with the best AUC) and the Gradient Boosting.

# In[219]:


#Kfold method for better model scoring
kf = KFold(n_splits=10)
rkf=RepeatedKFold(n_splits=5, n_repeats=2)
partition_methods=[kf,rkf]

#The 
selected_models=[ymodelLR,ymodel_DTC1,ymodelRFT1, ymodelNB, ymodel_NN2,ygb]


# In[221]:


#Function to score models with multiple scoring parameters (precision, accuracy, recall and f1-score)
def avg_score_partition(method,X,y,model):
    
    modelf=model
    
    score_acc_val = []
    score_prec_val = []
    score_recall_val = []
    score_f1_val = []
    score_train=[]
    score_test=[]

    for train_index, test_index in method.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        modelf = modelf.fit(X_train, y_train)
        dict_score=score_model(modelf,X_train,y_train, X_val, y_val)
        
        score_acc_val.append(dict_score['acc_score'])
        score_prec_val.append(dict_score['precision'])
        score_recall_val.append(dict_score['recall'])        
        score_f1_val.append(dict_score['f1'])
        
        value_train = modelf.score(X_train, y_train)
        value_test = modelf.score(X_val,y_val)
        score_train.append(value_train)
        score_test.append(value_test)

        
    print('Model score - Train:', np.mean(score_train))
    print('Model score - Test:', np.mean(score_test))
    print('Accuracy score:', np.mean(score_acc_val))
    print('Precision score:', np.mean(score_prec_val))
    print('Recall score:', np.mean(score_recall_val))
    print('F1-Score:', np.mean(score_f1_val))
    


# In[222]:


#Apply the scoring function to all the datasets, and the 

for a in datasets:
    for m in selected_models:
        print('\033[1mCurrent dataset\033[0m:', a.name)
        print('\033[1mCurrent model\033[0m:', m,'\n\n')
        for pm in partition_methods:
            avg_score_partition(pm,a,target_1y,m)
            print('\n')


# ### Final Model Selection
# 
# We will be picking the two models with the highest recall score: the Gaussian Naive Bayes and the MLPClassifier.

# In[259]:


X_train, X_val, y_train, y_val = train_test_split(ydr,target_1y,test_size = 0.3, random_state = 0, shuffle = True, stratify = target_1y)
dict_final=score_model(ymodel_NN2,X_train,y_train, X_val, y_val)
print('Confusion Matrix:\n',dict_final['cm'])
print('Accuracy:',dict_final['acc_score'])
print('Precison:',dict_final['precision'])
print('Recall:',dict_final['recall'])
print('F1-Score:',dict_final['f1'])
print('\n Full report:\n',dict_final['report'])
y_pred = ymodel_NN2.predict(ydr)
soma=sum(y_pred)
print('Total employees identified:',soma)


# **Conclusions**: although with a very high recall, this won't be our selected model - because it predicts almost every employee will be leaving. For this type of results, we could just assume everybody is leaving and a model isn't really necessary.

# In[230]:


X_train, X_val, y_train, y_val = train_test_split(yd1,target_1y,test_size = 0.3, random_state = 0, shuffle = True, stratify = target_1y)
dict_final=score_model(ymodelNB,X_train,y_train, X_val, y_val)
print('Confusion Matrix:\n',dict_final['cm'])
print('Accuracy:',dict_final['acc_score'])
print('Precison:',dict_final['precision'])
print('Recall:',dict_final['recall'])
print('F1-Score:',dict_final['f1'])
print('\n Full report:\n',dict_final['report'])
y_pred = ymodelNB.predict(yd1)
soma=sum(y_pred)
print('Total employees identified:',soma)


# **Conclusions**: The Gaussian Naives Bayes model is our best bet. It has a reasonable attrition recall (45%) while still maintaining a high overall recall and precision (71% and 81%, respectively).  

# In[ ]:




