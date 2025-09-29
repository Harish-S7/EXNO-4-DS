# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME  : NAVEEN KUMAR P
REG NO: 212224240102
```
```python

 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("income.csv",na_values=[ " ?"])
 data

```
<img width="1636" height="475" alt="image" src="https://github.com/user-attachments/assets/398c96e2-2b62-4606-bb78-35fc5d1e6a7c" />

```python

 data.isnull().sum()

```
<img width="1243" height="559" alt="image" src="https://github.com/user-attachments/assets/d1e3a404-de0a-4e03-b326-da4665e3121c" />


```python

 missing=data[data.isnull().any(axis=1)]
 missing

```
<img width="1527" height="470" alt="image" src="https://github.com/user-attachments/assets/82f91690-b1e6-44f7-9176-14e455e5e8e0" />


```python

data2=data.dropna(axis=0)
data2

```
<img width="1599" height="479" alt="image" src="https://github.com/user-attachments/assets/c5470e18-06b5-4b58-aa90-329a88c1d051" />


```python

 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])

```
<img width="1476" height="372" alt="image" src="https://github.com/user-attachments/assets/bdd27258-2001-4b0c-aa4e-a53152f3e046" />


```python

 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs

```
<img width="1208" height="483" alt="image" src="https://github.com/user-attachments/assets/4ccbac62-f88c-4d19-9558-0142813bd6bd" />


```python

 data2

```
<img width="1466" height="442" alt="image" src="https://github.com/user-attachments/assets/c4fb369a-b2ed-4a64-9612-e9cdf77b13ad" />


```python

 new_data=pd.get_dummies(data2, drop_first=True)
 new_data

```
<img width="1820" height="273" alt="image" src="https://github.com/user-attachments/assets/71081849-45e9-4f43-a0df-f7eba150ecd9" />


```python

columns_list=list(new_data.columns)
print(columns_list)

```
['age', 'capitalgain', 'capitalloss', 'hoursperweek', 'SalStat', 'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'gender_ Male', 'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia']

```python

 features=list(set(columns_list)-set(['SalStat']))
 print(features)

```
['nativecountry_ Taiwan', 'maritalstatus_ Never-married', 'nativecountry_ Iran', 'maritalstatus_ Separated', 'occupation_ Armed-Forces', 'nativecountry_ Portugal', 'nativecountry_ United-States', 'occupation_ Tech-support', 'race_ Other', 'JobType_ Private', 'hoursperweek', 'relationship_ Unmarried', 'occupation_ Craft-repair', 'occupation_ Transport-moving', 'EdType_ 9th', 'JobType_ Local-gov', 'EdType_ Preschool', 'nativecountry_ Yugoslavia', 'JobType_ State-gov', 'maritalstatus_ Married-spouse-absent', 'nativecountry_ Ireland', 'nativecountry_ Poland', 'nativecountry_ Columbia', 'nativecountry_ Hong', 'nativecountry_ Puerto-Rico', 'EdType_ 7th-8th', 'nativecountry_ Guatemala', 'nativecountry_ Philippines', 'occupation_ Exec-managerial', 'nativecountry_ Scotland', 'race_ White', 'relationship_ Own-child', 'occupation_ Priv-house-serv', 'JobType_ Self-emp-inc', 'EdType_ HS-grad', 'nativecountry_ China', 'nativecountry_ Peru', 'nativecountry_ France', 'nativecountry_ South', 'EdType_ 5th-6th', 'occupation_ Protective-serv', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ Canada', 'nativecountry_ Trinadad&Tobago', 'occupation_ Other-service', 'maritalstatus_ Widowed', 'nativecountry_ Nicaragua', 'race_ Asian-Pac-Islander', 'JobType_ Self-emp-not-inc', 'nativecountry_ Hungary', 'nativecountry_ Italy', 'capitalgain', 'maritalstatus_ Married-civ-spouse', 'occupation_ Prof-specialty', 'nativecountry_ England', 'nativecountry_ Thailand', 'nativecountry_ India', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'relationship_ Wife', 'EdType_ Prof-school', 'maritalstatus_ Married-AF-spouse', 'occupation_ Farming-fishing', 'race_ Black', 'relationship_ Other-relative', 'EdType_ 12th', 'nativecountry_ Japan', 'nativecountry_ Germany', 'nativecountry_ Cuba', 'JobType_ Without-pay', 'occupation_ Sales', 'nativecountry_ Jamaica', 'EdType_ 11th', 'nativecountry_ Laos', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'relationship_ Not-in-family', 'EdType_ Doctorate', 'EdType_ Masters', 'nativecountry_ Haiti', 'occupation_ Machine-op-inspct', 'nativecountry_ El-Salvador', 'EdType_ Assoc-acdm', 'gender_ Male', 'EdType_ Bachelors', 'occupation_ Handlers-cleaners', 'capitalloss', 'nativecountry_ Greece', 'age', 'nativecountry_ Vietnam', 'nativecountry_ Mexico', 'EdType_ Assoc-voc', 'EdType_ Some-college', 'EdType_ 1st-4th']

```python

 y=new_data['SalStat'].values
 print(y)

```
<img width="484" height="35" alt="image" src="https://github.com/user-attachments/assets/a3487d9c-971a-411d-b619-ee02e307b7d2" />

```python

 x=new_data[features].values
 print(x)

```
<img width="405" height="127" alt="image" src="https://github.com/user-attachments/assets/44b4733c-8627-41cc-a183-2d05d955e8cb" />


```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

```
<img width="164" height="57" alt="image" src="https://github.com/user-attachments/assets/bb4aca7b-76d2-4c60-8779-6cece18ae1f2" />

```python

 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)

```
<img width="358" height="83" alt="image" src="https://github.com/user-attachments/assets/be689b21-e157-47ee-8f7b-66eef6dab4e9" />


```python

 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)

```
0.8392087523483258

```python

 print("Misclassified Samples : %d" % (test_y !=prediction).sum())


```
Misclassified Samples : 1455

```python

 data.shape


```
(31978, 13)

```python

import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)

```
<img width="1692" height="83" alt="image" src="https://github.com/user-attachments/assets/3cc1eabd-e7dc-4713-a9be-58002738b079" />

```python

 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()

```
<img width="818" height="201" alt="image" src="https://github.com/user-attachments/assets/c319d855-b6e5-4608-be90-22342939d594" />


```python

 tips.time.unique()

```
<img width="836" height="64" alt="image" src="https://github.com/user-attachments/assets/473926ef-fa99-4289-9a12-467ab7ede29f" />


```python

 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)

```
<img width="447" height="81" alt="image" src="https://github.com/user-attachments/assets/e2cdcde4-faf9-47a4-ba1d-6b6b7f34bb1a" />


```python

 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")

```
<img width="528" height="47" alt="image" src="https://github.com/user-attachments/assets/264cf9b5-946c-4e5b-afae-d5d65f3da5da" />


      
# RESULT:

 Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
 save the data to a file is been executed.
