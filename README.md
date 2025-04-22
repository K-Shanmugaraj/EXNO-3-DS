# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:

Read the given Data.

STEP 2:

Clean the Data Set using Data Cleaning Process.

STEP 3:

Apply Feature Encoding for the feature in the data set.

STEP 4:

Apply Feature Transformation for the feature in the data set.

STEP 5:

Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : Shanmuga Raj.K
### Reg No : 212223040192

```python

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/bc70d3db-1f90-4721-b768-5969a1d0e4fa)



```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/ac4e4519-861a-47a2-9c82-19d0c705c138)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/87c37747-afeb-4fde-b37a-f0cfdd55eb3f)



```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a3c8d57c-6f69-4590-b166-bd375794478f)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/8722da97-cd2d-49ff-9661-e1448db02ac7)



```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/62bdf6c7-46ec-4e67-b22a-3a5a950d5795)


```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/fa26c6fa-45ff-46c2-855b-68f0cad32bd5)


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/e7e28c03-3a74-43b5-9bd9-dad4807729fc)


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/5bcb1603-df4f-4eb2-ac93-e0ea8d16fbe1)



```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/2c61dd20-c7fb-48d3-9785-9200ffff2962)


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/65c769e4-a98d-475a-aee5-ae2b1928138d)



```py
df.skew()
```
![image](https://github.com/user-attachments/assets/c0d9db52-d7fc-4ae9-827a-9ec6c1e39f9b)



```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/cc4ebbdb-43ea-4d8e-b06b-dacc8d20a84a)


```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2d69c58b-ef94-430b-8cbf-81d8e7ae4f8b)




```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0d5bfe58-2dfe-4a5e-9586-307feceaf22b)


```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/014d5a25-ee94-4714-abb4-38a731969e87)


```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/278191c9-8f79-4d36-bcdd-25c54a519640)


```py
df.skew()
```
![image](https://github.com/user-attachments/assets/62341ebd-a9ec-4f3b-a777-676d381197bd)



```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/a5ff57a4-6832-44db-8145-4386e5c43645)


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/5743b2dc-4199-4261-ac61-93129f4e9ae9)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/65b553a2-946d-4d42-97ed-3149646c86e3)


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/dd693f3a-54e6-4bba-a5d1-649b12cdff7d)




```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/1e25c63d-c7d3-490e-82e6-76b6572c47e6)



```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/2ee2d713-95d2-40f6-96f9-abb835636c14)

```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/4521fb8e-a1ff-4aa1-a9bd-41ad7d466cf4)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/b7a3b5f6-265c-4e7a-a2b2-d9e1ec9dd199)


```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/a9858d5e-0e88-498f-acda-e058a4598812)





## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
