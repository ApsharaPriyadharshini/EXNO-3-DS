## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
<img width="1247" height="381" alt="image" src="https://github.com/user-attachments/assets/b5d1a813-f1d2-47ae-a22a-b4fd4a7049b5" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
<img width="1253" height="228" alt="image" src="https://github.com/user-attachments/assets/da3c7871-2305-4eb9-b279-150b5dd9eedb" />

df['bo2']=e1.fit_transform(df[["ord_2"]])
df
<img width="1252" height="383" alt="image" src="https://github.com/user-attachments/assets/4dbe8025-7596-4da0-91a0-873fc76ef6ae" />

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
<img width="1252" height="381" alt="image" src="https://github.com/user-attachments/assets/7dda92d7-1295-4283-b276-83c463f69fd2" />

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
<img width="1254" height="385" alt="image" src="https://github.com/user-attachments/assets/5dbe9578-b7c5-473f-b31e-598aadc79b48" />

pd.get_dummies(df2,columns=["nom_0"])
<img width="1254" height="389" alt="image" src="https://github.com/user-attachments/assets/0397cb76-fd97-46f6-b75e-0824495dceb1" />

pip install --upgrade category_encoders
<img width="1252" height="510" alt="image" src="https://github.com/user-attachments/assets/9835964e-60ee-42fe-9e5b-54ef932b21ed" />

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
<img width="1254" height="375" alt="image" src="https://github.com/user-attachments/assets/6cfeca54-47da-4bde-8141-10a4e9aa8719" />

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb.drop(columns=['Ord_2'],inplace=True)
dfb
<img width="1250" height="394" alt="image" src="https://github.com/user-attachments/assets/81fe610e-a1e2-4471-b372-c6a811897d41" />

from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

<img width="1254" height="384" alt="image" src="https://github.com/user-attachments/assets/2f860d9f-3918-45c7-ae47-9905fe8bfb9d" />

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
<img width="1258" height="461" alt="image" src="https://github.com/user-attachments/assets/476bdadd-280f-44b8-90d7-cce5e1533014" />

df.skew()
<img width="1250" height="127" alt="image" src="https://github.com/user-attachments/assets/6d720713-33a3-4414-844e-0ef6b98a02a1" />

np.log(df["Highly Positive Skew"])
<img width="1264" height="281" alt="image" src="https://github.com/user-attachments/assets/6de9cc62-1887-4f29-b172-3d8a6238d675" />

np.reciprocal(df["Moderate Positive Skew"])
<img width="1252" height="288" alt="image" src="https://github.com/user-attachments/assets/ca6f859e-e0d7-49e2-b4c3-ad40f95df857" />

np.sqrt(df["Highly Positive Skew"])
<img width="1254" height="277" alt="image" src="https://github.com/user-attachments/assets/c7d6380d-79d9-4bfa-9917-01b65fda748b" />

np.square(df["Highly Positive Skew"])
<img width="1242" height="272" alt="image" src="https://github.com/user-attachments/assets/4e044eae-01da-4e1c-9b74-8203ab61c754" />

from sklearn.preprocessing import PowerTransformer
df["Highly Positive Skew"],_=stats.boxcox(df["Highly Positive Skew"])
pt=PowerTransformer(method='yeo-johnson')
df["Moderate Negative Skew_yeojohnson"]=pt.fit_transform(df[["Moderate Negative Skew"]])
df
<img width="1253" height="485" alt="image" src="https://github.com/user-attachments/assets/17abf6b1-63dc-4261-988d-e14bf73b677e" />

df["Modderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
<img width="1258" height="231" alt="image" src="https://github.com/user-attachments/assets/f72154a9-4066-4fc2-aa18-48021c2189a6" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
<img width="1247" height="561" alt="image" src="https://github.com/user-attachments/assets/2c12583f-2373-4c5f-81d7-df840bb4a463" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
<img width="1250" height="558" alt="image" src="https://github.com/user-attachments/assets/c3a83a29-95a6-4b98-ab31-7c4b071a4df8" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
<img width="1245" height="557" alt="image" src="https://github.com/user-attachments/assets/fcd1e007-f516-444f-b6cd-408a2958d138" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot (df['Highly Negative Skew'],line='45')
plt.show()
<img width="1250" height="567" alt="image" src="https://github.com/user-attachments/assets/0cfd0959-8019-4356-bb20-7725f65246cf" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
<img width="1248" height="561" alt="image" src="https://github.com/user-attachments/assets/6d036551-f341-4a67-90d6-3be3e37c2d4e" />

dt=pd.read_csv("titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot (dt['Age'],line='45')
plt.show()
<img width="1244" height="557" alt="image" src="https://github.com/user-attachments/assets/586c7e90-348b-4864-93c7-20b54c571d23" />

sm.qqplot(dt['Age_1'],line='45')
plt.show()
<img width="1243" height="562" alt="image" src="https://github.com/user-attachments/assets/872efd05-7a10-4371-9248-e11230defd0d" />


# RESULT:
Thus, the feature encoding and feature transformation process is executed successfully and the output is verified.

       
