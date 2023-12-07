import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
ds=pd.read_csv("/kaggle/input/car-price-prediction/carvana.csv")
ds.head(3)
ds.shape
ds.duplicated().sum()
ds.drop_duplicates(inplace=True)
ds.shape
ds.info()
ds.isnull().sum()
ds["Name"].value_counts()
def edit_name(text):
    name=text.lstrip()
    return(name)
ds["Name"]=ds["Name"].apply(edit_name)   
ds["Year"].unique()
def edit(x):
    x_str=str(x)
    year_end=4
    year_str=x_str[:year_end]
    year=int(year_str)
    return(year)
ds["Year"]=ds["Year"].apply(edit)
#DATA SPLITTING
input_data=ds.iloc[:,:-1]
output_data=ds["Price"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.20,random_state=42)
num=Pipeline([("p4",SimpleImputer()),("p1",StandardScaler())])
cat=Pipeline([("p2",OrdinalEncoder()),("p3",StandardScaler())])
col_trans=ColumnTransformer(transformers=[("num1",num,["Year","Miles"]),("cat1",cat,["Name"])],remainder="passthrough")
col_trans
rfr=DecisionTreeRegressor(criterion="squared_error",splitter="best")
model=Pipeline(steps=[("col",col_trans),("model",rfr)])
model.fit(input_data,output_data)
pre=model.predict(x_test)
pre
file=open("car_price.txt","wb")
pickle.dump(model,file)
file.close()
import pickle
q=open("car_price.txt","rb")
model=pickle.load(q)
import pandas as pd
df=pd.DataFrame({"Name":["GMC Terrain"],"Year":[2020],"Miles":[45328]})
df
model.predict(df)
