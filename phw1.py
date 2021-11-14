import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


df = pd.read_csv("C:/Users/MOS/vscode_git_ML/PHW/flu_diagnosis.csv")
print(df)
print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
feature_names = list(df.select_dtypes(object))

X = df.drop(['flu'], axis=1)
y = df['flu']
x = df.copy()

encoder = LabelEncoder()
data=pd.DataFrame()

x=x.reset_index(drop=True)  #index가 있는채로 인코딩하면 결측값이 생기므로 인덱스를 reset해주고 drop=True를 해서 다른 column으로 나오는것을 방지함

for i in feature_names:
    x[i] = encoder.fit_transform(x[i])
scaler = StandardScaler()
x = scaler.fit_transform(x)
df_new = pd.DataFrame(x)
df_new.columns = df.columns


X = df_new.drop(['flu'], axis=1)
df_new = df_new.astype({'flu': 'int'}) #Change "flu" type float to int 
y = df_new['flu']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

print("Confusion Matrix\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report\n",classification_report(y_test,y_pred))
