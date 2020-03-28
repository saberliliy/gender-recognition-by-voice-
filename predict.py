# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno
from sklearn.externals import joblib
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder
clf = joblib.load("1.m")
features_to_use = ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]
test_df = pd.read_csv('./Data/voiceSamples.csv')
test_X = test_df[features_to_use]
text_Y=test_df["label"]
scaler=StandardScaler()
test_X.drop(['skew','kurt','mindom','maxdom',"centroid"],axis=1,inplace=True)
test_X['meanfreq']=test_X['meanfreq'].apply(lambda x:x*2)
test_X['median']=test_X['meanfreq']+test_X['mode']
test_X['median']=test_X['median'].apply(lambda x:x/3)
test_X['pear_skew']=test_X['meanfreq']-test_X['mode']
test_X['pear_skew']=test_X['pear_skew']/test_X['sd']
scaled_df=scaler.fit_transform(test_X)
pro=clf.predict(scaled_df)
text_Y=text_Y.replace({"male":1,"female":0})
text_Y=text_Y.values.reshape(1,-1)
acc_tmp=text_Y-pro
acc_tmp=acc_tmp.flatten()
print(acc_tmp)
def num_one(source_array):
 count = 0
 for x in source_array:
  if x == 0:
   count += 1
 return count
acc_number=num_one(acc_tmp)
acc=acc_number/len(text_Y.flatten())
print(acc)