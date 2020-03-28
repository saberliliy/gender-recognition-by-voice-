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
from sklearn.neural_network import MLPClassifier
#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix
#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

train=pd.read_csv('./Data/voice.csv')
df=train.copy()
df.isnull().any()
msno.matrix(df)
def calc_limits(feature):
    q1,q3=df[feature].quantile([0.25,0.75])
    iqr=q3-q1
    rang=1.5*iqr
    return(q1-rang,q3+rang)


def plot(feature):
    fig, axes = plt.subplots(1, 2)
    sns.boxplot(data=df, x=feature, ax=axes[0])
    sns.distplot(a=df[feature], ax=axes[1], color='#ff4125')
    fig.set_size_inches(15, 5)
    plt.show()
    plt.close()
    lower, upper = calc_limits(feature)
    l = [df[feature] for i in df[feature] if i > lower and i < upper]
    print("Number of data points remaining if outliers removed : ", len(l))
features_to_use = ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]
# for col in features_to_use:
#     plot(col)
# sns.countplot(data=df,x='label')
# plt.show()
# plt.close()
temp = []
for i in df.label:
    if i == 'male':
        temp.append(1)
    else:
        temp.append(0)
df['label'] = temp
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

df.drop('centroid',axis=1,inplace=True)
def plot_against_target(feature):
    sns.factorplot(data=df,y=feature,x='label',kind='box')
    fig=plt.gcf()
    fig.set_size_inches(7,7)
plot_against_target('kurt')
plt.show()
g = sns.PairGrid(df[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")
g = g.map(plt.scatter).add_legend()
# for col in df.columns:
#     lower,upper=calc_limits(col)
#     df = df[(df[col] >lower) & (df[col]<upper)]
temp_df=df.copy()
temp_df.drop(['skew','kurt','mindom','maxdom'],axis=1,inplace=True) # only one of maxdom and dfrange.
temp_df.head(10)
temp_df['meanfreq']=temp_df['meanfreq'].apply(lambda x:x*2)
fig,axes=plt.subplots(1,1)
sns.boxplot(data=temp_df,y='meanfreq',x='label')
plt.show()
plt.close()
temp_df['median']=temp_df['meanfreq']+temp_df['mode']
temp_df['median']=temp_df['median'].apply(lambda x:x/3)
sns.boxplot(data=temp_df,y='median',x='label')
temp_df['pear_skew']=temp_df['meanfreq']-temp_df['mode']
temp_df['pear_skew']=temp_df['pear_skew']/temp_df['sd']
sns.boxplot(data=temp_df,y='pear_skew',x='label')

# scaler=StandardScaler()
# scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
# X=scaled_df
X=temp_df.drop('label',axis=1)

Y=df['label'].as_matrix()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(30, 30), learning_rate_init=0.001)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))
joblib.dump(mlp,"1.m")
