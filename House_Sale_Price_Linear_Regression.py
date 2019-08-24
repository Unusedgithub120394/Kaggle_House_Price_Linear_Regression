import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
from scipy import stats

sns.set(style="whitegrid")


test = pd.read_csv('test.csv') #Reading test using panadas
train = pd.read_csv('train.csv') #Reading train using panadas

print('The test data shape:', test.shape, '\n' ) #inspecting size and shape of csv files
print('The train data shape:', train.shape)

print('1 \n')

plt.style.use(style = 'ggplot') #using a style of matplotlib.pyplot
plt.rcParams['figure.figsize'] = (10, 6)

print(train.SalePrice.describe()) #getting a statistical description of the SalePrice

print('The skew of the SalePrice is:', train.SalePrice.skew()) #Investigating skew of SalePrice
plt.hist(train.SalePrice, color = 'magenta') #Plotting histogram of skew
plt.title(label='Final Sale price of homes in Iowa 2003-13')
plt.xlabel(xlabel= 'SalePrice ($)')
plt.ylabel(ylabel= 'Number of Houses')
plt.show()

print('2 \n')

target = np.log(train.SalePrice) #Investigating skew of the natural log of SalePrice
print('\nLog skew of the SalePrice is:', target.skew())
plt.hist(target, color = 'green') #Plotting histogram of skew
plt.title(label='Final Sale price of homes in Iowa 2003-13')
plt.xlabel(xlabel= 'Ln(SalePrice) ($)')
plt.ylabel(ylabel= 'Number of Houses')
plt.show()

print('3 \n')

numeric_features = train.select_dtypes(include = [np.number]) #inspecting the correlation between features and final SalePrice
corr = numeric_features.corr()

High_corr_features = corr['SalePrice'].sort_values(ascending = False)[:5]
Low_corr_features = corr['SalePrice'].sort_values(ascending = False)[-5:]



print(High_corr_features, '\n') #Displaying the most correlated features with SalePrice
print(Low_corr_features, '\n') #Displaying the least correleated features with SalePrice

np.savetxt('High_corr_features.txt', High_corr_features, delimiter=',')
np.savetxt('Low_corr_features.txt', Low_corr_features, delimiter = ',')

print('4 \n')
################## OverallQual outlier removing ##############################
##############################################################################
sns.distplot(train.OverallQual)
print('OverallQual skew:', train.OverallQual.skew())
plt.show()

sns.scatterplot(x = 'OverallQual', y = 'SalePrice', data=train)
plt.title(label = 'SalePrice against OverallQual scatterplot')
plt.show()

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data=train)
plt.title(label='SalePrice against OverallQual (pre)' )
plt.show()

colls = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea']
train[colls].hist(figsize=(15,10))
plt.show()

#OQ = StandardScaler().fit_transform(train[colls].values)
#db = DBSCAN(eps=3.0, min_samples=10).fit(OQ)
#print(db)

#outliers_df = train(colls)
#print (Counter (db.labels_))
#print (outliers_df(db.labels_ == -1))
#labels = db.labels_
#train.Series(labels).value_counts()


#Q1 = train['OverallQual'].quantile(.25)
#Q3 = train['OverallQual'].quantile(.75)
#q1 = Q1-1.5*(Q3-Q1)
#q3 = Q3+1.5*(Q3-Q1)


sns.boxplot(x = 'OverallQual', y = 'SalePrice', data=train)
plt.title(label='SalePrice against OverallQual (pre)' )
plt.show()

print('5 \n')

################## GrLivArea outlier removing ##############################
##############################################################################

sns.distplot(train.GrLivArea)
print('GrLivArea skew:', train.GrLivArea.skew())
plt.show()

sns.lmplot(x = 'GrLivArea', y = 'SalePrice', data = train)
plt.title(label='SalePrice against GrLivArea (pre)' )
plt.ylim(0,None)
plt.show()

print('6 \n')

train = train[train['GrLivArea'] < 4000] #Removing GrLivArea outliers for SalePrice from the data

sns.lmplot(x = 'GrLivArea', y = 'SalePrice', data = train)
plt.title(label='SalePrice against GrLivArea (post)' )
plt.xlim(0,6000)
plt.ylim(0,700000)
plt.show()


print('7 \n')

################## GarageCars outlier removing ##############################
##############################################################################

sns.distplot(train.GarageCars)
print('GarageCars skew:', train.GarageCars.skew())
plt.show()

sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = train)
plt.title(label='SalePrice against GarageCars (pre)' )
plt.show()

print('8 \n')

################## GarageArea outlier removing ##############################
##############################################################################

sns.distplot(train.GarageArea)
print('GarageArea skew:', train.GarageArea.skew())
plt.show()

print(train.GarageArea.describe())
print(target.describe())

plt.scatter(x= train['GarageArea'], y = np.log(train.SalePrice))
plt.title(label='SalePrice against GarageArea (pre)' )
plt.xlabel('GarageArea')
plt.ylabel('Sale Price ($)')
plt.show()

print('9 \n')

train = train[train['GarageArea'] < 1200] #Removing GarageArea outliers for SalePrice from the data

plt.scatter(x = train['GarageArea'], y=np.log(train.SalePrice)) #Replotting the data
plt.xlim(-200,1600) #Forcing the same scale
plt.title(label='SalePrice against GarageArea (post)' )
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

print('10 \n')

################## Non numeric data wrangling ##############################
##############################################################################

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending = False)[:25]) #Returns counts of features with null values
nulls.columns = ['Null count'] #Naming null count
nulls.index.name = 'Feature' #Naming Feature
print(nulls)

print('11 \n')

categoricals = train.select_dtypes(exclude = [np.number])# Investigating non numeric features
print(categoricals.describe()) #count = number of non null observations
unique = # of unique sort_values
top = most common value
freq = frequency of top value

print('12 \n')

#print('Originals: \n')
#print(train.Street.value_counts() , '\n' ) #printing the unique non numerical states and their count of 'Street' feature
#now using one hot encoding to turn these into bolean values

train['enc_street'] = pd.get_dummies(train.Street, drop_first = True) #assigning Gravel road a value of 1, and pavement a value of 0
test['enc_street'] = pd.get_dummies(test.Street, drop_first = True)

print('Encoded: \n')
print(train.enc_street.value_counts()) #Now printing the unique states and count in the 'Street' feature after one hot encoding

print('13\n')

#Constructing a pivot table displaying the median Sale price for corresponding Sale conditions
#This data is then made into a bar chart to visualise

condition_pivot = train.pivot_table(index = 'SaleCondition', values = 'SalePrice', aggfunc = np.median)#constructing privot table
condition_pivot.plot(kind = 'bar', color = 'blue') #Plotting bar chart to visualise pivot table
plt.xlabel('Sale Condition') #Naming x axis
plt.ylabel('Median Sale Price ($)') #Naming y axi
plt.xticks(rotation = 'vertical' ) #Selecting orientation of xticks
plt.title(label = 'Median Sale price for different Sale conditions in Iowa 2003-13') #Naming graph
plt.show()

print('14 \n')

def encode(x): #making a function to encode SaleCondition feature
    """Encodes the SaleCondition feature and sets 'Partial' result to a value of 1,
    and all other SaleCondition unique results to 0"""

    return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode) #applying this encoding to train data
test['enc_condition'] = test.SaleCondition.apply(encode) #applying this encoding to test data


condition_pivot = train.pivot_table(index = 'enc_condition', values = 'SalePrice', aggfunc=np.median) #constructing a pivot table showing the median sale price of the two possible new Sale Conditions ('Partial' or 'other')
condition_pivot.plot(kind = 'bar', color = 'blue') #plotting the pivot table as a bar chart
plt.xlabel('Encoded Sale condition') #labelling x axis for graph
plt.ylabel('Median Sale Price') #labelling y axis for graph
plt.xticks(rotation = 0) #setting orientation of x ticks
plt.show()

print('15 \n')

data = train.select_dtypes(include = [np.number]).interpolate().dropna() #assigning missing values an average value (interpolating)
print(sum(data.isnull().sum() !=0)) #test to see if 0 null values ###sum of all null data, is it not equal to 0? sum of this###

y = np.log(train.SalePrice) #allocating the variable y to the thing we want to predict (SalePrice)
X = data.drop(['SalePrice', 'Id'], axis = 1) #dropping sale price from the features, to train data and hence predict Sale Price, also allocating the features to X

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size = 0.33) #Splitting data to training and test data

lr = linear_model.LinearRegression() #applying linear regression model
model = lr.fit(X_train, y_train) #fitting LinearRegression to training data
print('R^2 score:', model.score(X_test, y_test)) #printing R^2 value to measure fit of regression line

print('16 \n')

predictions = model.predict(X_test) #predicting Sale price using linear regression model
print('RMS is:', mean_squared_error(y_test, predictions)) #Printing the Root mean square between predicted price and actual price
print('\n')

plt.scatter(predictions, y_test, alpha=0.75,) #Plotting graph to visually represent this
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.show()
