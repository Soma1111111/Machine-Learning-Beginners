import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

data =pd.read_csv('Bengaluru_house_data.csv')
# print(data.head())
# for column in data.columns:
#     print(data[column].value_counts())
#     print("*"*20)
    
# print(data.info())
# print(data.isnull().sum())
data.drop(['area_type','availability','society','balcony'],axis=1,inplace=True)#dropping the columns as they are of no use to the model(more null vaues/non-null values)
# print(data.head())
# print(data.describe())

#filling the missing values
# data['location'] = data['location'].fillna()
# print(data['size'].value_counts())
data['location'] = data['location'].fillna('Sarjapur Road')
data['size'] = data['size'].fillna('2 BHK')
# print(data.isnull().sum())
# print(data['bath'].value_counts()) 
# #necessary to put () in value_counts()
data['bath'] = data['bath'].fillna(data['bath'].median())
# to verify for any null values
# print(data.info())
data['bhk'] = data['size'].str.split().str.get(0).astype(int)#new column
# print(data[data.bhk>20])#printing the flats with bhk size>20
# print(data['total_sqft'].unique())#as some of the values here are coming in ranges we can take those values as median
# print(data['total_sqft'].value_counts())
# ranges --> median

def conversion_for_total_sqft(i):
    array = i.split('-')
    if(len(array) == 2):
        return (float(array[0])+float(array[1]))/2
    try:
        return float(i)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(conversion_for_total_sqft)
# print(data.head())

# Price per square Feet new column
data['price_per_sqft'] = data['price']*100000/data['total_sqft']
# print(data.describe())
# print(data['location'].value_counts())
#replace the <10 values with others

data['location'] = data['location'].apply(lambda x : x.strip()) #to remove the leading and trailing whitespaces 
loc_count = data['location'].value_counts()
# print(loc_count)
# now the values decreased to 1295
loc_count_less_than_10 = loc_count[loc_count<=10]
# print(loc_count_less_than_10)
data['location'] = data['location'].apply(lambda x: 'other' if x in loc_count_less_than_10 else x)
# print(data['location'].value_counts())

#Impossible values are to be discarded(Outlier detection and removal)
# print(data.describe())
#here, while cchecking the total_sqft, minimum is 1 which is an outlier, so we are hecking for total_sqft/bhk
# print((data['total_sqft']/data['bhk']).describe())
#first quartile is 473, so less than 300/bhk,we are removing those
data = data[((data['total_sqft']/data['bhk'])>=300)]
# print(data.price_per_sqft.describe())
# print(data.describe())

#function to remove the outliers

def rem_imp_values(df):
    dfout = pd.DataFrame()
    for key,value in df.groupby('location'):
        mean = np.mean(value.price_per_sqft)
        std1 = np.std(value.price_per_sqft)
        new_df = value[(value.price_per_sqft > (mean-std1)) & (value.price_per_sqft <= (mean+std1))] # writing & only gives the result but writing and doesn't give the result
        dfout = pd.concat([dfout,new_df],ignore_index=True)
    return dfout
 
data = rem_imp_values(data)
# print(data.describe())

# outlier for bhk
def rem_bhk(df):
    removals = np.array([])
    for location,loc_df in df.groupby('location'):
        bhkstats = {}
        for bhk,bhk_df in loc_df.groupby('bhk'):
            bhkstats[bhk] = {'mean' : np.mean(bhk_df.price_per_sqft) , 'std' : np.std(bhk_df.price_per_sqft) , 'count' : bhk_df.shape[0]}
        # print(location,bhkstats)
        for bhk,bhk_df in loc_df.groupby('bhk'):
            stats = bhkstats.get(bhk-1) #if not found then None value is taken as stats
            if stats and stats['count']>5:
                removals = np.append(removals,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(removals,axis = 'index')

data = rem_bhk(data)
# print(data.groupby('location'))
# print(data.shape)
data.drop(columns = ['size','price_per_sqft'],inplace = True)
# print(data.head())
#final calculated data
data.to_csv("New_Final_Data.csv")
#finding x and y
x = data.drop(columns = ['price'])
y = data['price'] #because we have to find the predicted price

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
# print(x_train.shape)
# print(x_test.shape)

#Linear Regression
column_transformer = make_column_transformer((OneHotEncoder(sparse_output = False),['location']),remainder = 'passthrough')#categorical(location)
scaler = StandardScaler()
linreg = LinearRegression()
pipe = make_pipeline(column_transformer,scaler,linreg)
pipe.fit(x_train,y_train)
ypred = pipe.predict(x_test)
print(r2_score(y_test,ypred))
# pickle.dump(pipe,open('linemodel','wb'))