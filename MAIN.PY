import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import os
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt #use to visualize dataset vallues
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
features_scaler = MinMaxScaler(feature_range = (0, 1))
target_scaler = MinMaxScaler(feature_range = (0, 1))
full_dataset = pd.read_csv("Dataset/stocksdata.csv")
full_dataset
risk_ratings = pd.read_csv("Dataset/SP 500 ESG Risk Ratings.csv")
risk_ratings
full_dataset.isnull().sum()
risk_ratings.isnull().sum()
label_encoder = []
columns = risk_ratings.columns
types = risk_ratings.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object' and columns[i] != 'Symbol': #finding column with object type
        le = LabelEncoder()
        risk_ratings[columns[i]] = pd.Series(le.fit_transform(risk_ratings[columns[i]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[i], le])
risk_ratings.fillna(risk_ratings.median(), inplace = True)#applying median technique to handle missing values
risk_ratings.isnull().sum()
full_dataset.describe()
full_dataset.hist(figsize=(8, 7))
plt.title("Representation of Dataset Attributes")
plt.show()
data = full_dataset.groupby(["ticker"]).size().sort_values(ascending=False).reset_index(name='Count')[:50]
g = sns.barplot(x = 'ticker', y = 'Count', data = data)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.get_figure().set_size_inches(18, 4)
plt.title("Different Stock Companies Data Graph")
plt.show()
risk_scores = risk_ratings[['Symbol', 'Total ESG Risk score']]
risk_scores.rename(columns={'Symbol': 'ticker'}, inplace=True)
merge_df = pd.merge(full_dataset, risk_scores, on='ticker')
merge_df
data = full_dataset.groupby(["ticker"])['close'].mean().sort_values(ascending=False).reset_index(name='Average Closing Price')[:50]
plt.figure(figsize=(16, 4))
plt.plot(data['ticker'].ravel(), data['Average Closing Price'].ravel())
plt.xticks(rotation=70)
plt.xlabel("Stock Names")
plt.ylabel("Average Closing Price")
plt.title("Average Closing Price Graph for Top 50 Companies")
plt.show()
data = merge_df.groupby(["ticker"])['Total ESG Risk score'].mean().sort_values(ascending=False).reset_index()[:50]
g = sns.barplot(x='Total ESG Risk score', y="ticker", data=data)
g.get_figure().set_size_inches(6, 12)
plt.title('ESG Risk Score Graph')
plt.xlabel("Risk Score")
plt.ylabel("Stock Name")
plt.xticks(rotation=90)
plt.show()
data = merge_df.groupby(["ticker", 'Total ESG Risk score'])['close'].mean().sort_values(ascending=False).reset_index()[:10]
data = data.values
temp = []
for i in range(len(data)):
    temp.append([data[i,0], data[i,1], 0])
    temp.append([data[i,0], data[i,1], data[i,2]])
data = pd.DataFrame(temp, columns=['ticker', 'Total ESG Risk score', 'close'])    
plot = sns.boxplot(x="Total ESG Risk score", y = 'close', hue='ticker', data = data, palette = "icefire")
plot.get_figure().set_size_inches(12, 3)
plt.setp(plot.get_xticklabels(), rotation=90)
plt.title('Top 10 Stock Wise Risk Score & Closing Prices')
plt.show()
merge_df['date'] = pd.to_datetime(merge_df['date'])
merge_df['year'] = merge_df.date.dt.year
merge_df['month'] = merge_df.date.dt.month
merge_df['day'] = merge_df.date.dt.day
#if you want analysis for different stock then change AAPl to some other stock
choosen_stock = merge_df.loc[merge_df['ticker'] == "AAPL"]
fig, ax = plt.subplots(figsize=(8, 3))
sns.lineplot(data=choosen_stock, x='year', y='close', legend='full', ax=ax)
plt.title("Choosen AAPL Yearwise Performance")
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.show()
corr_matrix = merge_df.corr()
# Create the heatmap using Seaborn
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
choosen_stock['MA_10'] = choosen_stock['close'].rolling(window=10).mean()
choosen_stock['MA_50'] = choosen_stock['close'].rolling(window=50).mean()
# Drop NaN values
choosen_stock = choosen_stock.dropna()
# get training X and target Y features
X = choosen_stock[['close', 'MA_10', 'MA_50', 'Total ESG Risk score']]
Y = choosen_stock['close']
X = X.values
Y = Y.values
Y = Y.reshape(-1, 1)
X = features_scaler.fit_transform(X)
Y = target_scaler.fit_transform(Y)
print("Normalized Features = "+str(X))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("Train & Test Dataset Split")
print("80% records used to train algorithms : "+str(X_train.shape[0]))
print("20% records features used to test algorithms : "+str(X_test.shape[0]))
rsquare = []
rmse = []
mae = []
def calculateMetrics(algorithm, predict, test_labels):
    mse_error = mean_squared_error(test_labels, predict)
    mae_error = mean_absolute_error(test_labels, predict)
    r2_scores = r2_score(test_labels, predict)
    rmse_error = sqrt(mse_error)
    rsquare.append(r2_scores)
    rmse.append(rmse_error)
    mae.append(mae_error)
    predict = predict.reshape(-1, 1)
    predict = target_scaler.inverse_transform(predict)
    test_label = target_scaler.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    print()
    print(algorithm+" MAE : "+str(mae_error))
    print(algorithm+" RMSE : "+str(rmse_error))
    print(algorithm+" R2 : "+str(r2_scores))
    print()
    for i in range(0, 10):
        print("True Stock Prices : "+str(test_label[i])+" Predicted Prices : "+str(predict[i]))
    plt.figure(figsize=(5,3))
    plt.plot(test_label[0:100], color = 'red', label = 'True Stock Prices')
    plt.plot(predict[0:100], color = 'green', label = 'Predicted Prices')
    plt.title(algorithm+' S&P Stock Prices Forecasting Graph')
    plt.xlabel('Number of Test Samples')
    plt.ylabel('Stock Prices Forecasting')
    plt.legend()
    plt.show()  
rf_cls = RandomForestRegressor()
#tuning random forest with different parameters
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [ 5, 10]}
rf_cls = GridSearchCV(estimator=rf_cls, param_grid = param_grid)
rf_cls.fit(X_train, y_train.ravel())
#perform prediction on test datat
predict = rf_cls.predict(X_test)
#call this function to calculate performance metrics
calculateMetrics("Random Forest", predict, y_test)
gb_cls = GradientBoostingRegressor()
#tuning gradient with different parameters
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [ 5, 10]}
gb_cls = GridSearchCV(estimator=gb_cls, param_grid = param_grid)
gb_cls.fit(X_train, y_train.ravel())
#perform prediction on test datat
predict = gb_cls.predict(X_test)
#call this function to calculate performance metrics
calculateMetrics("Gradient Boosting Regression", predict, y_test)
height = rsquare
bars = ['Random Forest', 'Gradient Boosting']
y_pos = np.arange(len(bars))
plt.figure(figsize = (4, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Algorithms")
plt.ylabel("R2 Score")
plt.title("All Algorithms R2Score Comparison Graph")
plt.tight_layout()
plt.show()
algorithms = ['Random Forest', 'Gradient Boosting']
data = []
for i in range(len(rmse)):
    data.append([algorithms[i], rsquare[i], rmse[i], mae[i]])
data = pd.DataFrame(data, columns=['Algorithm Name', 'R2Score', 'RMSE', 'MAE'])
data  
