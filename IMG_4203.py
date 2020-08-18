######STEP1: SETUP
import pandas as pd                #DataFrame: Transform data easily
import matplotlib.pyplot as plt    #Data visualization for analysis
import numpy as np                 #Sequences be converted to numpy arrays
import statsmodels.api as sm       #Supporting specifying models
import math

from sklearn.model_selection import train_test_split                             #built in machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error     #Stat Metrics

from scipy import stats                                                          #Easier to interpret output
from scipy.stats import kurtosis, skew

######STEP2: DATA LOADING
price_data = pd.read_excel(r'/Users/lucaswong/Desktop/Project1/Modelling/Step2.xlsx')

price_data.index = pd.to_datetime(price_data['Date'])                            #Set index to data column
price_data = price_data.drop(['Date'], axis = 1)                                 #Cancel the second column - inital dates

######STEP3: DATA CLEANING
new_column = {'Toyota':'Toyota Stock Price','Volks':'Volkswagen Stock Price','Ford':'Ford Stock Price','GM':'General Motors Stock Price'}   #Define new names
price_data = price_data.rename(columns = new_column) 
price_data = price_data.dropna()   #drop any missing values

######STEP4: GRAPH MAKING
x = price_data['Oil Price']
y = price_data[price_data.columns[0]]
plt.figure(1)
plt.plot(x, y, 'x', color = 'steelblue', label = 'Daily Price')                 #Creating a scattered graph
plt.title('How Oil Price Affects '+ price_data.columns[0])
plt.xlabel('Oil Price')
plt.ylabel(price_data.columns[0])

y1 = price_data[price_data.columns[1]]
plt.figure(2)
plt.plot(x, y1, 'x', color = 'steelblue', label = 'Daily Price')
plt.title('How Oil Price Affects '+ price_data.columns[1])
plt.xlabel('Oil Price')
plt.ylabel(price_data.columns[1])

y2 = price_data[price_data.columns[2]]
plt.figure(3)
plt.plot(x, y2, 'x', color = 'steelblue', label = 'Daily Price')
plt.title('How Oil Price Affects '+ price_data.columns[2])
plt.xlabel('Oil Price')
plt.ylabel(price_data.columns[2])

y3 = price_data[price_data.columns[3]]
plt.figure(4)
plt.plot(x, y3, 'x', color = 'steelblue', label = 'Daily Price')
plt.title('How Oil Price Affects '+ price_data.columns[3])
plt.xlabel('Oil Price')
plt.ylabel(price_data.columns[3])

plt.show()

######STEP5: CORRELATION MEASUREMENT
print("Correlation of automaker stock price and oil price:")
print(price_data.corr())
print("\n")
print("Short Statistical Summary:")
print(price_data.describe())
print("\n")

######STEP6: OUTLIERS CHECK
price_data.hist(grid = False, color = 'steelblue')
plt.show()

def truncate(m, decimals = 4):
    muti = 10 ** decimals
    return int(m * muti) / muti

for ele in range(4):
    kurt = truncate(kurtosis(price_data[price_data.columns[ele]], fisher = True))     #Calculating excess kurtosis using fisher method
    ske = truncate(skew(price_data[price_data.columns[ele]]))
    print('Kurtosis of ' + price_data.columns[ele] + ': ' + str(kurt), end = "   ")
    if 2.9 <= kurt <= 3.1:
        print('*Mesokurtic*')
    elif kurt < 2.9:
        print('*Platykurtic*')
    else:
        print('*Leptokurtic*')
    print(stats.kurtosistest(price_data[price_data.columns[ele]]))
    print('Skewness of ' + price_data.columns[ele] + ': ' + str(ske), end = "   ")
    if ske < -1 or ske > 1:
        print('*Highly skewed*')
    elif -0.5 < ske < 0.5:
        print('*Approximately symmetric*')
    else:
        print('*Moderately skewed*')
    print(stats.skewtest(price_data[price_data.columns[ele]]), end = "\n\n")

######STEP7: MODEL BUILDING
print("Which automaker would you like to investogate more?")
print("1:Toyota / 2:Volkswagen / 3:Ford / 4:General Motors")
n = input("Please input a number between 1 to 4: ")
Y = price_data[[price_data.columns[int(n)-1]]]
X = price_data[['Oil Price']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 1)   #20% of data will be used in testing
regression = LinearRegression()
regression.fit(X_train, Y_train)
intercept = regression.intercept_[0]
coefficient = regression.coef_[0][0]
print('Intercept of model = ', truncate(intercept))
print('Coefficent of model = ', truncate(coefficient), end = "\n\n")

a = input("Please input an oil price for prediction: ")
predict = regression.predict([[float(n)]])
predicted_value = predict [0][0]
print('The predicted value is ', truncate(predicted_value), end ="\n\n")

######STEP8: MODEL EVALUATING
Y_predict = regression.predict(X_test)
print("Reagrding " + price_data.columns[int(n)-1] + ":")
mse = mean_squared_error(Y_test, Y_predict)
mae = mean_absolute_error(Y_test, Y_predict)
rmse = math.sqrt(mse)
print("Mean squared error: " + str(truncate(mse)))
print("Mean absolute eerror: " + str(truncate(mae)))
print("Root mean squared error: " + str(truncate(rmse)))
print("R-squared: " + str(truncate(r2_score(Y_test,Y_predict))), end = "\n\n")

X2 = sm.add_constant(X)
model = sm.OLS(Y,X2)
est = model.fit()
print(est.summary(), end = "\n\n")

print("================================== END ======================================")









 




