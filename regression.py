from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import datetime
def regression(df):
    print("Regressão com um parametro")
    dt1850 = datetime.datetime(1850, 1, 1)
    df = df[df['dt'] > dt1850]
    feature = df['LandMaxTemperature']
    target = df['LandAverageTemperature']
    feature = feature.dropna(how='any')  
    target = target.dropna(how='any')  
    # Create the regressor: reg
    reg = LinearRegression()

    x = np.array(feature).reshape((-1, 1))
    y = np.array(target)

    print("Cross Validation: ")
    cvscores_3 = cross_val_score(reg, x,y,cv=3)
    print("3:",np.mean(cvscores_3))

    # Perform 10-fold CV
    cvscores_10 = cross_val_score(reg, x,y,cv=10)
    print("10:",np.mean(cvscores_10))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

    print("Fitting: ")
    # Fit the model to the data
    reg.fit(X_train,y_train)

    # Compute predictions over the prediction space: y_pred
    y_pred = reg.predict(X_test)
    
    # Print R^2 
    print(reg.score(X_test, y_test))

    print('intercept:', reg.intercept_)
    print('slope:', reg.coef_)

def regressionMultipleParameters(df):
    print("Regressão com dois parametros")
    dt1850 = x = datetime.datetime(1850, 1, 1)
    df = df[df['dt'] > dt1850]
    features = df[['LandMaxTemperature','LandMinTemperature']]
    target = df['LandAverageTemperature']
    features = features.dropna(how='any')  
    target = target.dropna(how='any')  
    # Create the regressor: reg
    reg = LinearRegression()
    
    x = np.array(features)#.reshape(()
    y = np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

    print("Cross Validation: ")
    cvscores_5 = cross_val_score(reg, x,y,cv=5)
    print(np.mean(cvscores_5))

    print("Normal regression: ")
    # Fit the model to the data
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    r_sq = reg.score(X_test, y_test)
    print('coefficient of determination:', r_sq)
    print('intercept:', reg.intercept_)
    print('slope:', reg.coef_)
    print("Mean Square Error: ", metrics.mean_squared_error(y_test, y_pred))
