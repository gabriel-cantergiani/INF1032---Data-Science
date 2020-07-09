import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
def correlation(df):
    plt.style.use('ggplot')
    print("Correlacao")
    dt1850 = datetime.datetime(1850, 1, 1)
    df = df[df['dt'] > dt1850]
    feature = df['LandAverageTemperature']
    target = df['LandAndOceanAverageTemperature']

    x = np.array(feature)
    y = np.array(target)
    r = np.corrcoef(x,y)
    print(r)
    plt.scatter(x, y)
    plt.show()

def regressionLandToOcean(df):
    print("Regressão com um parametro")
    dt1850 = datetime.datetime(1850, 1, 1)
    df = df[df['dt'] > dt1850]
    feature = df['LandAverageTemperature']
    target = df['LandAndOceanAverageTemperature']
    feature = feature.dropna(how='any')  
    target = target.dropna(how='any')  
    # Create the regressor: reg
    reg = LinearRegression()

    x = np.array(feature).reshape((-1, 1))
    y = np.array(target)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

    print("Fitting: ")
    # Fit the model to the data
    reg.fit(X_train,y_train)

    # Compute predictions over the prediction space: y_pred
    y_pred = reg.predict(X_test)
    
    # Print R^2 
    print("Score:", reg.score(X_test, y_test))
    return reg
def predictLandAndOcean(df, averageLand):
    reg = regressionLandToOcean(df)
    sample = np.array([averageLand]).reshape((1,-1))
    predict = reg.predict(sample)
    print("Predicao: " ,predict)
    print("Valor real: 14.774")
