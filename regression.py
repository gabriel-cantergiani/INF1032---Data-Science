from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
def regression(df):
    dt1850 = x = datetime.datetime(1850, 1, 1)
    df = df[df['dt'] > dt1850]
    feature = df['LandMaxTemperature']
    target = df['LandAverageTemperature']
    feature = feature.dropna(how='any')  
    target = target.dropna(how='any')  
    # Create the regressor: reg
    reg = LinearRegression()

    # Create the prediction space
    prediction_space = np.array(feature).reshape((-1, 1))
    x = np.array(feature).reshape((-1, 1))
    y = np.array(target)
    print(x)
    print(y)
    # Fit the model to the data
    reg.fit(x,y)

    # Compute predictions over the prediction space: y_pred
    y_pred = reg.predict(prediction_space)

    # Print R^2 
    print(reg.score(x, y))