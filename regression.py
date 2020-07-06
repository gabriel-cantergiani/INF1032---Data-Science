from sklearn.linear_model import LinearRegression
import numpy as np
def regression(df):
    feature = df['LandMaxTemperature'].notna()
    target =  df['LandAverageTemperature'].notna()
    print(feature)
    print(target)
    # Create the regressor: reg
    reg = LinearRegression()

    # Create the prediction space
    prediction_space = np.array(feature).reshape(-1,1)
    
    # Fit the model to the data
    reg.fit(feature,np.array)

    # Compute predictions over the prediction space: y_pred
    y_pred = reg.predict(prediction_space)

    # Print R^2 
    print(reg.score(feature, target))