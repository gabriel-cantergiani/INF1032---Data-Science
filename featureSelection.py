# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# load data
def featureSelection(dataframe):
    dataframe = dataframe[dataframe['LandAverageTemperature'].notna()]
    dataframe = dataframe[dataframe['LandMaxTemperature'].notna()]
    features = dataframe.drop(['LandAverageTemperature'], 1)
    array = dataframe.values
    print(features.columns.values)
    features_values = features.values 
    X = features_values[:,1:7] #Num deu pra usar a data porque é string
    Y = array[:,1]
    # # feature extraction
    test = SelectKBest(score_func=f_classif, k=4)
    fit = test.fit(X, Y)
    # # summarize scores
    set_printoptions(precision=3)
    print("Scores: ")
    print(fit.scores_)
    features = fit.transform(X)
    #summarize selected features
    print("#summarize selected features")
    print(features[0:6,:])