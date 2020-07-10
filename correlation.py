import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
def correlation(df):
    plt.style.use('ggplot')
    dt1850 = datetime.datetime(1850, 1, 1)
    df = df[df['dt'] > dt1850]
    feature = df['LandAverageTemperature']
    target = df['LandAndOceanAverageTemperature']

    x = np.array(feature)
    y = np.array(target)
    r = np.corrcoef(x,y)
    print("Correlacao: ", r)
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

def graficoPredicao(dataset_temperaturas, regression, ocean_limite):
    land_passado = []
    ocean_passado = []

    # Obtendo lista com Anos Passados
    ocean_passado = np.unique(dataset_temperaturas['LandAndOceanAverageTemperature'])
    land_passado = np.unique(dataset_temperaturas['LandAverageTemperature'])
    # Juntando Anos passados e Anos futuros (para previsao)

    land_futuro = []
    i = land_passado[-1]
    while(i < ocean_limite):
        i = i + 0.2
        land_futuro.append(i)

    land_previsao = np.concatenate((land_passado, np.array(land_futuro)))
    X = land_previsao.reshape(-1, 1)

    # Gerando previsao a partir do modelo
    Y_predict = regression.predict(X)

    # Linhas de Plot
    linha_previsao_futuro = go.Scatter(
        x = land_previsao,
        y = Y_predict.tolist(),
        mode='lines',
        line=dict(
            color='red'
        ),
        name="Médias Previstas"
    )

    linha_medias_passado = go.Scatter(
        x = land_passado,
        y = ocean_passado,
        mode='lines',
        line=dict(
            color='blue'
        ),
        name="Médias Passadas"
    )


    data = [linha_previsao_futuro, linha_medias_passado]
    layout = dict(
        xaxis=dict(title='ano'),
        yaxis=dict(title='temperatura'),
        title='Temperatúra Média Superficie x Temperatúra Média Superficie e Oceano (Passado + Previsão)',
        height=700
    )

    # Criando figura e Plotando
    fig = go.Figure(
        data=data,
        layout=layout
    )

    fig.show()
