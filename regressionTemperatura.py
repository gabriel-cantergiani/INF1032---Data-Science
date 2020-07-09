from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.subplots as sub


def regressao_ano_temp_media(temperaturas):


    anos = []
    medias = []
    incerteza_medias = []

    # Obtendo lista com anos
    anos = np.unique(temperaturas['year'])

    for ano in anos:
        medias.append(temperaturas[temperaturas['year'] == ano]['LandAverageTemperature'].mean())
        incerteza_medias.append(temperaturas[temperaturas['year'] == ano]['LandAverageTemperatureUncertainty'].mean())


    # Gerando modelo de regressao Linear

    X = anos.reshape(-1, 1)
    Y = medias

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    regression = LinearRegression()

    regression.fit(X_train, Y_train)

    print("Intercept: ", regression.intercept_)
    print("Coeficient: ", regression.coef_)

    Y_predict = regression.predict(X_test)

    print("Mean Absolute Error: ", metrics.mean_absolute_error(Y_test, Y_predict))
    print("Mean Square Error: ", metrics.mean_squared_error(Y_test, Y_predict))
    print("Root Mean Square Error: ", np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))

    Y_predict_total = regression.predict(X).tolist()

    # Linhas dos Plots de Media

    linha_media = go.Scatter(
        x = anos,
        y = medias,
        mode='lines',
        line=dict(
            color='red'
        ),
        name="Média"
    )

    linha_media_superior = go.Scatter(
        x = anos,
        y = np.array(medias) + np.array(incerteza_medias),
        fill= None,
        mode='lines',
        line=dict(
            color='rgb(212, 121, 121)'
        ),
        name="Média Superior"
    )

    linha_media_inferior = go.Scatter(
        x = anos,
        y = np.array(medias) - np.array(incerteza_medias),
        fill= 'tonexty',
        mode='lines',
        line=dict(
            color='rgb(212, 121, 121)'
        ),
        name="Média Inferior"
    )

    # Linha do Plot de Previsao

    linha_previsao = go.Scatter(
        x = anos,
        y = Y_predict_total,
        mode='lines',
        line=dict(
            color='blue'
        ),
        name="Previsao"
    )

    # PLOT Somente dos Anos de Teste

    # Corrigindo e ajeitando as listas
    X_test_list = []
    for item in X_test.tolist():
        X_test_list.append(item[0])

    Y_test_list = Y_test
    Y_predict_list = []

    for item in Y_predict.tolist():
        Y_predict_list.append(item)


    # Linhas de Medias e Previsao (Somente Teste)
    linha_media_teste = go.Scatter(
        x = X_test_list,
        y = Y_test_list,
        mode='markers',
        line=dict(
            color='red'
        ),
        name="Média (dados de teste)"
    )

    linha_previsao_teste = go.Scatter(
        x = X_test_list,
        y = Y_predict_list,
        mode='lines',
        line=dict(
            color='blue'
        ),
        name="Previsão"
    )


    # Criando e Plotando Figura

    fig = sub.make_subplots(rows=3, cols=1, subplot_titles=("Temperatura Média Anual e Modelo", "Temperatura Média Anual (só dados de Teste) e Modelo"))

    fig.add_trace(linha_media_superior, row=1, col=1)
    fig.add_trace(linha_media_inferior, row=1, col=1)
    fig.add_trace(linha_media, row=1, col=1)
    fig.add_trace(linha_previsao, row=1, col=1)

    fig.add_trace(linha_media_teste, row=2, col=1)
    fig.add_trace(linha_previsao_teste, row=2, col=1)

    fig.update_layout(xaxis=dict(title='ano'), yaxis=dict(title='temperatura'), height=2000)

    fig.show()

    return regression


def previsao_temp_media_futura(dataset_temperaturas, regression, ano_limite):


    anos_passado = []
    medias_passado = []

    # Obtendo lista com Anos Passados
    anos_passado = np.unique(dataset_temperaturas['year'])

    # Obtendo lista com Medias de Temperaturas Passadas
    for ano in anos_passado:
        medias_passado.append(dataset_temperaturas[dataset_temperaturas['year'] == ano]['LandAverageTemperature'].mean())

    # Juntando Anos passados e Anos futuros (para previsao)
    anos_futuro = [i for i in range(anos_passado[-1] + 1, ano_limite)]
    anos_previsao = np.concatenate((anos_passado, np.array(anos_futuro)))

    X = anos_previsao.reshape(-1, 1)

    # Gerando previsao a partir do modelo
    Y_predict = regression.predict(X)
    

    # Linhas de Plot
    linha_previsao_futuro = go.Scatter(
        x = anos_previsao,
        y = Y_predict.tolist(),
        mode='lines',
        line=dict(
            color='red'
        ),
        name="Médias Previstas"
    )

    linha_medias_passado = go.Scatter(
        x = anos_passado,
        y = medias_passado,
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
        title='Temperatúra Média Anual Global (Passado + Previsão)',
        height=700
    )

    # Criando figura e Plotando
    fig = go.Figure(
        data=data,
        layout=layout
    )

    fig.show()