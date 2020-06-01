import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.subplots as sub


def graficoBrasil(temperaturas):
    anos = []
    medias = []
    incerteza_medias = []
    mins = []
    incerteza_mins = []
    maxs = []
    incerteza_maxs = []

    temps_brasil = temperaturas[temperaturas['Country'] == 'Brazil']

    # Obtendo lista com anos
    anos = np.unique(temps_brasil['dt'].apply(lambda x: x[:4]))


    for ano in anos:
        medias.append(temps_brasil[temps_brasil['dt'].apply(lambda x: x[:4]) == ano]['AverageTemperature'].mean())
        incerteza_medias.append(temps_brasil[temps_brasil['dt'].apply(lambda x: x[:4]) == ano]['AverageTemperatureUncertainty'].mean())

    # MEDIAS

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

    data = [linha_media_superior, linha_media_inferior, linha_media]
    layout = dict(
        xaxis=dict(title='ano'),
        yaxis=dict(title='temperatura'),
        title='Temperatúra média Brasil'
    )

    # Montando figura
    fig = go.Figure(
        data=data,
        layout=layout
    )

    return fig