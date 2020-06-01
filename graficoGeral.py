import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.subplots as sub


def graficoGeral(temperaturas):
    anos = []
    medias = []
    incerteza_medias = []
    mins = []
    incerteza_mins = []
    maxs = []
    incerteza_maxs = []


    # Obtendo lista com anos
    # anos = np.unique(temperaturas['dt'].apply(lambda x: x[:4]))
    anos = np.unique(temperaturas['year'])

    for ano in anos:
        medias.append(temperaturas[temperaturas['year'] == ano]['LandAverageTemperature'].mean())
        incerteza_medias.append(temperaturas[temperaturas['year'] == ano]['LandAverageTemperatureUncertainty'].mean())

        mins.append(temperaturas[temperaturas['year'] == ano]['LandMinTemperature'].mean())
        incerteza_mins.append(temperaturas[temperaturas['year'] == ano]['LandMinTemperatureUncertainty'].mean())

        maxs.append(temperaturas[temperaturas['year'] == ano]['LandMaxTemperature'].mean())
        incerteza_maxs.append(temperaturas[temperaturas['year'] == ano]['LandMaxTemperatureUncertainty'].mean())

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

    # MINIMOS

    linha_min = go.Scatter(
        x = anos,
        y = mins,
        mode='lines',
        line=dict(
            color='green'
        ),
        name="Mínima"
    )

    linha_min_superior = go.Scatter(
        x = anos,
        y = np.array(mins) + np.array(incerteza_mins),
        mode='lines',
        line=dict(
            color='rgb(86, 168, 86)'
        ),
        name="Mínima Superior"
    )

    linha_min_inferior = go.Scatter(
        x = anos,
        y = np.array(mins) - np.array(incerteza_mins),
        fill= 'tonexty',
        mode='lines',
        line=dict(
            color='rgb(86, 168, 86)'
        ),
        name="Mínima Inferior"
    )


    # MAXIMOS

    linha_max = go.Scatter(
        x = anos,
        y = maxs,
        mode='lines',
        line=dict(
            color='blue'
        ),
        name="Máxima"
    )

    linha_max_superior = go.Scatter(
        x = anos,
        y = np.array(maxs) + np.array(incerteza_maxs),
        mode='lines',
        line=dict(
            color='rgb(91, 91, 194)'
        ),
        name="Máxima Inferior"
    )

    linha_max_inferior = go.Scatter(
        x = anos,
        y = np.array(maxs) - np.array(incerteza_maxs),
        fill= 'tonexty',
        mode='lines',
        line=dict(
            color='rgb(91, 91, 194)'
        ),
        name="Máxima Inferior"
    )

    # Montando figura

    fig = sub.make_subplots(rows=3, cols=1, subplot_titles=("Temperatura Média", "Temperatura Mínima", "Temperatura Máxima"))

    fig.add_trace(linha_media_superior, row=1, col=1)
    fig.add_trace(linha_media_inferior, row=1, col=1)
    fig.add_trace(linha_media, row=1, col=1)

    fig.add_trace(linha_min_superior, row=2, col=1)
    fig.add_trace(linha_min_inferior, row=2, col=1)
    fig.add_trace(linha_min, row=2, col=1)

    fig.add_trace(linha_max_superior, row=3, col=1)
    fig.add_trace(linha_max_inferior, row=3, col=1)
    fig.add_trace(linha_max, row=3, col=1)

    fig.update_layout(xaxis=dict(title='ano'), yaxis=dict(title='temperatura'), height=2000)

    return fig


def graficoDiferencaGeral(temperaturas_cidades):

    paises = temperaturas_cidades.groupby( by = ['Country', 'year'] ).mean().reset_index()

    # Obtem media, min e maximo da temperatura media de cada pais ao longo de todo o periodo de tempo
    mean = paises.groupby(['Country'])['AverageTemperature'].mean().reset_index()
    maximum = paises.groupby(['Country'])['AverageTemperature'].max().reset_index()
    minimum = paises.groupby(['Country'])['AverageTemperature'].min().reset_index()

    # Junta temp media, min e maxima
    difference_min_max = pd.merge(left = minimum, right = maximum, on = ['Country'], suffixes=('_min', '_max'))
    difference_med_max_min = pd.merge(left = difference_min_max, right=mean, on=['Country'])

    # Cria coluna com diferencas
    difference_med_max_min['diff_med_max'] = difference_med_max_min['AverageTemperature_max'] - difference_med_max_min['AverageTemperature']
    difference_med_max_min['diff_med_min'] = difference_med_max_min['AverageTemperature'] - difference_med_max_min['AverageTemperature_min']
    difference_med_max_min['diff_max_min'] = difference_med_max_min['AverageTemperature_max'] - difference_med_max_min['AverageTemperature_min']


    # Montando grafico
    fig = sub.make_subplots(rows=1, cols=3, subplot_titles=("Max(temps anuais) - Média(temps anuais)", "Média(temps anuais) - Min(temps anuais)", "Max(temps anuais) - Min(temps anuais)"))

    fig.update_layout(title="Considerando a temperatura média anual para cada ano, os gráficos mostram a diferença entre a menor média, a maior média e a média das médias, por país")

    top_20_med_max = difference_med_max_min.nlargest(20, 'diff_med_max')
    trace_med_max = go.Bar(
                    x = top_20_med_max['diff_med_max'], 
                    y = top_20_med_max['Country'], 
                    orientation = 'h',
                    marker=dict(
                        color='rgb(222,184,135)', 
                        line=dict( 
                            color='rgb(188,143,143)', 
                            width=0.6)
                        )
                    )

    top_20_med_min = difference_med_max_min.nlargest(20, 'diff_med_min')
    trace_med_min = go.Bar(
                    x = top_20_med_min['diff_med_min'], 
                    y = top_20_med_min['Country'], 
                    orientation = 'h',
                    marker=dict(
                        color='rgb(222,184,135)', 
                        line=dict( 
                            color='rgb(188,143,143)', 
                            width=0.6)
                        )
                    )

    top_20_max_min = difference_med_max_min.nlargest(20, 'diff_max_min')
    trace_max_min = go.Bar(
                    x = top_20_max_min['diff_max_min'], 
                    y = top_20_max_min['Country'], 
                    orientation = 'h',
                    marker=dict(
                        color='rgb(222,184,135)', 
                        line=dict( 
                            color='rgb(188,143,143)', 
                            width=0.6)
                        )
                    )


    fig.add_trace(trace_med_max, row=1, col=1)
    fig.add_trace(trace_med_min, row=1, col=2)
    fig.add_trace(trace_max_min, row=1, col=3)

    return fig