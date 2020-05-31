import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.subplots as sub


# inicializacao variaveis
temperaturas_globais = pd.read_csv('data/GlobalTemperatures.csv')
anos = []
medias = []
incerteza_medias = []
mins = []
incerteza_mins = []
maxs = []
incerteza_maxs = []


# Obtendo lista com anos
anos = np.unique(temperaturas_globais['dt'].apply(lambda x: x[:4]))


for ano in anos:
    medias.append(temperaturas_globais[temperaturas_globais['dt'].apply(lambda x: x[:4]) == ano]['LandAverageTemperature'].mean())
    incerteza_medias.append(temperaturas_globais[temperaturas_globais['dt'].apply(lambda x: x[:4]) == ano]['LandAverageTemperatureUncertainty'].mean())

    mins.append(temperaturas_globais[temperaturas_globais['dt'].apply(lambda x: x[:4]) == ano]['LandMinTemperature'].mean())
    incerteza_mins.append(temperaturas_globais[temperaturas_globais['dt'].apply(lambda x: x[:4]) == ano]['LandMinTemperatureUncertainty'].mean())

    maxs.append(temperaturas_globais[temperaturas_globais['dt'].apply(lambda x: x[:4]) == ano]['LandMaxTemperature'].mean())
    incerteza_maxs.append(temperaturas_globais[temperaturas_globais['dt'].apply(lambda x: x[:4]) == ano]['LandMaxTemperatureUncertainty'].mean())

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

fig.show()

