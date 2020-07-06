import pandas as pd
from graficoGeral import graficoGeral, graficoDiferencaGeral
from graficoBrasil import graficoBrasil
from featureSelection import featureSelection
from regression import regression
# le arquivos csv em dataframes
temperaturas_globais = pd.read_csv('data/GlobalTemperatures.csv')
temperaturas_globais_paises = pd.read_csv('data/GlobalLandTemperaturesByCountry.csv')
#temperaturas_globais_cidades = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')

# Converte data para formato datetime
temperaturas_globais['dt'] = pd.to_datetime(temperaturas_globais['dt'])
temperaturas_globais_paises['dt'] = pd.to_datetime(temperaturas_globais_paises['dt'])
#temperaturas_globais_cidades['dt'] = pd.to_datetime(temperaturas_globais_cidades['dt'])
"""
# adiciona coluna de ano
temperaturas_globais['year'] = temperaturas_globais['dt'].dt.year
temperaturas_globais_paises['year'] = temperaturas_globais_paises['dt'].dt.year
temperaturas_globais_cidades['year'] = temperaturas_globais_cidades['dt'].dt.year

# Remove dados iniciais onde grau de incerteza eh muito alto
temperaturas_globais = temperaturas_globais[temperaturas_globais['year'] >= 1825]
temperaturas_globais_paises = temperaturas_globais_paises[temperaturas_globais_paises['year'] >= 1825]
temperaturas_globais_cidades = temperaturas_globais_cidades[temperaturas_globais_cidades['year'] >= 1825]

# Remove medicoes com valores nulos
temperaturas_globais.dropna(axis=0, inplace=True)
temperaturas_globais_paises.dropna(axis=0, inplace=True)
temperaturas_globais_cidades.dropna(axis=0, inplace=True)

# Grafico com aumento da temperatura ao longo dos anos
figGeral = graficoGeral(temperaturas_globais)
figGeral.show()

# Grafico com aumento da temperatura no Brasil ao longo dos anos
figBrasil = graficoBrasil(temperaturas_globais_paises)
figBrasil.show()

# Grafico com top paises com maior diferenca entre temperatura média ao longo dos anos e temperatura máxima (que tiveram maior aumento)
figDiferenca = graficoDiferencaGeral(temperaturas_globais_cidades)
figDiferenca.show()
"""
# Verifica as features mais relacionadas ao target
featureSelection(temperaturas_globais)
regression(temperaturas_globais)