import pandas as pd
from graficoGeral import graficoGeral, graficoDiferencaGeral
from graficoBrasil import graficoBrasil
from featureSelection import featureSelection
from regression import regression,regressionMultipleParameters
from regressionTemperatura import regressao_ano_temp_media, previsao_temp_media_futura, regressao_ano_temp_media_BRASIL, previsao_temp_media_futura_BRASIL
from correlation import correlation, predictLandAndOcean,regressionLandToOcean,graficoPredicao

pd.set_option('display.max_columns', None)

# le arquivos csv em dataframes
temperaturas_globais = pd.read_csv('data/GlobalTemperatures.csv')
temperaturas_globais_paises = pd.read_csv('data/GlobalLandTemperaturesByCountry.csv')
temperaturas_globais_cidades = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')

# Converte data para formato datetime
temperaturas_globais['dt'] = pd.to_datetime(temperaturas_globais['dt'])
temperaturas_globais_paises['dt'] = pd.to_datetime(temperaturas_globais_paises['dt'])
temperaturas_globais_cidades['dt'] = pd.to_datetime(temperaturas_globais_cidades['dt'])

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

# Verifica as features mais relacionadas ao target
featureSelection(temperaturas_globais)
regression(temperaturas_globais)
regressionMultipleParameters(temperaturas_globais)

# Regressoes para Media Anual de Temperatura Global
modelo = regressao_ano_temp_media(temperaturas_globais)
previsao_temp_media_futura(temperaturas_globais, modelo, 2101)

# Regressoes para Media Anual de Temperatura Global
modelo = regressao_ano_temp_media(temperaturas_globais)
previsao_temp_media_futura(temperaturas_globais, modelo, 2101)
modelo_brasil = regressao_ano_temp_media_BRASIL(temperaturas_globais_cidades)
previsao_temp_media_futura_BRASIL(temperaturas_globais_cidades, modelo_brasil, 2101)

# Correlação Temp Media com Temp LandAndOcean
correlation(temperaturas_globais)
predictLandAndOcean(temperaturas_globais, 5.518) # pode ser maneiro usar aqui em vez de hardcoded um valor que foi previsto anterior
modelo = regressionLandToOcean(temperaturas_globais)
graficoPredicao(temperaturas_globais,modelo,20)