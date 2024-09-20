import pandas as pd 
import requests 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# URLs dos dados
url_dados = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=QYQZGTEZZDKM4F3RSAFB52WNH&taskId=09c01168e057451c7df52d399a6521f3&zip=false'
url_historico = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=QYQZGTEZZDKM4F3RSAFB52WNH&taskId=aa774405d4af47b4c9c2891d4f7b984a&zip=false'
url_historico2 = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=QYQZGTEZZDKM4F3RSAFB52WNH&taskId=4f1a13a2e5313474280cb6a9a6ea5f62&zip=false'
url_historico3 = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=QYQZGTEZZDKM4F3RSAFB52WNH&taskId=508d8599f2f39be05f0524b3f35269af&zip=false'
url_historico4 = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=QYQZGTEZZDKM4F3RSAFB52WNH&taskId=7415b2c71d95148e93641c83079f9683&zip=false'

# Fazendo o download dos dados
resposta_dados = requests.get(url_dados)
resposta_historico = requests.get(url_historico)
resposta_historico2 = requests.get(url_historico2)
resposta_historico3 = requests.get(url_historico3)
resposta_historico4 = requests.get(url_historico4)

# Salvando os dados atuais
if resposta_dados.status_code == 200:
    with open('Goiânia.csv', 'wb') as file:
        file.write(resposta_dados.content)
else:
    print(f"Erro ao acessar a API de dados atuais: {resposta_dados.status_code}")

# Salvando o histórico
if resposta_historico.status_code == 200:
    with open('Goiânia 2024-03-22 to 2024-03-31.csv', 'wb') as file:
        file.write(resposta_historico.content)
else:
    print(f"Erro ao acessar a API de histórico: {resposta_historico.status_code}")

# Salvando o histórico2 
if resposta_historico2.status_code == 200:
    with open('Goiânia 2024-09-03 to 2024-09-10.csv', 'wb') as file:
        file.write(resposta_historico2.content)
else:
    print(f"Erro ao acessar a API de histórico: {resposta_historico2.status_code}")
# Salvando o histórico3 
if resposta_historico3.status_code == 200:
    with open('Goiânia 2024-07-02 to 2024-07-09.csv', 'wb') as file:
        file.write(resposta_historico3.content)
else:
    print(f"Erro ao acessar a API de histórico: {resposta_historico3.status_code}")

# Salvando o histórico4
if resposta_historico4.status_code == 200:
    with open('Goiânia 2023-09-19 to 2023-10-03.csv', 'wb') as file:
        file.write(resposta_historico4.content)
else:
    print(f"Erro ao acessar a API de histórico: {resposta_historico4.status_code}")

# Lendo os arquivos CSV
try:
    dados = pd.read_csv('Goiânia.csv')
    historico = pd.read_csv('Goiânia 2024-03-22 to 2024-03-31.csv')
    historico2 = pd.read_csv('Goiânia 2024-09-03 to 2024-09-10.csv')
    historico3 = pd.read_csv('Goiânia 2024-07-02 to 2024-07-09.csv')
    historico4 = pd.read_csv('Goiânia 2023-09-19 to 2023-10-03.csv')

except Exception as e:
    print(f"Erro ao ler os arquivos CSV: {e}")

# Criando a variável precipbinary
dados['precipbinary'] = dados['preciptype'].apply(lambda x: 1 if x == 'rain' else 0)
historico['precipbinary'] = historico['preciptype'].apply(lambda x: 1 if x == 'rain' else 0)
historico2['precipbinary'] = historico2['preciptype'].apply(lambda x: 1 if x == 'rain' else 0)
historico3['precipbinary'] = historico3['preciptype'].apply(lambda x: 1 if x == 'rain' else 0)
historico4['precipbinary'] = historico4['preciptype'].apply(lambda x: 1 if x == 'rain' else 0)

# Combinando os históricos
historico_combinado = pd.concat([historico, historico2, historico3, historico4], ignore_index=True)

# Selecionando as colunas desejadas
colunas_x = ['precipprob', 'precip', 'precipcover', 'humidity', 'dew', 'cloudcover', 'windspeed', 'windgust', 'tempmax', 
             'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike']
colunas_y = 'precipbinary'

# Filtrando as colunas
try:
    treino_x = historico_combinado[colunas_x]
    treino_y = historico_combinado[colunas_y]
    x = dados[colunas_x]
    y = dados[colunas_y]
except KeyError as e:
    print(f"Erro ao acessar colunas: {e}")

# Normalizando os dados
scaler = StandardScaler()
treino_x = scaler.fit_transform(treino_x)
x = scaler.transform(x)

# Assegurando que treino_y e y sejam vetores unidimensionais
treino_y = treino_y.values.ravel() if treino_y.ndim > 1 else treino_y
y = y.values.ravel() if y.ndim > 1 else y

# Treinando e avaliando o modelo
try:
    modelo = LinearSVC()
    modelo.fit(treino_x, treino_y)
    previsoes = modelo.predict(x)
    acuracia = accuracy_score(y, previsoes) * 100
    print("A acurácia foi %.2f%%" % acuracia)
except Exception as e:
    print(f"Erro ao treinar ou avaliar o modelo: {e}")

# Exibindo a previsão de chuva
for i in range(len(dados)):
    clima = dados.iloc[i]['conditions']
    data = dados.iloc[i]['datetime']
    probabilidade = dados.iloc[i]['precipprob']
    if 'rain' in clima.lower() and probabilidade > 50.0:
        print(f"Possibilidade de chuva em {data}: {clima}")
