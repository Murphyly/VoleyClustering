# -*- coding: utf-8 -*-
## Importando bibliotecas básicas
"""

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from scipy import stats

"""## Importando a base de dados"""

#Montar o google drive para importar a pasta da base
from google.colab import drive
drive.mount('/content/drive')

#Importando a pasta da base
base_path_men = '/content/drive/MyDrive/FIBV_2020_Statistics_PreliminaryRound/Men_Statistics'
base_path_women = '/content/drive/MyDrive/FIBV_2020_Statistics_PreliminaryRound/Women_Statistics'

def load_and_process_csvs(base_path):
    all_dataframes = []

    for file in os.listdir(base_path):
        if file.endswith('.csv'):
            file_path = os.path.join(base_path, file)
            df = pd.read_csv(file_path)

            # Remove a coluna 'Rank' se existir
            df = df.drop(columns=['Rank'], errors='ignore')

            # Adiciona o DataFrame à lista
            all_dataframes.append(df)

    # Concatena todos os DataFrames verticalmente
    combined_data = pd.concat(all_dataframes, axis=0, ignore_index=True)
    return combined_data


# Carrega e processa os dados
data_men = load_and_process_csvs(base_path_men)
data_women = load_and_process_csvs(base_path_women)

# Mescla os dados de homens e mulheres com base nas colunas 'ShirtNumber' e 'Name'
combined_data = pd.concat([data_men, data_women], axis=0, ignore_index=True)

# Consolida as colunas, eliminando valores NaN ao combinar linhas com os mesmos 'ShirtNumber' e 'Name'
consolidated_data = combined_data.groupby(['ShirtNumber', 'Name'], as_index=False).first()

# Remover a coluna 'ShirtNumber' e 'Name' do DataFrame, pois só precisavamos para juntar as bases de dados
consolidated_data = consolidated_data.drop(columns=['ShirtNumber', 'Name'])

# Carrega apenas os dados númericos
numeric_data = consolidated_data.select_dtypes(include=['float64', 'int64'])

# Verificar o resultado
print(consolidated_data.head())
print(consolidated_data.shape)

consolidated_data

"""## Análise dos dados"""

# frequência absoluta
team_counts = consolidated_data['Team'].value_counts()

# frequência absoluta acumulada
team_counts_cum = team_counts.cumsum()

# frequência relativa
team_counts_rel = team_counts / team_counts.sum()

# frequência relativa acumulada
team_counts_rel_cum = team_counts_rel.cumsum()

team_freq_df = pd.DataFrame({
    'Frequência Absoluta': team_counts,
    'Frequência Absoluta Acumulada': team_counts_cum,
    'Frequência Relativa (%)': team_counts_rel * 100,
    'Frequência Relativa Acumulada (%)': team_counts_rel_cum * 100
})

team_freq_df

plt.figure(figsize=(10, 6))
sns.barplot(x=team_counts.index, y=team_counts.values, palette='viridis')
plt.xlabel('Team')
plt.ylabel('Frequência')
plt.title('Frequencia dos Times')
plt.xticks(rotation=45)
plt.show()

"""### Análise dos times

SRB e USA têm a maior frequência, com 46 registros cada, essas equipes têm um número significativamente maior de registros em comparação com outras, como por exemplo DOM que possui apenas 14 registros. Podemos perceber ainda uma diversidade de times.
"""

mean_values = numeric_data.mean()
median_values = numeric_data.median()
mode_values = numeric_data.mode().iloc[0]

stats_df = pd.DataFrame({
    'Média': mean_values,
    'Mediana': median_values,
    'Moda': mode_values
})

stats_df.style.set_table_attributes('style="font-size: 12px; color: black;"').set_caption('Estatística descritiva dos dados')

"""Avaliando a média, mediana e moda da base de dados de Vólei podemos perceber algumas características, tais como

**Distribuições Simétricas**: Muitas variáveis mostram médias e medianas próximas, o que podemos deduzir como distribuições simétricas.

**Outliers**: Há muitas variáveis onde a média é significativamente maior que a mediana, o que nós indica a presença de outliers que precisamos remover no momento de pré processamento para não prejudicar o agrupamento.
"""

# Calcular os valores máximos
max_values = numeric_data.max()

# Calcular os valores mínimos
min_values = numeric_data.min()

# Calcular os percentis (25º, 50º, 75º)
percentiles = numeric_data.quantile([0.25, 0.5, 0.75])

# Criar um DataFrame para exibir essas estatísticas
stats_df = pd.DataFrame({
    'Máximo': max_values,
    'Mínimo': min_values,
    '25º Percentil': percentiles.loc[0.25],
    '50º Percentil (Mediana)': percentiles.loc[0.5],
    '75º Percentil': percentiles.loc[0.75]
})

print(stats_df)

numeric_data = consolidated_data.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()

scaled_data = scaler.fit_transform(numeric_data)

scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

plt.figure(figsize=(12, 8))
sns.boxplot(data=scaled_df)
plt.title('Box Plot dos Dados Normalizados')
plt.xticks(rotation=90)
plt.show()

numeric_columns = numeric_data.columns

plt.figure(figsize=(15, len(numeric_columns) * 5))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns), 1, i)
    sns.boxplot(x='Team', y=col, data=consolidated_data)
    plt.title(f'Boxplot of {col} by Team')
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

"""**Outliers**: Existem alguns outliers significativos, especialmente nas variáveis relacionadas a ações de jogo como "Total Attempts", "Serve Receptions", e "Running Sets". Esses outliers indicam que, embora a maioria dos jogadores tenha desempenhos moderados, alguns poucos se destacam significativamente em determinadas ações.

**Distribuição Assimétrica:** Diferente da análise quando observamos apenas a média e mediana, aqui muitas variáveis apresentam distribuições assimétricas, com uma concentração de valores mais baixos e caudas longas à direita, sugerindo que poucos jogadores têm desempenhos muito superiores aos demais.

**Variabilidade:** Variáveis como "Efficiency %", "Success %", e "Average Per Set" mostram uma variabilidade menor, um dos motivos pode ser a alta presença de NAN pois nem todos os jogadores possuem esse valor definido.

## Pré processamento
"""

# Trata a questão das colunas com muitos NaN, preenhendo com a média
numeric_columns = numeric_data.columns

# Preencher valores NaN nas colunas numéricas com a média de cada coluna
consolidated_data[numeric_columns] = consolidated_data[numeric_columns].fillna(consolidated_data[numeric_columns].mean())

consolidated_data

"""### Remoção de outliers"""

def remove_outliers_iqr(df):
    df_clean = df.copy()
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Criar máscara para valores dentro dos limites
        mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
        df_clean = df_clean[mask]

    return df_clean

df_clean = remove_outliers_iqr(consolidated_data)

df_clean

"""### Normalização dos dados"""

numeric_data = df_clean.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()

scaled_data = scaler.fit_transform(numeric_data)

scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

plt.figure(figsize=(12, 8))
sns.boxplot(data=scaled_df)
plt.title('Box Plot dos Dados Normalizados')
plt.xticks(rotation=90)
plt.show()

"""## PCA"""

num_components=2

# Aplica PCA
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(scaled_data)

# Cria um DataFrame com os componentes principais
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(num_components)])

print('Taxa de variância:\n', pca.explained_variance_ratio_)

variances = np.array(pca.explained_variance_ratio_)
cumulative_variance = np.cumsum(variances)
print(cumulative_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(variances) + 1), np.cumsum(variances), marker='o', linestyle='--', color='b')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Variância Explicada Acumulada por Número de Componentes')
plt.grid(True)
plt.show()

"""### Análise PCA

A partir do cálculo da taxa de variancia e acumulativa dessa, foi possivel perceber que com 2 componentes é possível explicar ~66% da variância dos dados, com 3 e 4 componentes a diferença da txa de variância accumulativa não foi grande. Não foi tão grande, então sem aumentar a complexidade, mas conseguindo representar uma maior proporção da variância.

## Agrupamento particional - K-Means
"""

# DataFrame com as colunas PC1 e PC2
X = pca_df[['PC1', 'PC2']]

# Determina o número ideal de clusters
num_clusters_range = range(2, 11)
distortions = []

for k in num_clusters_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X)
    distortions.append(kmeans_temp.inertia_)

plt.plot(num_clusters_range, distortions, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Distorção')
plt.title('Definição de nº de grupos - Elbow Method')
plt.show()

"""### Número de clusters avaliando por Elbow Method

Percebe-se que o ponto de inflexão da curva se dá mais próximo de 5, dessa forma podemos testar com esse númnero de clusters
"""

# Aplica K-means
num_clusters_kmeans = 5

kmeans = KMeans(n_clusters=num_clusters_kmeans, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Adiciona rótulos de cluster ao DataFrame
pca_df['Cluster'] = kmeans_labels

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotar os pontos com cores diferentes para cada cluster
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', s=50)

# Adicionar uma barra de cores
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Agrupamento K-means- Componentes PCA')

plt.show()

"""### SIlhoette score kmeans

Com 2 componentes e 5 clusters foi possível obter um maior indíce de silhouette
"""

silhouette_avg_kmeans = silhouette_score(X, kmeans_labels)
silhouette_vals_kmeans = silhouette_samples(X, kmeans_labels)

silhouette_avg_kmeans

silhouette_vals_table_kmeans = silhouette_samples(X, kmeans_labels)
silhouette_summary_kmeans = pd.DataFrame({'Cluster': kmeans_labels, 'Silhouette Score': silhouette_vals_table_kmeans}).groupby('Cluster').mean()

silhouette_summary_kmeans

"""## Agrupamento hierárquico"""

# Define o método de linkage
linkage_method = 'average'  # linkage = ‘ward’, ‘complete’, ‘average’, ‘single’

num_clusters_agglo = 3

# Aplica o Agglomerative Clustering com o método de linkage definido
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters_agglo, affinity='euclidean', linkage=linkage_method)
#affinity = 'euclidean', 'manhattan', 'cosine', or 'precomputed'
agg_clustering_labels = agg_clustering.fit_predict(pca_df[['PC1', 'PC2']])

pca_df['Cluster'] = agg_clustering_labels

# Plota o gráfico de dispersão
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', s=50)

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'Agrupamento aglomerativo ({linkage_method})')

plt.show()

"""### Silhouette agrupamento aglomerativo

O average link foi o que retornou o maior indíce, porém o gráfico de dispersão como ward ficou com grupos mais bem definidos visualmente

Definindo affinity como euclidean ou manhattan retornou o mesmo indíce
"""

silhouette_avg_agglo = silhouette_score(X, agg_clustering_labels)
silhouette_vals_agglo = silhouette_samples(X, agg_clustering_labels)

silhouette_avg_agglo

"""### Dendograma

O dendograma utilizando a linkage average ficou disforme, enquanto o ward ficou bem mais definido. Após refazer o pré-processamento, se ajustou
"""

# Plota o dendograma
plt.figure(figsize=(10, 7))
linked = linkage(X, linkage_method)
dendrogram(linked, orientation='top', distance_sort='descending')
plt.title('Dendograma')
plt.show()

"""### Silhouette score

Quando dividimos em quatro grupos e usando complete link, percebemos que a tabela de silhouette score estava com números abaixo de 0.5, o ward também deu valores bem baixos. Após a mudança para average link e 3 clusters utilizando o elbow method, podemos perceber na tabela abaixo a melhora do indíce silhouette
"""

silhouette_vals_table = silhouette_samples(X, agg_clustering_labels)
silhouette_summary = pd.DataFrame({'Cluster': agg_clustering_labels, 'Silhouette Score': silhouette_vals_table}).groupby('Cluster').mean()

silhouette_summary

silhouette_summary['Silhouette Score'].plot(kind='line', figsize=(8, 4), title='Silhouette Score')
plt.gca().spines[['top', 'right']].set_visible(False)
