import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns

 df = pd.read_csv('CC_GENERAL.csv')

 df.head()

 df.describe()

 df.info()

 df.isna().mean()*100

 df.drop(['CUST_ID'], axis=1, inplace=True)

 df.dropna(subset=['CREDIT_LIMIT'], inplace=True)

 df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)

 plt.figure(figsize=(20,35))
 for i, col in enumerate(df.columns):
     if df[col].dtype != 'object':
         ax = plt.subplot(9, 2, i+1)
         sns.kdeplot(df[col], ax=ax)
         plt.xlabel(col)

 plt.show()

 cols = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']

 for col in cols:
     df[col] = np.log(1 + df[col])

 plt.figure(figsize=(15,20))
 for i, col in enumerate(cols):
     ax = plt.subplot(6, 2, i+1)
     sns.kdeplot(df[col], ax=ax)
 plt.show()

 plt.figure(figsize=(12,12))
 sns.heatmap(df.corr(), annot=True)
 plt.show()

 from sklearn.decomposition import PCA

 pca = PCA(n_components=0.95)
 X_red = pca.fit_transform(df)

 from sklearn.cluster import KMeans

 kmeans_models = [KMeans(n_clusters=k, random_state=23).fit(X_red) for k in range (1, 10)]
 innertia = [model.inertia_ for model in kmeans_models]

 plt.plot(range(1, 10), innertia)
 plt.title('Elbow method')
 plt.xlabel('Number of Clusters')
 plt.ylabel('WCSS')
 plt.show()




 from sklearn.metrics import silhouette_score

 silhoutte_scores = [silhouette_score(X_red, model.labels_) for model in kmeans_models[1:4]]
 plt.plot(range(2,5), silhoutte_scores, "bo-")
 plt.xticks([2, 3, 4])
 plt.title('Silhoutte scores vs Number of clusters')
 plt.xlabel('Number of clusters')
 plt.ylabel('Silhoutte score')
 plt.show()


 from sklearn.metrics import silhouette_score

 kmeans = KMeans(n_clusters=2, random_state=23)
 kmeans.fit(X_red)

 print('Silhoutte score of our model is ' + str(silhouette_score(X_red, kmeans.labels_)))


 df['cluster_id'] = kmeans.labels_

 for col in cols:
     df[col] = np.exp(df[col])

 plt.figure(figsize=(10,6))
 sns.scatterplot(data=df, x='ONEOFF_PURCHASES', y='PURCHASES', hue='cluster_id')
 plt.title('Distribution of clusters based on One off purchases and total purchases')
 plt.show()

 plt.figure(figsize=(10,6))
 sns.scatterplot(data=df, x='CREDIT_LIMIT', y='PURCHASES', hue='cluster_id')
 plt.title('Distribution of clusters based on Credit limit and total purchases')
 plt.show()

 kmeans = KMeans(n_clusters=3, random_state=23)
 kmeans.fit(X_red)

 df['cluster_id'] = kmeans.labels_



 plt.figure(figsize=(10,6))
 sns.scatterplot(data=df, x='ONEOFF_PURCHASES', y='PURCHASES', hue='cluster_id')
 plt.title('Distribution of clusters based on One off purchases and total purchases')
 plt.show()


 plt.figure(figsize=(10,6))
 sns.scatterplot(data=df, x='CREDIT_LIMIT', y='PURCHASES', hue='cluster_id')
 plt.title('Distribution of clusters based on Credit limit and total purchases')
 plt.show()
