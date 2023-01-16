/* Задание:
1.   Используйте метод K-средних и метод DBSCAN на самостоятельно сгенерированной выборке с количеством кластеров не менее 4. Для увеличения числа кластеров при генерации можно задать количество центров в функции make_blobs через параметр centers.
2.   Используйте эти же два метода на датасете [Mall_Customers](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).
3.   Для каждого метода необходимо построить график.
*/

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd 
from sklearn.cluster import DBSCAN, KMeans 
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize']=(12,8)

# генерируем данные 
X, y = make_blobs(n_samples=200, random_state=3, centers=4) 
plt.scatter(X[:, 0], X[:, 1])  

# делаем кластеризацию с помощью метода DBSCAN 
clustering = DBSCAN(eps=1, min_samples=2).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=clustering);
print(clustering)

# разбиваем датасет кластеры и обучаем модель 
kmeansModel = KMeans(n_clusters=2,random_state=0) 
kmeansModel.fit(X) 
labels = kmeansModel.labels_ 
# визуализация полученных результатов 
plt.scatter(X[:, 0], X[:, 1], c=labels) 
print(labels)

# воспользуемся методом локтя для определения оптимального количества кластеров 
criteries = [] 
for k in range(2, 10): 
    kmeansModel = KMeans(n_clusters=k, random_state=3) 
    kmeansModel.fit(X) 
    criteries.append(kmeansModel.inertia_) 
print(criteries)
plt.plot(range(2, 10), criteries);

# теперь смотрим как будет выглядеть с 4 кластерами, которые мы получили из метода локтя 
kmeansModel = KMeans(n_clusters=4, random_state=0) 
kmeansModel.fit(X) 
labels = kmeansModel.labels_ 
plt.scatter(X[:, 0], X[:, 1], c=labels);

data = pd.read_csv("/content/drive/MyDrive/Mall_Customers.csv")
data.head()

# достаем нужные нам данные из датасета 
data = pd.read_csv('/content/drive/MyDrive/Mall_Customers.csv') 
x = data[['Annual Income (k$)','Spending Score (1-100)']].iloc[:, :].values 

plt.scatter(x[:,0], x[:,1]);

clustering = DBSCAN(eps=9, min_samples=3).fit_predict(x)
print(clustering)
plt.scatter(x[:,0], x[:,1], c=clustering);

criteries = []
for k in range(2,10):
  kmeansModel=KMeans(n_clusters=k, random_state=3)
  kmeansModel.fit(x)
  criteries.append(kmeansModel.inertia_)
print(criteries)
plt.plot(range(2,10), criteries)

kmeansModel=KMeans(n_clusters=5, random_state=0)
kmeansModel.fit(x) 
labels = kmeansModel.labels_
plt.scatter(x[:,0], x[:,1], c=labels);

