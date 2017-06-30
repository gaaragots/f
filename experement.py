import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from deps.util import log_to_file, tokeniser, save_model, load_model
import os.path
from sklearn.model_selection import train_test_split
from dbn.models import UnsupervisedDBN
import pickle as pkl


model_path = 'model/dbn-model.pkl'
tfidf_matrix_path = 'result/item_feature_matrix.pickle'
tfidf_model_path = 'model/tfidf_model.pickle'
items_path = 'input/items.csv'
full_rating_path = 'input/u.data'

log_file_name = 'Experement result'

# def drawCluster(trs_tfidf_matrix,kmeans):
# 	import numpy as np
# 	import matplotlib.pyplot as plt
# 	from mpl_toolkits.mplot3d import Axes3D
# 	from sklearn import decomposition
# 	from sklearn import datasets

# 	X_norm = (trs_tfidf_matrix - trs_tfidf_matrix.min())/(trs_tfidf_matrix.max() - trs_tfidf_matrix.min())

# 	pca = decomposition.PCA(n_components = 3) #3-dimensional PCA
# 	X = pca.fit_transform(X_norm)

# 	fig = plt.figure(1, figsize=(4, 3))
# 	plt.clf()
# 	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# 	plt.cla()
# 	labels = kmeans.labels_#est.labels_

# 	ax.scatter(X[:, 0], X[:, 1], c=labels.astype(np.float))

# 	plt.show()

# 	print sum(pca.explained_variance_ratio_)

def neiber(itemId):
	items = pd.read_csv(items_path, sep=';', encoding='ISO-8859-1')
	items_train, items_test = train_test_split(items, train_size=0.8, random_state=0)

	train = items_train.reset_index()
	from sklearn.metrics.pairwise import cosine_similarity

	# get description
	desc = items[items['movie id'] == itemId]['movie desription']
	
	# tf-idf representation
	tfidf = tfidf_model.transform(desc.values.astype('U'))
	
	# select relevant features
	dbn_tfidf = dbn.transform(tfidf.A)
	
	# get the cluster
	label = kmeans.predict(dbn_tfidf)
	
	# get items of the same cluster
	items_cluster = trs_tfidf_matrix[kmeans.labels_== label]
	
	# calculate sim (cos sim)
	sim = cosine_similarity(dbn_tfidf, items_cluster)
	
	#return sorted result
	x = train[kmeans.labels_== label].reset_index()['movie id']
	y = sim[0]
	df = pd.DataFrame([x, y]).transpose()
	df.columns = ['movie id', 'similarity']
	df['movie id'] = df['movie id'].astype(int)
	return df.sort_values('similarity', ascending=False).reset_index()[['movie id', 'similarity']]

@log_to_file(log_file_name)
def main(tfidfModel=None, tfidfMatrix=None, dbn_model=None, kmeans_model=None):
	dbn = UnsupervisedDBN.load(model_path)

	tfidf_model = pkl.load(open(tfidf_model_path, 'rb'))
	tfidf_matrix = pkl.load(open(tfidf_matrix_path, 'rb'))

	trs_tfidf_matrix = dbn.transform(tfidf_matrix.A)

	kmeans = KMeans(n_clusters=5, random_state=0).fit(trs_tfidf_matrix)

	# drawCluster(trs_tfidf_matrix,kmeans)

	items = pd.read_csv(items_path, sep=';', encoding='ISO-8859-1')
	items_train, items_test = train_test_split(items, train_size=0.8, random_state=0)

	full_rating = pd.read_csv(full_rating_path, sep='\t')
	full_rating.columns=['user id','movie id','rating','timestamp']

	rating_test = pd.merge(full_rating, items_test, on='movie id')[['user id','movie id','rating','timestamp']]
	rating_train = pd.merge(full_rating, items_train, on='movie id')[['user id','movie id','rating','timestamp']]
	rating_test.head()

	# print rating_test.head(3)
	# calculate the rating
	predictions = []
	real_rating = []
	for i, df in rating_test.iterrows():
		# list of items that have been rated by the curent user
		pdData = pd.merge(rating_train[rating_train['user id'] == df['user id']], neiber(df['movie id']), how='inner', on=['movie id']).sort_values('similarity', ascending=False)
		
		if pdData.shape[0] > 0:
		
			pdData = pdData[:30]
			prediction = sum(pdData['rating'] * pdData['similarity'])/sum(pdData['similarity'])

			predictions.append(prediction)
			real_rating.append(df['rating'])
		else:
			print df['user id'], '\t', df['movie id']

	from sklearn.metrics import mean_absolute_error
	print 'MAE error: ', mean_absolute_error(real_rating, predictions)

	for i in range(len(real_rating)):
		print  real_rating[i], '\t', predictions[i]

if __name__ == '__main__':
	main()