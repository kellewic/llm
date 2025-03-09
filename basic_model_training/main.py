import json, math, os, pickle, random, statistics, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter
from dotenv import find_dotenv, load_dotenv
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from huggingface_hub import login
from items import Item
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from testing import Tester

load_dotenv(dotenv_path=find_dotenv())

with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

with open('test.pkl', 'rb') as file:
    test = pickle.load(file)

## TRIVIAL MODEL
## Start with a trival model
def run_average_prices_model(data=train):
	average = statistics.mean([item.price for item in data])

	def average_pricer(item):
		return average

	Tester.test(average_pricer, data)

## Prepare for LinearRegression model
## Create a new "features" field on items, and populate with json parsed from details dict
def do_linear_regression():
	for item in train:
		item.features = json.loads(item.details)

	for item in test:
		item.features = json.loads(item.details)

	feature_count = Counter()
	for item in train:
		for f in item.features.keys():
			feature_count[f] += 1

	def get_weight(item):
		weight_str = item.features.get('Item Weight')

		if weight_str:
			parts = weight_str.split(' ')
			amount = float(parts[0])
			unit = parts[1].lower()

			if unit=="pounds":
				return amount
			elif unit=="ounces":
				return amount / 16
			elif unit=="grams":
				return amount / 453.592
			elif unit=="milligrams":
				return amount / 453592
			elif unit=="kilograms":
				return amount / 0.453592
			elif unit=="hundredths" and parts[2].lower()=="pounds":
				return amount / 100
			else:
				print(weight_str)

		return None

	weights = [get_weight(t) for t in train]
	weights = [w for w in weights if w]
	average_weight = statistics.mean(weights)

	def get_weight_with_default(item):
		return get_weight(item) or average_weight

	def get_rank(item):
		rank_dict = item.features.get("Best Sellers Rank")

		if rank_dict:
			return statistics.mean(rank_dict.values())

		return None

	ranks = [get_rank(t) for t in train]
	ranks = [r for r in ranks if r]
	average_rank = statistics.mean(ranks)

	def get_rank_with_default(item):
		return get_rank(item) or average_rank

	def get_text_length(item):
		return len(item.test_prompt())

	brands = Counter()
	for t in train:
		brand = t.features.get("Brand")

		if brand:
			brands[brand] += 1


	TOP_BRANDS = ["kohler", "hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]
	def is_top_brand(item):
		brand = item.features.get("Brand")
		return brand and brand.lower() in TOP_BRANDS

	def get_features(item):
		return {
			"weight": get_weight_with_default(item),
			"rank": get_rank_with_default(item),
			"text_length": get_text_length(item),
			"is_top_brand": 1 if is_top_brand(item) else 0
		}

	## A utility function to convert features into a pandas dataframe
	def list_to_dataframe(items):
		features = [get_features(item) for item in items]
		df = pd.DataFrame(features)
		df['price'] = [item.price for item in items]

		return df

	train_df = list_to_dataframe(train)
	test_df = list_to_dataframe(test[:250])

	## Traditional Linear Regression
	np.random.seed(42)

	## Separate features and target
	feature_columns = ['weight', 'rank', 'text_length', 'is_top_brand']

	X_train = train_df[feature_columns]
	y_train = train_df['price']
	X_test = test_df[feature_columns]
	y_test = test_df['price']

	# Train a Linear Regression
	model = LinearRegression()
	model.fit(X_train, y_train)

	for feature, coef in zip(feature_columns, model.coef_):
		print(f"{feature}: {coef}")

	print(f"Intercept: {model.intercept_}")

	## Predict the test set and evaluate
	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error: {mse} (${math.sqrt(mse):.2f} off)")
	print(f"R-squared Score: {r2}")

	## Predict price for a new item
	def linear_regression_pricer(item):
		features = get_features(item)
		features_df = pd.DataFrame([features])
		return model.predict(features_df)[0]

	Tester.test(linear_regression_pricer, test)


## Moving on from feature engineering to using text
prices = np.array([float(item.price) for item in train])
documents = [item.test_prompt() for item in train]

## bag-of-words text features + linear regression
def run_bow_lr():
	np.random.seed(42)
	vectorizer = CountVectorizer(max_features=1000, stop_words='english')
	X = vectorizer.fit_transform(documents)
	regressor = LinearRegression()
	regressor.fit(X, prices)

	def bow_lr_pricer(item):
		x = vectorizer.transform([item.test_prompt()])
		return max(regressor.predict(x)[0], 0)

	Tester.test(bow_lr_pricer, test)


## try out some other models
def run_other_model(model=1):
	np.random.seed(42)

	print("Pre-processing docs")
	processed_docs = [simple_preprocess(doc) for doc in documents]
	w2v_model = Word2Vec(sentences=processed_docs, vector_size=400, window=5, min_count=1, workers=12)

	def document_vector(doc):
		doc_words = simple_preprocess(doc)
		word_vectors = [w2v_model.wv[word] for word in doc_words if word in w2v_model.wv]
		return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(w2v_model.vector_size)

	print("Creating feature matrix")
	X_w2v = np.array([document_vector(doc) for doc in documents])

	def word2vec_lr_pricer(item):
		doc = item.test_prompt()
		doc_vector = document_vector(doc)
		return max(0, word2vec_lr_regressor.predict([doc_vector])[0])

	def svr_pricer(item):
		np.random.seed(42)
		doc = item.test_prompt()
		doc_vector = document_vector(doc)
		return max(float(svr_regressor.predict([doc_vector])[0]),0)

	def random_forest_pricer(item):
		doc = item.test_prompt()
		doc_vector = document_vector(doc)
		return max(0, rf_model.predict([doc_vector])[0])

	if model == 1:
		print("Running Word2Vec LinearRegression")
		word2vec_lr_regressor = LinearRegression()
		word2vec_lr_regressor.fit(X_w2v, prices)

		Tester.test(word2vec_lr_pricer, test)
	elif model == 2:
		print("Running LinearSVR")
		svr_regressor = LinearSVR()
		svr_regressor.fit(X_w2v, prices)

		Tester.test(svr_pricer, test)
	else:
		print("Running RandomForestRegressor")
		rf_model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=12)
		rf_model.fit(X_w2v, prices)

		Tester.test(random_forest_pricer, test)


#run_other_model(3)

