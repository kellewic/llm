import joblib, modal, pickle, re, sys

import numpy as np

from agents import EnsembleAgent, FrontierAgent, RandomForestAgent, SpecialistAgent
from dotenv import find_dotenv, load_dotenv
from product_database import ProductDatabase
from sklearn.ensemble import RandomForestRegressor
from testing import Tester

load_dotenv(dotenv_path=find_dotenv())

with open('test.pkl', 'rb') as file:
    test = pickle.load(file)


db = ProductDatabase()
collection = db.create_or_get_collection()

#ensemble_agent = EnsembleAgent(collection)
#ensemble_agent.disable_logging()

#Tester.test(ensemble_agent.price, test)







sys.exit()

#import joblib
#import pandas as pd
#from sklearn.linear_model import LinearRegression
#from tqdm import tqdm

#frontier_agent = FrontierAgent(collection)
#frontier_agent.disable_logging()

#specialist_agent = SpecialistAgent()
#specialist_agent.disable_logging()

#random_forest_agent = RandomForestAgent()
#random_forest_agent.disable_logging()

#specialists = []
#frontiers = []
#random_forests = []
#prices = []
#for item in tqdm(test[1000:1250]):
#    specialists.append(specialist_agent.price(item))
#    frontiers.append(frontier_agent.price(item))
#    random_forests.append(random_forest_agent.price(item))
#    prices.append(item.price)

#mins = [min(s,f,r) for s,f,r in zip(specialists, frontiers, random_forests)]
#maxes = [max(s,f,r) for s,f,r in zip(specialists, frontiers, random_forests)]

#X = pd.DataFrame({
#    'Specialist': specialists,
#    'Frontier': frontiers,
#    'RandomForest': random_forests,
#    'Min': mins,
#    'Max': maxes,
#})

#y = pd.Series(prices)

#np.random.seed(42)
#lr = LinearRegression()
#lr.fit(X, y)

#feature_columns = X.columns.tolist()

#for feature, coef in zip(feature_columns, lr.coef_):
#    print(f"{feature}: {coef:.2f}")

#print(f"Intercept={lr.intercept_:.2f}")

#joblib.dump(lr, 'ensemble_model.pkl')

