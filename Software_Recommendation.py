import json, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from datetime import datetime

file_name = "training_dataset.json"
k = 3

def find_similar_properties(file_name, k):
    with open(file_name) as f:
        data = json.load(f)
    
    subject = data.get("subject", [])
    properties = data.get("properties", [])
    comps = data.get("comps", [])
    

find_similar_properties(file_name, k)