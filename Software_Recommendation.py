# Import necessary libraries
import json, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from datetime import datetime

#These are our configuration variables
file_name = "training_dataset.json"
k = 3

# This is used to parse the date to a datetime object.
# def parse_effective_date(date_str):
#     if date_str == "N/A":
#         return None
#     try:
#         return datetime.strptime(date_str, "%Y-%m-%d").date()
#     except ValueError:
#         return None

def get_age(year_built_or_age, effective_year):
    if year_built_or_age == "N/A":
        return None
    try:
        year_built = int(year_built_or_age)
        #effective_year = parse_effective_date(effective_year)
        if effective_year:
            if effective_year.year - year_built < 0:
                return None
            return effective_year.year - year_built
        else:
            return None
    except ValueError:
        return None

def preprocess_property_data(subject, properties, effective_date_str):
    #effective_date = parse_effective_date(effective_date_str)
    #effective_year = effective_date.year if effective_date else None

    procecessed_props = []

    # Here we will process the subject's key property characterisitics.
    subject_features = {
        'id':'subject',
        'gla': subject.get("gla"),
        "rooms": subject.get("room_total", 0),
        "age": get_age(subject.get("year_built", "N/A"), effective_date_str),
        "structure_type": str(subject.get("structure_type"))
    }
    print (subject_features)

def find_similar_properties(file_name, k):
    with open(file_name) as f:
        data = json.load(f)
    
    subject = data.get("subject", [])
    properties = data.get("properties", [])
    comps = data.get("comps", [])
    
    print (preprocess_property_data(subject, properties, comps[0].get("effective_date")))

find_similar_properties(file_name, k)