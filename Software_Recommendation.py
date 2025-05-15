# Import necessary libraries
import json, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from datetime import datetime

#These are our configuration variables
file_name = "training_dataset.json"
k = 3

#This is used to parse the date to a datetime object.
def parse_effective_date(date_str):
    if date_str == "N/A":
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None

# Here we are defining a function to calculate the age of the property.
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
    effective_date = parse_effective_date(effective_date_str)
    effective_year = effective_date.year if effective_date else None

    procecessed_props = []

    # Here we will process the subject's key property characterisitics.
    subject_features = {
        'id':'subject',
        'gla': subject.get("gla"),
        "rooms": subject.get("room_total", 0),
        "age": get_age(subject.get("year_built", "N/A"), effective_date_str),
        "structure_type": str(subject.get("structure_type"))
    }
    procecessed_props.append(subject_features)

    # Here we will process the key property characterisitics of the properties.
    for i, property in enumerate(properties):
        prop_features = {
            'id': "property",
            'gla': property.get("gla"),
            "rooms": property.get("room_total", 0),
            "age": get_age(property.get("year_built", "N/A"), effective_date_str),
            "structure_type": str(property.get("structure_type"))
        }
        procecessed_props.append(prop_features)
    
    df = pd.DataFrame(procecessed_props)

    # Here we will process the key property characterisitics of the comps.
    df = pd.get_dummies(df, columns=["structure_type"], drop_first=True)

    # Here we're splitting the dataset into two parts: the subject and the properties.
    subject_df = df[df['id'] == 'subject'].copy()
    properties_df = df[df['id'] != 'subject'].copy()

    # Here we are splitting the columns into two parts: the numerical features and the one-hot encoded features.
    numerical_features = ['gla', 'rooms', 'age']
    one_hot_columns = [col for col in df.columns if col.startswith('structure_type_')]
    # Here we're combining the numerical features and the one-hot encoded features into a single list.
    final_feature_columns = numerical_features + one_hot_columns

    # Here we're ensuring that all columns are present in both the subject and properties dataframes.
    for col in final_feature_columns:
        if col not in subject_df.columns:
            subject_df[col] = 0
        if col not in properties_df.columns:
            properties_df[col] = 0
    
    # Here we're reordering the columns to match the final feature columns.
    subject_df = subject_df[final_feature_columns]
    candidate_df = properties_df[final_feature_columns]

    # Here we're using the scaler to scale the data. It will be between 0 and 1 and could change based on the minimum and maximum values of the dataset.
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([subject_df, candidate_df], ignore_index=True))

    # Here we're applying the scaler to the subject and properties dataframes.
    # This will scale the data to be between 0 and 1 and to ensure that features like GLA and room count are equally important.
    subject_scaled_df = pd.DataFrame(scaler.transform(subject_df), columns=final_feature_columns, index = subject_df.index)
    candidates_scaled_df = pd.DataFrame(scaler.transform(candidate_df), columns=final_feature_columns, index = candidate_df.index)

    # Here we're adding the original id to the scaled dataframe for the properties.
    # This would be used to identify the properties in the original dataset.
    candidates_scaled_df['original_id'] = candidate_df.index.map(df.set_index('id').id)
    candidates_scaled_df['address'] = candidates_scaled_df.apply(lambda row: euclidean(subject_scaled_df.iloc[0], row[final_feature_columns]), axis=1)

    # Here we're returning the row representing the subject property, the scaled properties dataframe, and the list of features used for distance calculation (euclidean distance calculations).
    return subject_scaled_df.iloc[0], candidates_scaled_df, final_feature_columns
def find_similar_properties(file_name, k):
    with open(file_name) as f:
        data = json.load(f)
    
    subject = data.get("subject", [])
    properties = data.get("properties", [])
    comps = data.get("comps", [])
    
    print (preprocess_property_data(subject, properties, comps[0].get("effective_date")))

find_similar_properties(file_name, k)