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
    if not date_str:
        return datetime.now()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.now()

# This function is used to parse the GLA (Gross Living Area) value.
def safe_parse_gla(gla_value):
    if not gla_value:
        return 0.0
    try:
        cleaned = str(gla_value).replace("SqFt", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0

# Here we are defining a function to calculate the age of the property.
def get_age(year_built_or_age, effective_year):
    if year_built_or_age is None:
        return 0
    try:
        val = int(float(str(year_built_or_age)))
        if 0 < val < 150:
            return val
        elif val >= 1800:
            return effective_year - val
    except ValueError:
        pass
    return 0

def preprocess_property_data(subject, properties, effective_date_str):
    effective_date = parse_effective_date(effective_date_str)
    effective_year = effective_date.year if isinstance(effective_date, datetime) else datetime.now().year

    procecessed_props = []

    # Here we will process the subject's key property characterisitics.
    subject_features = {
        'id':'subject',
        'gla': safe_parse_gla(subject.get("gla")),
        "rooms": subject.get("room_total", 0),
        "age": get_age(subject.get("year_built", "N/A"), effective_year),
        "structure_type": str(subject.get("structure_type"))
    }
    procecessed_props.append(subject_features)

    # Here we will process the key property characterisitics of the properties.
    for i, property in enumerate(properties):
        prop_features = {
            'id': property.get("id", f"property_{i}"),
            'gla': safe_parse_gla(property.get("gla")),
            "rooms": property.get("room_total", 0),
            "age": get_age(property.get("year_built", "N/A"), effective_year),
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
    # ✅ Corrected version:
    candidates_scaled_df['original_id'] = properties_df['id'].values
    candidates_scaled_df['address'] = properties_df.get('address', pd.Series([""] * len(properties_df)))


    # Here we're returning the row representing the subject property, the scaled properties dataframe, and the list of features used for distance calculation (euclidean distance calculations).
    return subject_scaled_df.iloc[0], candidates_scaled_df, final_feature_columns
def find_similar_properties(file_name, k):
    with open(file_name) as f:
        data = json.load(f)
    
    # Here we're extracting the subject property and the properties from the JSON data.
    subject = data.get("subject", [])
    properties = data.get("properties", [])
    comps = data.get("comps", [])
    all_candidates = properties + comps

    # Here we're extracting the effective date from the subject property.
    effective_date_str = subject.get("effective_date", "")

    #Preprocess the data
    subject_scaled, candidates_scaled, final_feature_columns = preprocess_property_data(subject, all_candidates, effective_date_str)

    # Here we're defining the weights for the features used in the distance calculation.
    weights = {
        'gla': 1.0,
        'rooms': 1.0,
        'age': 1.0
    }
    # Here we're adding the weights for the one-hot encoded columns.
    for col in final_feature_columns:
        if col not in weights:
            weights[col] = 0.2  # default weight for one-hot columns

    #Calculating the distances
    distances = []
    subject_vector = subject_scaled[final_feature_columns].values.astype(float) 
    subject_vector *= np.array([weights[col] for col in final_feature_columns]) # Apply weights to the subject vector

    for index , row in candidates_scaled.iterrows():
        candidate_vector = row[final_feature_columns].values.astype(float)
        dist = euclidean(subject_vector, candidate_vector)
        distances.append({
            'id': row['original_id'],
            'distance': dist,
            'address': row['address']
        })
    
    #Putting rhe distances in a dataframe
    distances_df = pd.DataFrame(distances)
    # Here we're sorting the distances in ascending order.
    sorted_distances_df = distances_df.sort_values(by='distance', ascending=True)
    # Here we're resetting the index of the sorted distances dataframe.
    print (f"\nTop {k} most similar properties")
    top_k_properties = []
    output_rows = []
    for i in range(min(k, len(sorted_distances_df))):
       # Here we're getting the top 3 properties based on the sorted distances.
       neighbour = sorted_distances_df.iloc[i]
       # Here we're getting the original property details from the properties list.
       original_property_detail = next((prop for prop in properties if prop.get("id") == neighbour['id']), None)
       is_comp = any(str(comp.get("id")) == str(neighbour['id']) for comp in comps)

        # Here we're printing the details of the top 3 properties.
       print (f"\nRank {i+1}:")
       print (f"Property ID: {neighbour['id']}")
       print (f"Distance: {neighbour['distance']}")
       print (f"Address: {neighbour['address']}")
       print (f"Is Comp: {'yes' if is_comp else 'no'}")
       if original_property_detail:
           print (f"Property Details: {json.dumps(original_property_detail, indent=2)}")
           row = {
                'rank': i + 1,
                'id': neighbour['id'],
                'distance': neighbour['distance'],
                'address': neighbour['address'],
                'is_comp': is_comp,
                'property_details': original_property_detail,
                'gla': original_property_detail.get("gla"), # type: ignore
                'Bedrooms': original_property_detail.get("room_total", 0), # type: ignore
           }
           output_rows.append(row)
           top_k_properties.append(original_property_detail)

    # Here we're saving the output to a CSV file.
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv('output.csv', index=False)
    print (f"\nOutput saved to output.csv")

    return top_k_properties

if __name__ == "__main__":
    # Call the function to find similar properties
    print ("Beginning the software recommendation process...")
    similar_props = find_similar_properties(file_name, k)
    if similar_props:
        print (f"\nFound {len(similar_props)} similar properties.")
    else:
        print ("No similar properties found.")

find_similar_properties(file_name, k)