# 🏡 Property Recommendation System

This project is a Python-based property recommendation tool that uses feature similarity to identify the top **k** comparable properties to a given subject property. It reads property data from a JSON file and outputs the most similar properties using a distance-based algorithm.

---

## 📂 Project Structure

- `training_dataset.json`: Input file containing subject property, candidate properties, and known comps.
- `output.csv`: Output file with the top k most similar properties.
- `property_recommender.py`: Main script performing property preprocessing, normalization, similarity calculation, and result export.

---

## ⚙️ How It Works

The algorithm:
1. **Parses** and cleans input data (e.g., `GLA`, `Year Built`, `Structure Type`).
2. **Encodes** categorical data (one-hot encoding for structure type).
3. **Scales** numerical values using `MinMaxScaler`.
4. **Calculates** Euclidean distance between the subject and all candidate properties using feature weighting.
5. **Returns** the top **k** most similar properties and saves them to `output.csv`.

---

## 📊 Features Used for Comparison

- Gross Living Area (GLA)
- Total Room Count
- Property Age (calculated from year built or age value)
- Structure Type (one-hot encoded)

---

## 🧮 Distance Weights

| Feature         | Weight |
|----------------|--------|
| GLA            | 1.0    |
| Rooms          | 1.0    |
| Age            | 1.0    |
| Structure Type | 0.2 (each encoded column) |

---

## 📥 Input Format (`training_dataset.json`)

```json
{
  "subject": {
    "id": "subject_1",
    "gla": "2000 SqFt",
    "room_total": 6,
    "year_built": "2005",
    "structure_type": "Detached",
    "effective_date": "2023-09-01"
  },
  "properties": [
    {
      "id": "p1",
      "gla": "1800 SqFt",
      "room_total": 5,
      "year_built": "2010",
      "structure_type": "Semi-Detached",
      "address": "123 Main St"
    }
  ],
  "comps": []
}
```

---

## ▶️ Usage

```bash
python property_recommender.py
```

Make sure to place your `training_dataset.json` in the same directory as the script.

---

## 💾 Output

The output is saved to `output.csv`, containing:

- Rank
- ID
- Distance from subject property
- Address
- If it was originally a comp
- Full property details (including GLA and bedroom count)

---

## 📦 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- scipy

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📘 License

MIT License
