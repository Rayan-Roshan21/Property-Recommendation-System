# 🏡 Property Recommendation System

This project is a Python-based property recommendation tool that uses feature similarity to identify the top **k** comparable properties to a given subject property. It reads property data from a JSON file and outputs the most similar properties using a distance-based algorithm.

---

## 📂 Project Structure

- `training_dataset.json`: Input file containing subject property, candidate properties, and known comps.
- `output.csv`: Output file with the top k most similar properties.
- `software_recommendation_notebook.ipynb`: Jupyter notebook containing the full workflow from preprocessing to distance computation and output.

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

### Required Keys

```json
{
  "subject": {...},
  "properties": [...],
  "comps": [...]
}
```

- `subject`: The target property you're trying to compare against.
- `properties`: A list of all potential comparable properties in the area.
- `comps`: A list of previously chosen comparable properties. These will also be evaluated alongside `properties` but flagged in the results.

---

## ▶️ Usage

Open the notebook using Jupyter:

```bash
jupyter notebook Software_recommendation_notebook.ipynb
```

Run each cell sequentially to perform preprocessing, compute similarity, and view/save results.

You can modify the number of recommendations by changing the value of `k` in the notebook.

---

## 💾 Output

The output is saved to `output.csv`, containing:

- Rank
- ID
- Distance from subject property
- Address
- Is Comp (True/False)
- Full property details (including GLA and bedroom count)

---

## 📦 Dependencies

Install required Python packages with:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install scipy
```

Python 3.7+ is recommended.

---

## 📘 License

MIT License
