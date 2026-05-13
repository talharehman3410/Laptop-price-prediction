# 💻 LaptopLens — AI Price Intelligence

A sleek Flask web application for predicting laptop prices using 4 ML models from your Jupyter notebook.

## 📁 Project Structure

```
laptop_price_app/
├── app.py              # Flask backend + routes
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Elegant frontend UI
├── static/             # (optional) extra CSS/JS assets
├── model.pkl           # ← Place your trained model here
└── preprocessed.csv    # ← Place your dataset here
```

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

## 🔗 Connecting Your Real ML Model

In `app.py`, replace the `mock_predict()` function with your real model:

```python
import pickle

# Load your trained model (saved from notebook)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your label encoders / preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

def real_predict(features):
    # Preprocess features the same way as in your notebook
    df = pd.DataFrame([features])
    X = preprocessor.transform(df)
    price = model.predict(X)[0]
    return round(float(price), 2)
```

## 🎨 Features

- ⚡ Real-time price prediction with animated counter
- 📊 Model accuracy comparison bar chart
- 🔄 Switch between 4 ML models (Gradient Boosting, CatBoost, XGBoost, SVM)
- 📝 Prediction history tracker
- 💰 Price confidence range (±8%)
- 🌙 Dark-mode luxury UI with micro-animations

## 📊 Models & Accuracy (from your notebook)

| Model              | R² Accuracy |
|--------------------|-------------|
| Gradient Boosting  | 86.40%      |
| CatBoost           | 85.25%      |
| XGBoost            | 84.50%      |
| SVM (SVC)          | 72.10%      |

## 🌐 API Endpoint

```
POST /predict
Content-Type: application/json

{
  "company": "Apple",
  "type_name": "Ultrabook",
  "inches": "13.3",
  "ram": "16GB",
  "gpu": "Intel Iris Plus Graphics 640",
  "opsys": "macOS",
  "weight": "1.37kg",
  "model": "gradient_boosting"
}

Response:
{
  "success": true,
  "price": 1542.75,
  "price_low": 1419.33,
  "price_high": 1666.17,
  "model": "Gradient Boosting",
  "accuracy": 86.40,
  "currency": "EUR"
}
```
