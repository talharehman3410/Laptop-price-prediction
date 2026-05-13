from flask import Flask, render_template, request, jsonify
import numpy as np
import re

app = Flask(__name__)

# ─── Mock prediction logic (replace with real model loading) ───────────────────
# In production: load your pickle model and preprocessed pipeline
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def extract_ram_gb(ram_str):
    """Extract numeric RAM value"""
    match = re.search(r'(\d+)', str(ram_str))
    return int(match.group(1)) if match else 8

def extract_weight_kg(weight_str):
    """Extract numeric weight value"""
    match = re.search(r'([\d.]+)', str(weight_str))
    return float(match.group(1)) if match else 2.0

def mock_predict(features):
    """
    Mock price predictor — replace with your real model.
    Simulates realistic laptop pricing based on specs.
    """
    company = features['company']
    type_name = features['type_name']
    inches = float(features['inches'])
    ram = extract_ram_gb(features['ram'])
    weight = extract_weight_kg(features['weight'])
    opsys = features['opsys']
    gpu_brand = features['gpu'].split()[0].lower() if features['gpu'] else 'intel'

    base = 400

    # Company premium
    company_multiplier = {
        'Apple': 2.2, 'Microsoft': 1.9, 'Razer': 2.0, 'LG': 1.5,
        'Samsung': 1.6, 'Dell': 1.3, 'HP': 1.1, 'Lenovo': 1.1,
        'Asus': 1.0, 'Acer': 0.9, 'MSI': 1.5, 'Toshiba': 0.95,
        'Huawei': 1.3, 'Xiaomi': 1.0
    }
    base *= company_multiplier.get(company, 1.0)

    # Type premium
    type_multiplier = {
        'Ultrabook': 1.4, 'Gaming': 1.6, 'Workstation': 2.0,
        'Notebook': 1.0, '2 in 1 Convertible': 1.3, 'Netbook': 0.6
    }
    base *= type_multiplier.get(type_name, 1.0)

    # RAM scaling
    base += (ram - 8) * 35

    # Screen size factor
    if inches >= 17:
        base += 80
    elif inches <= 12:
        base -= 50

    # GPU factor
    if 'nvidia' in gpu_brand or 'rtx' in gpu_brand or 'gtx' in gpu_brand:
        base += 300
    elif 'amd' in gpu_brand:
        base += 150

    # OS factor
    if 'macOS' in opsys:
        base += 200
    elif 'Windows 10 Pro' in opsys:
        base += 100

    # Weight (lighter = more premium)
    if weight < 1.5:
        base += 150
    elif weight > 2.5:
        base -= 50

    # Add some realistic variance
    noise = np.random.uniform(-40, 40)
    price = max(200, base + noise)

    return round(price, 2)

# ─── Data for dropdowns ────────────────────────────────────────────────────────
COMPANIES = ['Apple', 'Asus', 'Acer', 'Dell', 'HP', 'Huawei', 'Lenovo',
             'LG', 'Microsoft', 'MSI', 'Razer', 'Samsung', 'Toshiba', 'Xiaomi']

TYPE_NAMES = ['Ultrabook', 'Notebook', 'Gaming', 'Workstation',
              '2 in 1 Convertible', 'Netbook']

SCREEN_SIZES = ['11.6', '12.0', '12.5', '13.3', '14.0', '15.0', '15.6', '17.3']

RAM_OPTIONS = ['2GB', '4GB', '6GB', '8GB', '12GB', '16GB', '32GB', '64GB']

GPU_OPTIONS = [
    'Intel HD Graphics', 'Intel HD Graphics 620', 'Intel Iris Plus Graphics 640',
    'Intel Iris Plus Graphics 650', 'Intel UHD Graphics 620',
    'Nvidia GeForce 940MX', 'Nvidia GeForce GTX 1050', 'Nvidia GeForce GTX 1050 Ti',
    'Nvidia GeForce GTX 1060', 'Nvidia GeForce GTX 1070', 'Nvidia GeForce GTX 1080',
    'AMD Radeon RX 580', 'AMD Radeon Pro 555', 'AMD Radeon R5 M330',
    'Nvidia Quadro M2200', 'Nvidia Quadro P5000'
]

OPSYS_OPTIONS = [
    'Windows 10', 'Windows 10 Pro', 'Windows 10 S', 'Windows 7',
    'macOS', 'Linux', 'Chrome OS', 'No OS', 'Android'
]

WEIGHT_OPTIONS = ['0.9kg', '1.0kg', '1.2kg', '1.3kg', '1.37kg', '1.5kg',
                  '1.6kg', '1.7kg', '1.8kg', '1.86kg', '1.9kg', '2.0kg',
                  '2.1kg', '2.19kg', '2.2kg', '2.3kg', '2.5kg', '3.0kg', '3.5kg']

MODEL_ACCURACIES = {
    'gradient_boosting': 86.40,
    'catboost': 85.25,
    'xgboost': 84.50,
    'svm': 72.10
}

@app.route('/')
def index():
    return render_template('index.html',
        companies=COMPANIES,
        type_names=TYPE_NAMES,
        screen_sizes=SCREEN_SIZES,
        ram_options=RAM_OPTIONS,
        gpu_options=GPU_OPTIONS,
        opsys_options=OPSYS_OPTIONS,
        weight_options=WEIGHT_OPTIONS,
        model_accuracies=MODEL_ACCURACIES
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = {
            'company': data.get('company', 'HP'),
            'type_name': data.get('type_name', 'Notebook'),
            'inches': data.get('inches', '15.6'),
            'ram': data.get('ram', '8GB'),
            'gpu': data.get('gpu', 'Intel HD Graphics'),
            'opsys': data.get('opsys', 'Windows 10'),
            'weight': data.get('weight', '2.0kg'),
            'model': data.get('model', 'gradient_boosting')
        }

        price = mock_predict(features)
        model_name = features['model'].replace('_', ' ').title()
        accuracy = MODEL_ACCURACIES.get(features['model'], 85.0)

        # Price range (±8%)
        low = round(price * 0.92, 2)
        high = round(price * 1.08, 2)

        return jsonify({
            'success': True,
            'price': price,
            'price_low': low,
            'price_high': high,
            'model': model_name,
            'accuracy': accuracy,
            'currency': 'EUR'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
