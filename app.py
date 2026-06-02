import os
import random
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, g, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestRegressor
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'super_secret_key_macbook_m2'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- Models ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    valuations = db.relationship('ValuationHistory', backref='user', lazy=True)

class ValuationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_name = db.Column(db.String(100))
    variant_name = db.Column(db.String(100))
    year = db.Column(db.Integer)
    kms = db.Column(db.Integer)
    condition = db.Column(db.String(20))
    predicted_price = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------- CAR DATABASE ----------
CAR_MODELS = {
    1: "Maruti Suzuki Alto K10",
    2: "Maruti Suzuki WagonR",
    3: "Maruti Suzuki Swift",
    4: "Maruti Suzuki Baleno",
    5: "Maruti Suzuki Brezza",
    6: "Maruti Suzuki Ertiga",
    7: "Maruti Suzuki Grand Vitara",
    8: "Maruti Suzuki Dzire",
    9: "Maruti Suzuki Celerio",
    10: "Maruti Suzuki S-Presso",
    11: "Maruti Suzuki Ignis",
    12: "Maruti Suzuki Eeco",
    13: "Hyundai Grand i10 Nios",
    14: "Hyundai i20",
    15: "Hyundai Venue",
    16: "Hyundai Verna",
    17: "Hyundai Creta",
    18: "Hyundai Tucson",
    19: "Hyundai Aura",
    20: "Hyundai Alcazar",
    21: "Hyundai Exter",
    22: "Tata Tiago",
    23: "Tata Punch",
    24: "Tata Nexon",
    25: "Tata Harrier",
    26: "Tata Safari",
    27: "Tata Altroz",
    28: "Tata Tigor",
    29: "Mahindra XUV300",
    30: "Mahindra Scorpio-N",
    31: "Mahindra Thar",
    32: "Mahindra XUV700",
    33: "Mahindra Bolero",
    34: "Mahindra Scorpio Classic",
    35: "Toyota Glanza",
    36: "Toyota Innova Crysta",
    37: "Toyota Fortuner",
    38: "Toyota Urban Cruiser Hyryder",
    39: "Toyota Rumion",
    40: "Kia Sonet",
    41: "Kia Seltos",
    42: "Kia Carens",
    43: "Kia EV6",
    44: "Honda Amaze",
    45: "Honda City",
    46: "Honda Elevate",
    47: "Skoda Kushaq",
    48: "Skoda Slavia",
    49: "Volkswagen Taigun",
    50: "Volkswagen Virtus",
    51: "MG Hector",
    52: "MG Astor",
    53: "MG Comet EV",
    54: "Renault Kwid",
    55: "Renault Kiger",
    56: "Renault Triber",
    57: "Nissan Magnite",
    58: "Jeep Compass",
    59: "Citroen C3",
    60: "Mercedes-Benz A-Class",
    61: "BMW 2 Series Gran Coupe",
    62: "Audi A4",
    63: "Mini Cooper S",
    64: "Volvo XC40",
    65: "Lexus ES"
}

# ---------- Build Variants ----------
ALL_VARIANTS = []
variant_counter = 1

def add_variant(model_id, variant_name, base_price, fuel, transmission):
    global variant_counter
    ALL_VARIANTS.append({
        "variant_id": variant_counter,
        "model_id": model_id,
        "model_name": CAR_MODELS[model_id],
        "variant_name": variant_name,
        "base_price": base_price,
        "fuel": fuel,
        "transmission": transmission
    })
    variant_counter += 1

def add_standard_variants(model_id, base_prices, fuels, transmissions, trims=None):
    if trims is None:
        trims = ["Base", "Mid", "Top"]
    for i, price in enumerate(base_prices):
        trim = trims[i] if i < len(trims) else f"Variant {i+1}"
        fuel = fuels[i] if i < len(fuels) else fuels[0]
        trans = transmissions[i] if i < len(transmissions) else transmissions[0]
        add_variant(model_id, f"{trim} ({fuel}, {trans})", price, fuel, trans)

# Maruti Suzuki
add_standard_variants(1,  [4.5, 5.0, 5.5], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["Base","Mid","Top"])
add_standard_variants(2,  [6.0, 6.5, 7.2], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["LXI","VXI","VXI CNG"])
add_standard_variants(3,  [7.0, 7.8, 8.5, 8.8], ["Petrol","Petrol","Petrol","CNG"], ["Manual","Manual","Manual","Manual"], ["LXI","VXI","ZXI","VXI CNG"])
add_standard_variants(4,  [7.8, 8.5, 9.3, 10.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Sigma","Delta","Zeta","Alpha AT"])
add_standard_variants(5,  [9.5, 10.8, 12.0, 12.8], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["LXI","VXI","ZXI","ZXI AT"])
add_standard_variants(6,  [10.5, 11.5, 12.5], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["LXI","VXI","VXI CNG"])
add_standard_variants(7,  [12.5, 14.0, 15.5, 16.2], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Sigma","Delta","Zeta","Zeta AT"])
add_standard_variants(8,  [7.5, 8.2, 9.0], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["LXI","VXI","ZXI"])
add_standard_variants(9,  [5.8, 6.2, 6.8], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["LXI","VXI","ZXI"])
add_standard_variants(10, [5.0, 5.5, 6.0], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["STD","LXI","VXI"])
add_standard_variants(11, [6.5, 7.0, 7.5], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["Sigma","Delta","Zeta"])
add_standard_variants(12, [5.5, 6.0, 6.5], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["STD","LXI","CNG"])

# Hyundai
add_standard_variants(13, [7.0, 7.8, 8.5], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["Era","Magna","Sportz"])
add_standard_variants(14, [8.0, 9.0, 10.2, 10.9], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Magna","Sportz","Asta","Asta AT"])
add_standard_variants(15, [9.0, 10.5, 11.8, 12.5], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","DCT"], ["E","S","SX","SX DCT"])
add_standard_variants(16, [12.5, 13.5, 15.0, 16.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["E","S","SX","SX AT"])
add_standard_variants(17, [12.5, 14.2, 16.0, 17.0, 17.5], ["Petrol","Petrol","Petrol","Petrol","Diesel"], ["Manual","Manual","Manual","Automatic","Automatic"], ["E","S","SX","SX AT","Diesel SX"])
add_standard_variants(18, [28.0, 30.5, 33.0], ["Petrol","Petrol","Diesel"], ["Automatic","Automatic","Automatic"], ["Platinum","Signature","Diesel Signature"])
add_standard_variants(19, [7.5, 8.0, 8.5], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["E","S","SX"])
add_standard_variants(20, [16.0, 18.0, 20.0], ["Petrol","Petrol","Diesel"], ["Manual","Automatic","Automatic"], ["Prestige","Platinum","Diesel Platinum"])
add_standard_variants(21, [7.0, 7.8, 8.5], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["EX","S","SX"])

# Tata Motors
add_standard_variants(22, [6.0, 6.5, 7.0], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["XE","XT","XZ"])
add_standard_variants(23, [7.0, 7.8, 8.5, 9.0], ["Petrol","Petrol","Petrol","CNG"], ["Manual","Manual","Manual","Manual"], ["Pure","Adventure","Accomplished","CNG"])
add_standard_variants(24, [9.5, 10.8, 12.5, 13.2, 14.0], ["Petrol","Petrol","Petrol","Petrol","Diesel"], ["Manual","Manual","Manual","Automatic","Manual"], ["Smart","Pure","Creative","Creative AT","Diesel Creative"])
add_standard_variants(25, [18.0, 20.0, 22.5], ["Diesel","Diesel","Diesel"], ["Manual","Manual","Automatic"], ["Pure","Adventure","Dark"])
add_standard_variants(26, [20.0, 22.0, 24.5], ["Diesel","Diesel","Diesel"], ["Manual","Automatic","Automatic"], ["XMA","XTA+","XZA+"])
add_standard_variants(27, [7.5, 8.0, 8.8], ["Petrol","Petrol","Diesel"], ["Manual","Manual","Manual"], ["XE","XT","XZ"])
add_standard_variants(28, [6.5, 7.0, 7.5], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["XE","XT","XZ"])

# Mahindra
add_standard_variants(29, [9.5, 10.5, 11.5], ["Petrol","Petrol","Diesel"], ["Manual","Manual","Manual"], ["W4","W6","W8"])
add_standard_variants(30, [15.0, 17.5, 20.0], ["Diesel","Diesel","Diesel"], ["Manual","Manual","Automatic"], ["Z2","Z4","Z8"])
add_standard_variants(31, [14.0, 16.0, 17.5], ["Diesel","Diesel","Diesel"], ["Manual","Manual","Automatic"], ["AX","LX","LX AT"])
add_standard_variants(32, [16.0, 18.5, 21.0, 23.5, 24.5], ["Petrol","Petrol","Petrol","Petrol","Diesel"], ["Manual","Manual","Manual","Automatic","Automatic"], ["MX","AX3","AX5","AX7 AT","Diesel AX7"])
add_standard_variants(33, [9.0, 9.5, 10.0], ["Diesel","Diesel","Diesel"], ["Manual","Manual","Manual"], ["B4","B6","B6 Plus"])
add_standard_variants(34, [13.0, 14.5, 16.0], ["Diesel","Diesel","Diesel"], ["Manual","Manual","Manual"], ["S","S11","S11+"])

# Toyota
add_standard_variants(35, [8.0, 8.5, 9.3], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Automatic"], ["E","S","G"])
add_standard_variants(36, [22.0, 24.5, 27.0], ["Diesel","Diesel","Diesel"], ["Manual","Manual","Automatic"], ["G","V","Z"])
add_standard_variants(37, [38.0, 41.5, 46.0], ["Diesel","Diesel","Diesel"], ["Manual","Automatic","Automatic"], ["4x2 MT","4x2 AT","4x4 AT"])
add_standard_variants(38, [12.5, 13.5, 14.5], ["Petrol","Petrol","Hybrid"], ["Manual","Manual","Automatic"], ["S","G","V"])
add_standard_variants(39, [10.0, 10.5, 11.2], ["Petrol","Petrol","CNG"], ["Manual","Manual","Manual"], ["S","G","V"])

# Kia
add_standard_variants(40, [9.0, 10.5, 12.0, 13.5], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","DCT"], ["HTE","HTK","HTX","GTX"])
add_standard_variants(41, [12.0, 13.5, 15.5, 17.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","DCT"], ["HTE","HTK","HTX","GTX"])
add_standard_variants(42, [11.0, 12.5, 14.0], ["Petrol","Petrol","Diesel"], ["Manual","Manual","Automatic"], ["Premium","Prestige","Luxury"])
add_standard_variants(43, [60.0, 65.0, 70.0], ["Electric","Electric","Electric"], ["Automatic","Automatic","Automatic"], ["GT Line","GT Line AWD","First Edition"])

# Honda
add_standard_variants(44, [7.5, 8.2, 9.0], ["Petrol","Petrol","Petrol"], ["Manual","Manual","CVT"], ["E","S","VX"])
add_standard_variants(45, [13.0, 14.5, 16.0, 17.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","CVT"], ["SV","V","ZX","ZX CVT"])
add_standard_variants(46, [12.5, 13.5, 15.0], ["Petrol","Petrol","Petrol"], ["Manual","Manual","CVT"], ["SV","V","ZX"])

# Skoda / VW
add_standard_variants(47, [12.5, 14.5, 16.5, 17.5], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Active","Ambition","Style","Style AT"])
add_standard_variants(48, [12.0, 13.5, 15.0, 16.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Active","Ambition","Style","Style AT"])
add_standard_variants(49, [12.5, 14.5, 16.5, 17.5], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Comfortline","Highline","Topline","Topline AT"])
add_standard_variants(50, [12.0, 13.5, 15.0, 16.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","Automatic"], ["Comfortline","Highline","Topline","Topline AT"])

# MG
add_standard_variants(51, [16.0, 18.0, 20.0, 22.0], ["Petrol","Petrol","Petrol","Diesel"], ["Manual","Manual","Automatic","Manual"], ["Style","Smart","Sharp","Diesel Sharp"])
add_standard_variants(52, [11.0, 12.5, 14.0], ["Petrol","Petrol","Petrol"], ["Manual","Automatic","Automatic"], ["Style","Smart","Sharp"])
add_standard_variants(53, [8.5, 9.5, 10.5], ["Electric","Electric","Electric"], ["Automatic","Automatic","Automatic"], ["Pace","Play","Plush"])

# Renault / Nissan
add_standard_variants(54, [4.5, 5.0, 5.5], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["RXL","RXT","Climber"])
add_standard_variants(55, [7.0, 8.0, 9.2, 10.0], ["Petrol","Petrol","Petrol","Petrol"], ["Manual","Manual","Manual","CVT"], ["RXE","RXL","RXT","RXT CVT"])
add_standard_variants(56, [7.0, 7.5, 8.2], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["RXE","RXL","RXT"])
add_standard_variants(57, [7.0, 8.0, 9.0], ["Petrol","Petrol","Petrol"], ["Manual","Manual","CVT"], ["XE","XL","XV"])

# Jeep / Citroen
add_standard_variants(58, [22.0, 24.0, 26.5], ["Diesel","Diesel","Diesel"], ["Manual","Automatic","Automatic"], ["Sport","Longitude","Limited"])
add_standard_variants(59, [7.0, 7.5, 8.0], ["Petrol","Petrol","Petrol"], ["Manual","Manual","Manual"], ["Live","Feel","Shine"])

# Luxury
add_standard_variants(60, [35.0, 38.0, 42.0], ["Petrol","Diesel","Petrol"], ["Automatic","Automatic","Automatic"], ["A200","A200d","A35 AMG"])
add_standard_variants(61, [36.0, 39.0, 42.0], ["Petrol","Diesel","Petrol"], ["Automatic","Automatic","Automatic"], ["220i","220d","M235i"])
add_standard_variants(62, [45.0, 48.0, 52.0], ["Petrol","Diesel","Petrol"], ["Automatic","Automatic","Automatic"], ["Premium","Technology","S Line"])
add_standard_variants(63, [38.0, 40.0], ["Petrol","Petrol"], ["Automatic","Automatic"], ["Cooper S","JCW"])
add_standard_variants(64, [38.0, 42.0, 45.0], ["Petrol","Electric","Electric"], ["Automatic","Automatic","Automatic"], ["Momentum","Recharge","Ultimate"])
add_standard_variants(65, [58.0, 62.0, 66.0], ["Petrol","Hybrid","Petrol"], ["Automatic","Automatic","Automatic"], ["ES 300h","ES 300h Luxury","ES 350"])

# ---------- Helper Functions ----------
def encode_fuel(fuel):
    mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "Hybrid": 4}
    return mapping.get(fuel, 0)

def encode_transmission(trans):
    return 0 if "Manual" in trans else 1

def encode_condition(cond):
    mapping = {"Excellent": 1.0, "Good": 0.92, "Fair": 0.82}
    return mapping.get(cond, 0.92)

# ---------- ML Training ----------
brand_depreciation = {
    "Maruti": 0.92, "Hyundai": 0.91, "Tata": 0.90, "Mahindra": 0.89,
    "Toyota": 0.93, "Kia": 0.91, "Honda": 0.92, "Skoda": 0.88,
    "Volkswagen": 0.88, "MG": 0.87, "Renault": 0.85, "Nissan": 0.86,
    "Jeep": 0.87, "Citroen": 0.84, "Mercedes-Benz": 0.82, "BMW": 0.81,
    "Audi": 0.80, "Mini": 0.83, "Volvo": 0.82, "Lexus": 0.84
}

CURRENT_YEAR = 2026
X_train_list = []
y_train_list = []

for var in ALL_VARIANTS:
    base = var["base_price"]
    brand = var["model_name"].split()[0]
    if brand == "Maruti":
        brand = "Maruti"
    fuel_type = var["fuel"]
    transmission = var["transmission"]
    variant_id = var["variant_id"]

    if base >= 40:
        first_year_factor = 0.72
    elif base >= 25:
        first_year_factor = 0.75
    elif base >= 15:
        first_year_factor = 0.78
    else:
        first_year_factor = 0.80

    yearly_factor = brand_depreciation.get(brand, 0.90)

    for age in range(1, 16):
        year = CURRENT_YEAR - age
        avg_kms = age * 10000
        kms = avg_kms + random.randint(-2000, 5000)
        kms = max(kms, 1000)

        if age == 1:
            depreciation = first_year_factor
        else:
            depreciation = first_year_factor * (yearly_factor ** (age - 1))

        km_penalty = max(0, (kms - 50000) / 15000) * 0.01
        km_premium = max(0, (40000 - kms) / 40000) * 0.05

        fuel_bonus = 0.02 if fuel_type == "Diesel" else (-0.02 if fuel_type in ["Electric", "CNG"] else 0)
        trans_bonus = 0.03 if transmission == "Automatic" else 0

        for cond_name, cond_factor in [("Excellent",1.0), ("Good",0.92), ("Fair",0.82)]:
            calculated_price = base * depreciation * (1 - km_penalty + km_premium) * (1 + fuel_bonus + trans_bonus) * cond_factor
            final_price = max(base * 0.05, min(calculated_price, base * 1.05))
            X_train_list.append([variant_id, year, kms, encode_fuel(fuel_type), encode_transmission(transmission), encode_condition(cond_name)])
            y_train_list.append(final_price)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ---------- Authentication Helpers ----------
def get_current_user():
    """Return logged-in user or redirect to login."""
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        if user:
            return user
        # Invalid session – clear it
        session.clear()
    return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return redirect(url_for('login'))
        g.user = user  # store for easy access in routes
        return f(*args, **kwargs)
    return decorated_function

# ---------- Flask Routes ----------
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "<script>alert('Username already exists!'); window.location.href='/register';</script>"
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['username'] = username
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            return "<script>alert('Invalid Credentials!'); window.location.href='/login';</script>"
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = g.user
    return render_template('dashboard.html',
                           username=user.username,
                           car_models=CAR_MODELS,
                           all_variants=ALL_VARIANTS,
                           notification_count=ValuationHistory.query.filter_by(user_id=user.id).count())

@app.route('/get_variants/<int:model_id>')
@login_required
def get_variants(model_id):
    variants = [v for v in ALL_VARIANTS if v['model_id'] == model_id]
    return jsonify(variants)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user = g.user
    try:
        variant_id = int(request.form['variant_id'])
        year = int(request.form['year'])
        kms = int(request.form['kms'])
        condition = request.form.get('condition', 'Good')

        variant = next((v for v in ALL_VARIANTS if v['variant_id'] == variant_id), None)
        if not variant:
            return "Invalid variant selected."

        features = np.array([[
            variant_id, year, kms,
            encode_fuel(variant['fuel']),
            encode_transmission(variant['transmission']),
            encode_condition(condition)
        ]])
        prediction = model.predict(features)[0]
        predicted_price = round(prediction, 2)

        history_entry = ValuationHistory(
            user_id=user.id,
            model_name=variant['model_name'],
            variant_name=variant['variant_name'],
            year=year,
            kms=kms,
            condition=condition,
            predicted_price=predicted_price
        )
        db.session.add(history_entry)
        db.session.commit()

        return render_template('dashboard.html',
                               username=user.username,
                               car_models=CAR_MODELS,
                               all_variants=ALL_VARIANTS,
                               selected_model=variant['model_id'],
                               selected_variant=variant_id,
                               selected_condition=condition,
                               prediction_text=f"₹ {predicted_price} Lakhs – {variant['model_name']} {variant['variant_name']} ({year}, {kms} km, {condition} condition)",
                               notification_count=ValuationHistory.query.filter_by(user_id=user.id).count())
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/history')
@login_required
def history():
    user = g.user
    valuations = ValuationHistory.query.filter_by(user_id=user.id).order_by(ValuationHistory.timestamp.desc()).all()
    return render_template('history.html',
                           username=user.username,
                           valuations=valuations,
                           notification_count=len(valuations))

@app.route('/analytics')
@login_required
def analytics():
    user = g.user
    valuations = ValuationHistory.query.filter_by(user_id=user.id).all()
    brands = {}
    for v in valuations:
        brand = v.model_name.split()[0]
        if brand not in brands:
            brands[brand] = []
        brands[brand].append(v.predicted_price)
    brand_avg = {brand: sum(prices)/len(prices) for brand, prices in brands.items()}

    depreciation_labels = list(range(1, 16))
    depreciation_values = []
    if ALL_VARIANTS:
        sample_variant = ALL_VARIANTS[0]  # just an example
        for age in depreciation_labels:
            features = np.array([[sample_variant['variant_id'], CURRENT_YEAR - age, age*10000,
                                  encode_fuel(sample_variant['fuel']), encode_transmission(sample_variant['transmission']), 1.0]])
            val = model.predict(features)[0]
            depreciation_values.append(round(val, 2))

    return render_template('analytics.html',
                           username=user.username,
                           brand_avg=brand_avg,
                           depreciation_labels=depreciation_labels,
                           depreciation_values=depreciation_values,
                           notification_count=len(valuations))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user = g.user
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        if user and check_password_hash(user.password_hash, current_password):
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            msg = "Password updated successfully."
        else:
            msg = "Current password is incorrect."
        return render_template('settings.html', username=user.username, msg=msg,
                               notification_count=ValuationHistory.query.filter_by(user_id=user.id).count())
    return render_template('settings.html', username=user.username, msg=None,
                           notification_count=ValuationHistory.query.filter_by(user_id=user.id).count())

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)