from flask_login import login_required, current_user, logout_user
from werkzeug.security import check_password_hash
from flask_login import login_user
from flask import redirect, url_for, flash
from werkzeug.security import generate_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)

from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    @login_manager.user_loader
    def load_user(user_id):
       return User.query.get(int(user_id))
login_manager.login_view = "login"

import os
import requests
from tensorflow import keras

MODEL_PATH = "dr_final_model.keras"
MODEL_URL = "https://drive.google.com/file/d/1yyCdbNw6bOcS2guxEGJULNlHcagiQLv9/view?usp=sharing"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")

# Load model
model = keras.models.load_model(MODEL_PATH)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(229,229))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        confidence = float(np.max(prediction)) * 100

        result = class_names[predicted_class]

        return render_template(
    "prediction.html",
    result=result,
    confidence=round(confidence, 2),
    filename=file.filename
)

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)