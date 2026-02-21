from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ================= CONFIG =================
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = "static/uploads"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================= USER MODEL =================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= LOAD MODEL =================
MODEL_PATH = "dr_final_model.keras"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Xception class names
class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# ================= ROUTES =================

# 🔹 Public Landing Page (NEW HOME PAGE)
@app.route("/")
def landing():
    return render_template("home.html")


# 🔹 Dashboard (after login)
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html")


# 🔹 Register
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


# 🔹 Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password")

    return render_template("login.html")


# 🔹 Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("landing"))


# 🔹 Prediction Route (Xception Fixed)
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ Use 229x229 (your model input size)
        img = image.load_img(filepath, target_size=(229, 229))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # If you trained with /255.0 normalization
        img_array = img_array / 255.0

        # If you used Xception preprocess_input during training,
        # then use this instead of /255.0:
        # from tensorflow.keras.applications.xception import preprocess_input
        # img_array = preprocess_input(img_array)

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

    return redirect(url_for("dashboard"))


# ================= INIT DB =================
with app.app_context():
    db.create_all()

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)