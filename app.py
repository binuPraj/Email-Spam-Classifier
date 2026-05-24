from flask import Flask, render_template, request
import pickle
from pathlib import Path

app = Flask(__name__)

# Ensure model exists; if not, attempt to train a fresh one from emails.csv
MODEL_DIR = Path(__file__).resolve().parent / "model"
CV_PATH = MODEL_DIR / "cv.pkl"
CLF_PATH = MODEL_DIR / "clf.pkl"

if not (CV_PATH.exists() and CLF_PATH.exists()):
    # to train a model automatically
    try:
        from train_model import main as train_main

        print("Model files not found — training model now...")
        train_main()
    except Exception as e:
        print("Failed to train model automatically:", e)
        raise

with open(CV_PATH, "rb") as f:
    cv = pickle.load(f)
with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)


@app.route('/')
def index():
    return render_template("spam_checker.html", email_text="")


@app.route('/predict', methods=['post'])
def predict():
    # Allow a 'clear' flow when user wants to clear the textarea after checking
    if request.args.get("clear") == "1":
        email = request.form.get("email") or ""
        clear_after = True
    else:
        email = request.form.get("email") or ""
        clear_after = False

    X = cv.transform([email])
    raw_prediction = int(clf.predict(X)[0])
    is_spam = True if raw_prediction == 1 else False

    message = "The email appears to be SPAM." if is_spam else "The email appears to be NOT SPAM."
    return render_template("spam_checker.html", message=message, is_spam=is_spam, email_text=("" if clear_after else email))


if __name__ == "__main__":
    app.run(debug=True)
