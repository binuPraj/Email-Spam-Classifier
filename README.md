# Email Spam Classifier

Minimal Flask app that classifies email text as spam or not-spam.
Features
- Clean, user-friendly UI (`templates/spam_checker.html`) with example dropdowns and the result shown beneath the pasted email text.
- `train_model.py` to build model artifacts from `emails.csv` (recreate pickles for your environment).
- `app.py` will auto-train on first run if model files are missing.

Prerequisites
- Python 3.10 or newer (3.12 tested)
- Git (optional)

Quick start (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000
