# FaceDetect AI — Python + AI/ML Internship Prototype

## Features
- Public account registration (unlimited users)
- User ID + password login
- Password hashing
- 3 face samples per user
- Face detection and recognition against registered users
- Unknown Person result
- Streamlit interface, with Python application logic

## Setup (Mac / VS Code)
Use Python 3.10 or 3.11 for the smoothest compatibility with `face-recognition` / `dlib`.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the native build tools if needed:
```bash
xcode-select --install
brew install cmake
```

Then install packages:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

The browser opens automatically. Use Register to create accounts, then log in.

## Registration workflow
1. Enter full name, unique User ID and password.
2. Capture one clear face photo and click **Add this face photo**.
3. Repeat until 3 samples have been added.
4. Click **Register account**.
5. Log in and use **Recognize Face**.

## Important limitations
- This first-hour MVP uses a camera photo capture per scan, not continuous video streaming.
- It expects exactly one face in each registration or recognition photo.
- The face-recognition package depends on `dlib`; installation can be difficult on some Python/macOS combinations.
- The local SQLite database (`facedetect.db`) contains biometric face embeddings. Keep it private and get each person's informed consent before capturing or storing their face data.
- This is a learning prototype, not a production identity-verification or security system.
