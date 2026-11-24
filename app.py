
import streamlit as st
import cv2
import numpy as np
import sqlite3
import json
import os

from PIL import Image
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="FaceDetect AI",
    page_icon="🧠",
    layout="wide"
)

DB_NAME = "facedetect.db"

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

SIMILARITY_THRESHOLD = 0.363


# ---------------- DATABASE ----------------

def get_connection():
    return sqlite3.connect(DB_NAME)


def create_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            user_id TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            embeddings TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


create_database()


# ---------------- SESSION STATE ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "face_samples" not in st.session_state:
    st.session_state.face_samples = []

if "pending_face_photo" not in st.session_state:
    st.session_state.pending_face_photo = {}


# ---------------- AI MODELS ----------------

@st.cache_resource
def load_models():

    detector = cv2.FaceDetectorYN_create(
        DETECTOR_MODEL,
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF_create(
        RECOGNIZER_MODEL,
        ""
    )

    return detector, recognizer


# ---------------- IMAGE PROCESSING ----------------

def get_image(photo):

    image_bytes = photo.getvalue()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image


def get_faces(image, detector):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    return faces


def get_embedding(image, face, recognizer):

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    return embedding.flatten().tolist()


# ---------------- USER REGISTRATION ----------------

def register_user(full_name, user_id, password, embeddings):

    conn = get_connection()

    try:

        conn.execute("""
            INSERT INTO users
            (full_name, user_id, password, embeddings)
            VALUES (?, ?, ?, ?)
        """, (
            full_name,
            user_id,
            generate_password_hash(password),
            json.dumps(embeddings)
        ))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


# ---------------- USER LOGIN ----------------

def login_user(user_id, password):

    conn = get_connection()

    user = conn.execute("""
        SELECT * FROM users
        WHERE user_id = ?
    """, (user_id,)).fetchone()

    conn.close()

    if user and check_password_hash(user[3], password):
        return user

    return None


# ---------------- FACE RECOGNITION ----------------

def recognize_face(image):

    detector, recognizer = load_models()

    faces = get_faces(image, detector)

    results = []

    if faces is None:
        return image, results

    conn = get_connection()

    users = conn.execute("""
        SELECT full_name, user_id, embeddings
        FROM users
    """).fetchall()

    conn.close()

    for face in faces:

        embedding = np.array(
            get_embedding(image, face, recognizer),
            dtype=np.float32
        )

        best_name = "Unknown Person"
        best_score = -1

        for user in users:

            saved_embeddings = json.loads(user[2])

            for saved in saved_embeddings:

                saved_vector = np.array(
                    saved,
                    dtype=np.float32
                )

                score = recognizer.match(
                    embedding.reshape(1, -1),
                    saved_vector.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:

                    best_score = score
                    best_name = user[0]

        if best_score < SIMILARITY_THRESHOLD:
            best_name = "Unknown Person"

        x, y, w, h = face[:4].astype(int)

        color = (
            (0, 200, 0)
            if best_name != "Unknown Person"
            else (0, 0, 255)
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        cv2.putText(
            image,
            best_name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        results.append({
            "name": best_name,
            "score": round(float(best_score), 3)
        })

    return image, results


# ---------------- MAIN HEADER ----------------

st.title("🧠 FaceDetect AI")

st.subheader(
    "AI-Based Face Detection & Recognition System"
)

st.divider()


# ---------------- LOGGED-IN DASHBOARD ----------------

if st.session_state.logged_in:

    user = st.session_state.current_user

    st.sidebar.success(
        f"Welcome, {user[1]}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.current_user = None

        st.rerun()

    st.header("🏠 Dashboard")

    conn = get_connection()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    members = conn.execute("""
        SELECT full_name, user_id
        FROM users
    """).fetchall()

    conn.close()

    col1, col2 = st.columns(2)

    col1.metric(
        "Registered Users",
        total_users
    )

    col2.metric(
        "Current User",
        user[1]
    )

    st.subheader("👥 Registered Members")

    if members:

        for member in members:
            st.write(
                f"👤 {member[0]} — {member[1]}"
            )

    st.divider()

    st.header("📸 Face Recognition")

    st.write(
        "Camera se photo capture karke registered faces identify karo."
    )

    recognition_photo = st.camera_input(
        "Capture Photo for Recognition",
        key="recognition_camera"
    )

    if recognition_photo is not None:

        if st.button("🔍 Recognize Faces"):

            try:

                image = get_image(
                    recognition_photo
                )

                result_image, results = recognize_face(
                    image
                )

                result_image = cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    result_image,
                    caption="Recognition Result",
                    use_container_width=True
                )

                if results:

                    for result in results:

                        if result["name"] == "Unknown Person":

                            st.error(
                                "❌ Unknown Person"
                            )

                        else:

                            st.success(
                                f"✅ Recognized: {result['name']}"
                            )

                else:

                    st.warning(
                        "No face detected."
                    )

            except Exception as e:

                st.error(
                    f"Recognition Error: {e}"
                )


# ---------------- LOGIN AND REGISTRATION ----------------

else:

    login_tab, register_tab = st.tabs([
        "🔐 Login",
        "📝 Register"
    ])

    # ---------------- LOGIN ----------------

    with login_tab:

        st.header("Welcome Back")

        login_id = st.text_input(
            "User ID",
            key="login_id"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", type="primary"):

            user = login_user(
                login_id,
                login_password
            )

            if user:

                st.session_state.logged_in = True
                st.session_state.current_user = user

                st.success("Login Successful!")

                st.rerun()

            else:

                st.error(
                    "Invalid User ID or Password."
                )

    # ---------------- REGISTRATION ----------------

    with register_tab:

        st.header("Create New Account")

        full_name = st.text_input(
            "Full Name",
            key="reg_name"
        )

        new_uid = st.text_input(
            "Choose User ID",
            key="reg_uid"
        )

        new_password = st.text_input(
            "Create Password",
            type="password",
            key="reg_password"
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="reg_confirm"
        )

        st.write("### 📸 Face Registration")

        consent = st.checkbox(
            "I consent to capturing and storing my face data for this demo."
        )

        sample_count = len(
            st.session_state.face_samples
        )

        st.write(
            f"### Capture Face Sample {sample_count + 1}"
            if sample_count < 3
            else "### All Face Samples Collected"
        )

        if sample_count < 3:

            photo = st.camera_input(
                f"Capture Face Sample {sample_count + 1}",
                key=f"camera_{sample_count}"
            )

            # IMPORTANT FIX:
            # Save captured photo in Session State
            # so it remains available after button rerun.

            if photo is not None:

                st.session_state.pending_face_photo[
                    sample_count
                ] = photo.getvalue()

            if st.button("Add Face Sample"):

                saved_photo = (
                    st.session_state.pending_face_photo.get(
                        sample_count
                    )
                )

                if saved_photo is None:

                    st.warning(
                        "First capture your face."
                    )

                else:

                    try:

                        detector, recognizer = load_models()

                        image = get_image(
                            type(
                                "Photo",
                                (),
                                {
                                    "getvalue": lambda self: saved_photo
                                }
                            )()
                        )

                        faces = get_faces(
                            image,
                            detector
                        )

                        if faces is None or len(faces) != 1:

                            st.error(
                                "Exactly one face must be visible."
                            )

                        else:

                            embedding = get_embedding(
                                image,
                                faces[0],
                                recognizer
                            )

                            st.session_state.face_samples.append(
                                embedding
                            )

                            st.session_state.pending_face_photo.pop(
                                sample_count,
                                None
                            )

                            st.success(
                                "Face sample added successfully!"
                            )

                            st.rerun()

                    except Exception as e:

                        st.error(
                            f"Face Processing Error: {e}"
                        )

        else:

            st.success(
                "All 3 face samples collected!"
            )

        st.write(
            f"### Face samples collected: {len(st.session_state.face_samples)}/3"
        )

        if st.button("Clear Face Samples"):

            st.session_state.face_samples = []
            st.session_state.pending_face_photo = {}

            st.rerun()

        st.divider()

        if st.button(
            "Register Account",
            type="primary"
        ):

            if not full_name or not new_uid or not new_password:

                st.error(
                    "Please fill all required fields."
                )

            elif new_password != confirm:

                st.error(
                    "Passwords do not match."
                )

            elif len(new_password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            elif not consent:

                st.error(
                    "Please provide face data consent."
                )

            elif len(st.session_state.face_samples) != 3:

                st.error(
                    "Please collect all 3 face samples first."
                )

            else:

                success = register_user(
                    full_name.strip(),
                    new_uid.strip(),
                    new_password,
                    st.session_state.face_samples
                )

                if success:

                    st.success(
                        "Registration Successful! Please Login."
                    )

                    st.session_state.face_samples = []
                    st.session_state.pending_face_photo = {}

                else:

                    st.error(
                        "This User ID already exists. Choose another one."
                    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "FaceDetect AI | Python | Streamlit | OpenCV | YuNet | SFace"
)
