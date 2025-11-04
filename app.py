import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# ------------- Config -------------
nimgs = 10
BACKGROUND_PATH = "background.png"
HAAR_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "static/face_recognition_model.pkl"
DATETIME_FMT = "%H:%M:%S"

# Distance threshold for KNN to decide "unknown".
# You will likely need to tune this for your data. Lower => stricter.
UNKNOWN_DISTANCE_THRESHOLD = 3000.0

# ------------- Setup -------------
# Load or create background
if os.path.isfile(BACKGROUND_PATH):
    imgBackground = cv2.imread(BACKGROUND_PATH)
else:
    # Create a blank background (800x1280) if no file present
    imgBackground = np.zeros((800, 1280, 3), dtype=np.uint8)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Face detector
if not os.path.isfile(HAAR_PATH):
    raise FileNotFoundError(f"Haarcascade file not found at {HAAR_PATH}")
face_detector = cv2.CascadeClassifier(HAAR_PATH)

# Ensure folders
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Ensure attendance file exists with header
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# ------------- Helpers -------------
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception:
        return []

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        user_folder = os.path.join('static', 'faces', user)
        for imgname in os.listdir(user_folder):
            imgpath = os.path.join(user_folder, imgname)
            img = cv2.imread(imgpath)
            if img is None:
                print(f"Warning: could not read {imgpath}")
                continue
            # use color as before (consistent with prediction)
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel().astype(np.float32))
            labels.append(user)
    if len(faces) == 0:
        raise ValueError("No faces found to train.")
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, MODEL_PATH)
    print("Model trained and saved.")

def load_model_safe():
    if os.path.isfile(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def read_attendance_df():
    # read CSV safely, if only header exists return empty DF with columns
    try:
        df = pd.read_csv(attendance_file)
    except Exception:
        df = pd.DataFrame(columns=['Name','Roll','Time'])
    return df

def extract_attendance():
    df = read_attendance_df()
    if 'Name' not in df.columns:
        df = pd.DataFrame(columns=['Name','Roll','Time'])
    names = df['Name'] if not df.empty else []
    rolls = df['Roll'] if not df.empty else []
    times = df['Time'] if not df.empty else []
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
    except Exception:
        print("add_attendance: name format invalid:", name)
        return
    current_time = datetime.now().strftime(DATETIME_FMT)

    df = read_attendance_df()
    # Guard against empty / no 'Roll' column
    existing_rolls = list(df['Roll'].astype(int)) if (not df.empty and 'Roll' in df.columns) else []
    try:
        if int(userid) not in existing_rolls:
            with open(attendance_file, 'a') as f:
                f.write(f'{username},{userid},{current_time}\n')
    except Exception:
        # fallback - append anyway (prevents crash)
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    for i in userlist:
        if '_' in i:
            name, roll = i.split('_', 1)
        else:
            name, roll = i, "0"
        names.append(name)
        rolls.append(roll)
    l = len(userlist)
    return userlist, names, rolls, l

# ------------- Flask Routes -------------
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    model = load_model_safe()
    if model is None:
        return render_template('home.html', names=names, rolls=rolls, times=times,
                               l=l, totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model. Please add a new face to continue.')

    # open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', names=names, rolls=rolls, times=times,
                               l=l, totalreg=totalreg(), datetoday2=datetoday2,
                               mess='Cannot open camera. Check camera index and permissions.')

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: no frame from camera")
                break

            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                try:
                    face_resized = cv2.resize(face_roi, (50, 50)).ravel().astype(np.float32).reshape(1, -1)
                except Exception:
                    face_resized = None

                if face_resized is not None:
                    # use kneighbors to get distance for confidence
                    distances, indices = model.kneighbors(face_resized, n_neighbors=5, return_distance=True)
                    # distances is shape (1, n_neighbors)
                    mean_dist = float(np.mean(distances))
                    predicted_name = model.predict(face_resized)[0]  # label predicted

                    # Decide known / unknown by threshold
                    if mean_dist <= UNKNOWN_DISTANCE_THRESHOLD:
                        # recognized
                        add_attendance(predicted_name)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
                        cv2.putText(frame, f'Face Match OK - {predicted_name}', (x, y-12),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # unknown / low confidence
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 0, 255), -1)
                        cv2.putText(frame, "Unknown Face!", (x, y-12),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "Please Register Your Face", (50, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # debug print
                        print(f"Unrecognized - mean_dist={mean_dist:.1f}, threshold={UNKNOWN_DISTANCE_THRESHOLD}")

            # place into background safely (check sizes)
            bh, bw = imgBackground.shape[:2]
            fh, fw = frame.shape[:2]
            # if frame is larger than background, resize the frame to fit
            max_h = min(480, fh)
            max_w = min(640, fw)
            frame_small = cv2.resize(frame, (max_w, max_h))
            # choose position similar to previous script
            y0, x0 = 162, 55
            if y0 + max_h <= bh and x0 + max_w <= bw:
                img_display = imgBackground.copy()
                img_display[y0:y0 + max_h, x0:x0 + max_w] = frame_small
            else:
                img_display = frame.copy()

            cv2.imshow('Face Recognition By Abdul Hameed', img_display)
            if cv2.waitKey(1) == 27:  # ESC to exit
                break
    except Exception:
        print("Error in recognition loop:")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    # Basic form validation
    try:
        newusername = request.form['newusername'].strip()
        newuserid = request.form['newuserid'].strip()
    except Exception:
        return "Please provide newusername and newuserid in the form."

    if not newusername or not newuserid:
        return "Username or userid empty."

    userimagefolder = os.path.join('static', 'faces', f'{newusername}_{newuserid}')
    os.makedirs(userimagefolder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Cannot open camera."

    i = 0
    j = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                # Save every 5 frames to avoid near-duplicates (like original)
                if j % 5 == 0 and i < nimgs:
                    face_img = frame[y:y+h, x:x+w]
                    try:
                        face_small = cv2.resize(face_img, (200, 200))
                        fname = os.path.join(userimagefolder, f'{newusername}_{i}.jpg')
                        cv2.imwrite(fname, face_small)
                        i += 1
                    except Exception as e:
                        print("Error saving image:", e)
                j += 1

            cv2.imshow('New User Form By Abdul Hameed', frame)
            if cv2.waitKey(1) == 27:
                break
            if i >= nimgs:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print('Training Model...')
    try:
        train_model()
    except Exception as e:
        print("Training failed:", e)
        traceback.print_exc()
        return "Training failed. See server logs."

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)
