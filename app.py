from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for, flash
import threading, time, os
import cv2, numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'anishistheboss'
# ——— simple SQLite user store ———
import sqlite3
DB = 'users.db'
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users(
              id INTEGER PRIMARY KEY,
              username TEXT UNIQUE,
              password_hash TEXT
            )
        ''')
init_db()
from werkzeug.security import generate_password_hash, check_password_hash
app.config.update(
  UPLOAD_FOLDER = 'uploads',
  EMBEDDINGS_FOLDER = 'embeddings',
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EMBEDDINGS_FOLDER'], exist_ok=True)

# ——— load YOLOv8 models ———
weapon_model   = YOLO('models/weapon_best.pt')
accident_model = YOLO('models/accident_best.pt')

# ——— face detection setup ———
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# ——— camera toggle flag ———
camera_on = True

def load_offender_db():
    db = {}
    for fn in os.listdir(app.config['EMBEDDINGS_FOLDER']):
        if fn.endswith('_embedding.npy'):
            name = fn.split('_')[0]
            db[name] = np.load(os.path.join(app.config['EMBEDDINGS_FOLDER'],fn))
    return db

offender_db = load_offender_db()
current_alert = None
lock = threading.Lock()

# ——— video generator with on/off control ———
def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        error = np.zeros((480,640,3),np.uint8); error[:] = (0,0,255)
        _,buf = cv2.imencode('.jpg',error)
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    while True:
        if not camera_on:
            # placeholder "Camera Off" frame
            off = np.zeros((480,640,3),np.uint8)
            cv2.putText(off, "Camera Off", (200,240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            _,buf = cv2.imencode('.jpg',off)
            time.sleep(0.1)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            break

        # — YOLO weapon & accident —
        for r in weapon_model.predict(source=frame, conf=0.4, verbose=False)[0].boxes:
            x1,y1,x2,y2 = map(int, r.xyxy[0])
            conf = r.conf.item()
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"Wpn {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        for r in accident_model.predict(source=frame, conf=0.4, verbose=False)[0].boxes:
            x1,y1,x2,y2 = map(int, r.xyxy[0])
            conf = r.conf.item()
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,f"Acc {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        # — DeepFace recognition —
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,5)
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            rep = DeepFace.represent(face,
                                     model_name="Facenet",
                                     detector_backend='opencv',
                                     enforce_detection=False)
            if len(rep)==0: continue
            emb = rep[0]["embedding"]
            for name, db_emb in offender_db.items():
                sim = np.dot(emb,db_emb)/(np.linalg.norm(emb)*np.linalg.norm(db_emb))
                if sim>0.7:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,name,(x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                    with lock:
                        current_alert = f"ALERT: {name} detected!"
                    break

        # encode & yield
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# Simple login check decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        user = request.form['username'].strip()
        pwd  = request.form['password']
        # look up user
        cur = sqlite3.connect(DB).execute(
            'SELECT password_hash FROM users WHERE username=?', (user,))
        row = cur.fetchone()
        if row and check_password_hash(row[0], pwd):
            session['logged_in']=True
            session['username']=user
            return redirect(url_for('index'))
        flash('Invalid credentials','error')
    return render_template('login.html')

# ——— registration endpoints ———
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        user = request.form['username'].strip()
        pwd  = request.form['password']
        if not user or not pwd:
            flash('Username and password required','error')
        else:
            pw_hash = generate_password_hash(pwd)
            try:
                with sqlite3.connect(DB) as conn:
                    conn.execute('INSERT INTO users(username,password_hash) VALUES(?,?)',
                                 (user,pw_hash))
                flash('Registration successful — please log in','success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already taken','error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(),
                                   mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/get_alert')
def get_alert():
    global current_alert
    with lock:
        a = current_alert; current_alert=None
    return jsonify(alert=a)

# ——— toggle camera endpoint ———
@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return jsonify(camera_on=camera_on)

ALLOWED = {'png','jpg','jpeg'}
def allowed_file(f): return f.rsplit('.',1)[-1].lower() in ALLOWED

@app.route('/upload')
def upload(): return render_template('upload_photos.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'file' not in request.files or 'name' not in request.form:
        return render_template('upload_photos.html', error="Missing name or file.")
    f = request.files['file']; name = request.form['name'].strip()
    if not name or f.filename=='' or not allowed_file(f.filename):
        return render_template('upload_photos.html', error="Invalid input.")
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(path)
    try:
        emb = DeepFace.represent(img_path=path, model_name='Facenet')[0]['embedding']
    except:
        return render_template('upload_photos.html', error="No face found.")
    np.save(os.path.join(app.config['EMBEDDINGS_FOLDER'], f"{name}_embedding.npy"), emb)
    offender_db.clear(); offender_db.update(load_offender_db())
    return render_template('upload_photos.html', success=f"{name} added.")

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,threaded=True)