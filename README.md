# 📸🔐 Live Security Surveillance

A real-time security surveillance system using **YOLOv8**, **DeepFace**, **Flask**, and **OpenCV**, built for detecting weapons, accidents, and known faces through live webcam feeds. 
This project allows uploading faces to a local database, login-protected access, and visual alerting directly in the browser.

---

## 📸 Features

- 🔍 **Weapon Detection** using YOLOv8
- 🚗 **Accident Detection** via custom-trained YOLOv8 model
- 🧠 **Facial Recognition** with DeepFace
- 🧑‍💻 **Login and Registration** (with SQLite and hashed passwords)
- ⬆️ Upload face images to add to database
- 🎥 Real-time webcam feed with detection overlays
- 🔐 Secure routes using Flask session-based login
- 💾 Embeddings and uploads stored locally

---

## 🧠 Tech Stack

| Technology     | Purpose                          |
|----------------|----------------------------------|
| Flask          | Web server & routing             |
| OpenCV         | Camera feed & video processing   |
| YOLOv8 (Ultralytics) | Object detection (weapons, accidents) |
| DeepFace       | Facial recognition               |
| SQLite         | User authentication DB           |
| HTML / Jinja   | Frontend templating              |
| NumPy          | Vector similarity + embeddings   |

---

## 🗂️ Project Structure

Live_Security_Surveillance/<br>
│<br>
├── app.py # Main Flask application<br>
├── templates/ # HTML templates<br>
├── models/ # YOLOv8 model weights (.pt)<br>
├── embeddings/ # Face embeddings (.npy)<br>
├── uploads/ # Uploaded user images<br>
├── users.db # SQLite database (auto-created)<br>
├── requirements.txt # Python dependencies<br>
├── .gitignore<br>

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/anishvs14/Live_Security_Surveillance.git
cd Live_Security_Surveillance
```

2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. Install Requirements
```bash
pip install -r requirements.txt
```
4. Run the App
```bash
python app.py
Visit http://localhost:5000 in your browser.
```


## 👤 User Roles
Register/Login required to access the live surveillance dashboard.

Upload Faces via the form for recognition.

Admin view with live camera + detection overlays.

## ⚠️ Notes
Webcam Access: The app uses your local webcam (cv2.VideoCapture(0)).

Model Files: The YOLOv8 weights (weapon_best.pt, accident_best.pt) are required in the models/ directory.

Embeddings: Stored as .npy files inside the embeddings/ folder.

## 📷 Screenshots
![image](https://github.com/user-attachments/assets/f1025089-3141-403b-8f28-1668c2b504e6)
Login page
![image](https://github.com/user-attachments/assets/372f5852-07d7-4e5a-9322-5deebefbfa4b)<br>
Weapon Detection
![image](https://github.com/user-attachments/assets/d5783b4b-7cf3-4ab6-be8d-9f67b34a5d7d)<br>
Accident Detection
![image](https://github.com/user-attachments/assets/5e27e2c6-a8f0-449d-8dbb-2e7bc5997641)
Upload Images of People
![Screenshot 2025-07-08 154311](https://github.com/user-attachments/assets/c428a332-3953-46da-bbf7-223a80e1dab1)<br>
Facial Recognition
