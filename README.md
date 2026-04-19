# 😴 Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![dlib](https://img.shields.io/badge/dlib-Computer%20Vision-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
A real-time **Driver Drowsiness Detection System** using Computer Vision and facial landmark detection. The system monitors the driver's eye movements and triggers an **alert alarm** when drowsiness is detected — potentially saving lives by preventing accidents caused by fatigue.

---

## 🎯 Problem Statement
Every year, thousands of road accidents occur due to driver fatigue and drowsiness. This system provides a low-cost, real-time solution to detect drowsiness using a standard webcam and alert the driver before an accident happens.

---

## ✨ Features
- 👁️ Real-time eye detection using facial landmarks
- 📊 Eye Aspect Ratio (EAR) calculation for drowsiness detection
- 🔔 Audio alert when drowsiness is detected
- 📹 Live webcam feed with visual indicators
- ⚡ Fast and lightweight — works on standard hardware

---

## 🛠️ Tech Stack
| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming language |
| OpenCV | Video capture & image processing |
| dlib | Facial landmark detection |
| imutils | Image utilities |
| scipy | EAR calculation |
| pygame / playsound | Alert sound |

---

## 📁 Project Structure
```
drowsiness_detection/
│
├── drowsiness_detection.py    # Main Python script
├── shape_predictor_68_face_landmarks.dat  # Pre-trained model
├── alert.wav                  # Alert sound file
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ How It Works

### 1. Face Detection
The system uses **dlib's frontal face detector** to locate the driver's face in each video frame.

### 2. Facial Landmark Detection
Using the **68-point facial landmark model**, it identifies key points around the eyes.

### 3. Eye Aspect Ratio (EAR)
The EAR is calculated using the formula:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

- When eyes are **open** → EAR is high (~0.3)
- When eyes are **closed** → EAR drops below threshold (~0.25)

### 4. Alert System
If EAR stays below threshold for **consecutive frames** → alarm is triggered!

---

## 🚀 Installation & Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the landmark model
Download `shape_predictor_68_face_landmarks.dat` from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project folder.

### Step 4: Run the project
```bash
python drowsiness_detection.py
```

---

## 📦 Requirements
```
opencv-python
dlib
imutils
scipy
pygame
numpy
```

---

## 📊 Results
| Condition | EAR Value | System Response |
|-----------|-----------|-----------------|
| Eyes Open | > 0.25 | No Alert ✅ |
| Eyes Closing | 0.20 - 0.25 | Warning State ⚠️ |
| Eyes Closed | < 0.20 | ALERT! 🔔 |

---

## 🔮 Future Improvements
- Add yawning detection
- Add head pose estimation
- Mobile app integration
- Night vision support

---

## 👩‍💻 Author
**Palak Gupta**
- 🎓 Computer Science Student
- 📧 [your-email@example.com]
- 🔗 [LinkedIn Profile]

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).

---
⭐ If you found this project helpful, please give it a star!
