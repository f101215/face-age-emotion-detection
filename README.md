# Real-Time Face Age, Gender & Emotion Detection
This project performs **real-time face detection and facial attribute analysis** using a WEBCAM.
It detects and displays:

- Face location
- Estimated Age
- Gender
  -Dominant Emotion (Happy, Sad, Angry, Neutral, etc.)
  
The system uses **OpenCV (Haar Cascade)** for face detection and **DeepFace (Deep Learning models)** for facial attribute analysis.
---------------------------
# Technologies Used

- Python
- OpenCV
- DeepFace
- Haar Cascade Classifier
- NumPy
----------------------------
# System Workflow

1. Capture live video from webcam using OpenCV.
2. Convert frame to grayscale (faster face detection).
3. Detect faces using Haar Cascade classifier.
4. Extract detected face region (ROI).
5. Analyze the face using DeepFace:
   - Age (detected once to optimize performance)
   - Gender (detected once)
   - Emotion (detected continuously in real-time)
6. Draw:
   - Blue bounding box
   - Green reference points
   - Text displaying Age, Gender, and Emotion

-----------------------------

# Installation

Install required libraries:

```bash
pip install opencv-python deepface
```
---------------------------------
to run :
use python detect.py
to close press Q
--------------------------------
output display :
blue rectangle around detected face
Green facial reference points
Text above the face showing:
Emotion
Estimated Age
gender
-------------------------------
developed by farah 
Artificial Intelligence Student
Computer Vision & Deep Learning Enthusiast