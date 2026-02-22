import cv2 # مكتبه لمعالجه الصور والفيديو
from deepface import DeepFace # نستخدمها للتحليل 

# Haar Cascade Classifier مودل من مكتبه cv2 مدرب انه يحدد ويكشف الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the camera
video_capture = cv2.VideoCapture(0)

# Variables to store age and gender
detected_age = None
detected_gender = None

while True: # تبقى الكاميرا تشتغل حتى اقفلها
    ret, frame = video_capture.read() #5-8 تشوف اذا في وجه بالفريم وهي مربوطه ب لاين 
    if not ret:
        break

    # Convert the frame to gray  لان الرمادي اسرع في معالجة الصور من ناحيه كل بيكسل ياخذ درجات الرمادي ..قيمه وحده
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces.... يكشف الوجه من المودل cascade....حجم التصغير للصوره لما يدور عالوجه
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face             لون المستطيل ازرق
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        points = [
            (x, y), (x + w, y), (x, y + h), (x + w, y + h),
            (x + w // 2, y), (x, y + h // 2), (x + w, y + h // 2), (x + w // 2, y + h // 2)
        ]
        for point in points: # points around the frame
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

        # Extract the face يقص الوجه المحدد داخل الاطار من لاين 23 ويبدا ينفذ التحليل المطلوب
        face = frame[y:y+h, x:x+w]

        # Analyze age and gender once
        if detected_age is None or detected_gender is None:
            try:    #  تحلل الصوره عشان تحدد العمر والجنس
                analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False) # يسكب التحقق اذا مالقى وجه واضح

                # يشوف اذا كانت ليست 
                if isinstance(analysis, list):
                    analysis = analysis[0] # تاخذ العنصر الاول بس لان المطلوب تحلل وجه واحد فقط

                detected_age = analysis['age'] # دكشنري بعد ماتاكد من الوجه يخزن التقريب فيهم
                detected_gender = analysis['gender']

            except Exception as e:#اذا كان في اخطاء بالوضوح للصوره يتوقف ويعطي انه غير موجوده 
                print(f"Error analyzing age and gender: {e}")
                detected_age = "N/A"
                detected_gender = "N/A"
                
        # Analyze emotion اطلبه التعابير فقط عشان تكون متغيره باستمرار مع الوجه
        try:
            emotion_analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            if isinstance(emotion_analysis, list):
                emotion_analysis = emotion_analysis[0]
             #تاخذ اكثر تعبير مسيطر بالوجه وتحلل بناء عليه
            detected_emotion = max(emotion_analysis['emotion'], key=emotion_analysis['emotion'].get)
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            detected_emotion = "N/A"

        # Display the results
    text = f"Emo: {detected_emotion}, Age: {detected_age}, Gen: {detected_gender}"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # تضيف النصوص فوق الفريم 

    # Display frame يعرض الفريم مع المستطيل والنصوص
    cv2.imshow('Face Detection', frame)

    # يقفل الكاميرا 
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()