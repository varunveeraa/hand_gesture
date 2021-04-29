#importing the libraries
import cv2
import mediapipe as mp
import pickle
import pandas as pd


#mediapipe variables
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with open('gesture.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
aMusic = 'C:/Users/Hp/AppData/Local/Amazon Music/Amazon Music.exe'
od = 'C:/Users/Hp\AppData/Local/Microsoft/OneDrive/OneDrive.exe'

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image = cv2.flip(image, 1) #flip on horizontal
        
        image.flags.writeable = False #flag set false
        
        #detections
        results = hands.process(image)
        
        image.flags.writeable = True #flag set true
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            
        #exporting the coordinates of the new gesture/action
        try:
            hand_pose = results.multi_hand_landmarks[0].landmark 
            rowTemp = list([[landmark.x, landmark.y, landmark.z] for landmark in hand_pose])
            row = sum(rowTemp, []) #addin' up lists
           
            #making detections
            X = pd.DataFrame([row])
            gesture_class = model.predict(X)[0]
            gesture_prob = model.predict_proba(X)[0]
            
            #printing the results
            cv2.putText(image, gesture_class, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(gesture_class, gesture_prob)
            
        except:
            pass
        
        cv2.imshow('handTracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
