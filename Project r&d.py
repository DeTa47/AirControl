import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from math import hypot
from time import time

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose #Solution which performs pose detection
mp_draw = mp.solutions.drawing_utils #Solution which draws detected pose

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7) #Contains the actual algorithm for pose detection, instance of solutions pose class created and stored in pose variable

def hipAngle():
    if results.pose_landmarks != None:
        

        # Extract landmark coordinates
        right_hip = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * h
        ])
        right_knee = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * h
        ])
        right_shoulder = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
        ])

        # Calculate vectors
        vector_hip_to_knee = right_knee - right_hip
        vector_hip_to_shoulder = right_shoulder - right_hip

       
        hip_angle_deg = calc(vector_hip_to_knee, vector_hip_to_shoulder)

        # Display angle on the video
        cv2.putText(vid, f'Hip Angle: {hip_angle_deg:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    else:
        cv2.putText(vid,'Pose detection failed',(5,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

def calc(vector_b,vector_a):
        
        # Calculate angle (in degrees) using dot product and arccosine
        dot_product = np.dot(vector_b, vector_a)
        magnitude_product = np.linalg.norm(vector_b) * np.linalg.norm(vector_a)
        angle_rad = np.arccos(dot_product / magnitude_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg


def controls(vid,results,h,w):
    
    if results.pose_landmarks != None:
      
        left_y  = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)

        mid_y = abs(left_y+right_y)//2
        
        right_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
        
        left_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
                                
        right_mid = np.array([w,mid_y])
        left_mid = np.array([mid_y,w])

        right_ankle = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h
        ])
        right_knee = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * h
        ])
        right_foot_index = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h
        ])

        left_ankle = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
        ])
        left_knee = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h
        ])
        left_foot_index = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * w,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h
        ])
        
       
        vector_ankle_to_kneeR = right_knee - right_ankle
        vector_ankle_to_footR = right_foot_index - right_ankle

        vector_ankle_to_kneeL = left_knee - left_ankle
        vector_ankle_to_footL = left_foot_index - left_ankle
     
        vector_mid_to_Rwrist = right_wrist - right_mid
        vector_mid_to_Lwrist = left_wrist - left_mid
        
        Rwrist_angle = calc(vector_mid_to_Rwrist, right_mid)
        
        
        Lwrist_angle = calc(vector_mid_to_Lwrist,left_mid)
       

        
        cv2.putText(vid,f'Angle L_Hand: {Lwrist_angle:.2f}, Angle R_Hand: {Rwrist_angle:.2f}',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),1)
        

        """  
        if Rwrist_angle < 157 and Rwrist_angle > 146: 
            cv2.putText(vid,'A',(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            #pyautogui.hold(['a'])
        
        elif Lwrist_angle > 85 and Lwrist_angle < 110: 
            cv2.putText(vid,'D',(5,80),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            #pyautogui.hold(['d'])
        """
      

        
        
        rf_angle = calc(vector_ankle_to_footR,vector_ankle_to_kneeR)
        lf_angle = calc(vector_ankle_to_footL,vector_ankle_to_kneeL)

        cv2.line(vid, (w//2, 0), (w//2, h), (255, 0, 0), 2) #vertical line
        cv2.line(vid, (0, mid_y),(w, mid_y),(255, 0, 0), 2) #horizontal line
        
        """
        cv2.line(vid, (mid_y,right_wrist), (right_wrist,mid_y),(255, 0, 0),2) #right hand line
        cv2.line(vid, (mid_y,left_wrist), (left_wrist,mid_y),(255, 0, 0),2) #left hand line
        """
        cv2.putText(vid,f'RF Angle:{rf_angle:.2f}, LF Angle:{lf_angle:.2f}',(5,120),cv2.FONT_HERSHEY_COMPLEX,0.57,(0,0,0),2)
        
        """
        if cv2.waitKey(1) & 0xFF == 27:
            return 1
        """
             
    else:
        cv2.putText(vid,f'Subject out of frame',(5,30),cv2.FONT_HERSHEY_COMPLEX,0.57,(0,0,255),2) 


while True:
    stat, img = cap.read()
    vid = cv2.resize(img, (1280,720))
    h,w,c = vid.shape
    opVid = np.zeros([h,w,c])
    opVid.fill(0)
    results = pose.process(vid) #provides landmarks and connections
    
    mp_draw.draw_landmarks(vid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(opVid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    
    cv2.imshow("Extracted pose",opVid)

    print(results.pose_landmarks)

    controls(vid,results,h,w)

    cv2.imshow("Webcam footage", vid)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        