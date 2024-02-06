import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2

#import pyautogui

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose #Solution which performs pose detection
mp_draw = mp.solutions.drawing_utils #Solution which draws detected pose

pose = mp_pose.Pose() #Contains the actual algorithm for pose detection, instance of solutions pose class created and stored in pose variable


 
def calc(vector_b,vector_a):
        
        # Calculate angle (in degrees) using dot product and arccosine
        dot_product = np.dot(vector_b, vector_a)
        magnitude_product = np.linalg.norm(vector_b) * np.linalg.norm(vector_a)
        angle_rad = np.arccos(dot_product / magnitude_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def hipAngle():
      if results.pose_landmarks:

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
        
        
        hip_angle_deg = calc(vector_hip_to_knee,vector_hip_to_shoulder)
        return hip_angle_deg



def controls(vid,results,h,w):
    
    if results.pose_landmarks != None:
       
        left_y  = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)

        mid_y = abs(left_y+right_y)//2
        
        right_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
        
        left_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
                                
        right_mid = np.array([mid_y,w])
        left_mid = np.array([0,mid_y])

        #Quentin Tarantino approves
        right_ankle = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x])
        
        right_knee = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
        ])
        right_foot_index = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
        ])

        left_ankle = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
        ])
        left_knee = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x
        ])
        left_foot_index = np.array([
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
        ])
        
        lf_angle = calculate_angle(left_knee,left_ankle,left_foot_index)
        rf_angle = calculate_angle(right_knee,right_ankle,right_foot_index)
    
        """vector_ankle_to_kneeR = right_knee - right_ankle
        vector_ankle_to_footR = right_foot_index - right_ankle

        vector_ankle_to_kneeL = left_knee - left_ankle
        vector_ankle_to_footL = left_foot_index - left_ankle"""
     
        vector_mid_to_Rwrist = right_wrist - right_mid
        vector_mid_to_Lwrist = left_wrist - left_mid
        
        Rwrist_angle = calc(right_mid, vector_mid_to_Rwrist)
        
        Lwrist_angle = calc(left_mid, vector_mid_to_Lwrist)
        
        

        
        cv2.putText(vid,f'Angle L_Hand: {Lwrist_angle:.2f}, Angle R_Hand: {Rwrist_angle:.2f}',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),1)
        

        
        if Rwrist_angle < 157 and Rwrist_angle > 146: 
            cv2.putText(vid,'A',(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            #pyautogui.hold(['a'])
        
        elif Lwrist_angle > 85 and Lwrist_angle < 110: 
            cv2.putText(vid,'D',(5,80),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            #pyautogui.hold(['d'])
        
      

    

        cv2.putText(vid,f'RF Angle:{rf_angle:.2f}, LF Angle:{lf_angle:.2f}',(5,120),cv2.FONT_HERSHEY_COMPLEX,0.57,(0,155,0),2)
        
        """
        if cv2.waitKey(1) & 0xFF == 27:
            return 1
        """
        
        return 1
             
    else:
         return 0




while True:
        stat, img = cap.read()
        vid = cv2.resize(img, (640,480))
        results = pose.process(vid) #Provides landmarks and connections
        
        mp_draw.draw_landmarks(vid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h,w,c = vid.shape

        opVid = np.zeros([h,w,c])
        opVid.fill(0)
      
        mp_draw.draw_landmarks(opVid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

       
        """
        
        flag = 0
        estimated_Angle = hipAngle()
        
        if flag == 0:
        if estimated_Angle > "space for value" and wrist_euclidian_distance < "space for value":
            flag = 1
            
        else:
            flag = 0
        
        if flag == 1:
        while true:
            for_break = controls(vid,results,w,h)
            if for_break == 1:
                break
        
        
        """

        controls(vid,results,h,w)
        
        cv2.imshow("Extracted pose",opVid)

        print(results.pose_landmarks)

        cv2.imshow("Webcam footage", vid)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    #break 
cv2.destroyAllWindows