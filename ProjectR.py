import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from configparser import ConfigParser

config = ConfigParser()
config.read("control_config.ini")
config_data = config["CONTROLS"]

rf_pedal = config_data["rf_pedal"]
lf_pedal = config_data["lf_pedal"]
turn_left = config_data["steer_left"]
turn_right = config_data["steer_right"]

exit_flag = 0

def hipAngle(results,vid,w,h,mp_pose):

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

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle  

def calc(vector_b,vector_a):
        
        # Calculate angle (in degrees) using dot product and arccosine
        dot_product = np.dot(vector_b, vector_a)
        magnitude_product = np.linalg.norm(vector_b) * np.linalg.norm(vector_a)
        angle_rad = np.arccos(dot_product / magnitude_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg


def controls(vid,results,h,w,mp_pose):
    
      
        left_y  = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)

        mid_y = abs(left_y+right_y)//2
        
        right_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
        
        left_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
                                
        right_mid = np.array([w,mid_y])
        left_mid = np.array([mid_y,w])
        
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
        
        Rwrist_angle = calc(vector_mid_to_Rwrist, right_mid)
        
        
        Lwrist_angle = calc(vector_mid_to_Lwrist,left_mid)
       

        cv2.putText(vid,f'Angle L_Hand: {Lwrist_angle:.2f}, Angle R_Hand: {Rwrist_angle:.2f}',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,0),4)
        cv2.putText(vid,f'Angle L_Hand: {Lwrist_angle:.2f}, Angle R_Hand: {Rwrist_angle:.2f}',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)
        cv2.putText(vid,f'Angle L_foot: {lf_angle:.2f}, Angle R_foot: {rf_angle:.2f}',(5,120),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,0),4)
        cv2.putText(vid,f'Angle L_foot: {lf_angle:.2f}, Angle R_foot: {rf_angle:.2f}',(5,120),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)

    
        if Rwrist_angle < 185 and Rwrist_angle > 175: 
        
            cv2.putText(vid,f'D',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)
        
        elif Lwrist_angle > 137 and Lwrist_angle < 145:
          
            cv2.putText(vid,f'A',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)
        
        elif rf_angle > 171 and rf_angle < 180:
  
            cv2.putText(vid,f'W',(5,150),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)

        elif lf_angle > 150 and lf_angle < 165:

            cv2.putText(vid,f'S',(5,150),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)

    
        
        
        


def end_program():
   global exit_flag
   exit_flag = 1
   

def start_program():

    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose #Solution which performs pose detection
    mp_draw = mp.solutions.drawing_utils #Solution which draws detected pose

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7) #Contains the actual algorithm for pose detection, instance of solutions pose class created and stored in pose variable

    global exit_flag

    while exit_flag!=1:
        
        
        stat, img = cap.read()

        vid = cv2.resize(img, (640,480))

        h,w,c = vid.shape
       
        results = pose.process(vid) #provides landmarks and connections
        
        if results.pose_landmarks!=None:
         
         controls(vid,results,h,w,mp_pose)
         mp_draw.draw_landmarks(vid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
         cv2.imshow("Webcam footage", vid)


         key = cv2.waitKey(1)
         if key == 27:  # 'ESC' key
            break

        else: 
         mp_draw.draw_landmarks(vid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
         cv2.putText(vid,f'Subject out of frame',(5,30),cv2.FONT_HERSHEY_COMPLEX,0.57,(0,0,255),2)
         cv2.imshow("Webcam footage", vid)
         key = cv2.waitKey(1)
         if key == 27:
             break
          
         
        
    exit_flag = 0
    cap.release()
    cv2.destroyAllWindows()