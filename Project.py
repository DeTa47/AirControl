import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2
from math import hypot
import pydirectinput, asyncio

#import pyautogui
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose #Solution which performs pose detection
mp_draw = mp.solutions.drawing_utils #Solution which draws detected pose

pose = mp_pose.Pose() #Contains the actual algorithm for pose detection, instance of solutions pose class created and stored in pose variable
strt_flg = 0
sfl = 0


"""def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle"""

def euc_dis(point, base):
            return int(hypot(point[0]-base[0],point[1]-base[1]))

 
def calc(vector_b,vector_a):
        
        dot_product = np.dot(vector_b, vector_a)
        magnitude_product = np.linalg.norm(vector_b) * np.linalg.norm(vector_a)
        angle_rad = np.arccos(dot_product / magnitude_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

async def fing_cont(vid,w,h):

    vid_rgb = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
    results = hands.process(vid_rgb)
    
    if results.multi_hand_landmarks!= None:
 
      for hand_landmarks in results.multi_hand_landmarks:
           
            thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*h,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*w], dtype=np.float32)

            
            ring_finger_pip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y*h, 
                                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x*w], dtype=np.float32)

            #index_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*h, 
                                        #hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w], dtype=np.float32)

            #mid_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*h,
            #                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*w], dtype=np.float32)
            
            pinky_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y*h, 
                                  hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x*w], dtype=np.float32)
            
            thumb_to_ringpip = euc_dis(ring_finger_pip, thumb_tip) 

            #mid_to_ringpip = euc_dis(ring_finger_pip, mid_finger_tip)

            #index_to_ringPIP = euc_dis(ring_finger_pip, index_finger_tip)

            pinkytip_toMidPIP = euc_dis(ring_finger_pip, pinky_tip)

            global sfl

            cv2.putText(vid,f'{thumb_to_ringpip}',(5,115),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
            cv2.putText(vid,f'{thumb_to_ringpip}',(5,115),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

            cv2.putText(vid,f'{pinkytip_toMidPIP}',(5,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
            cv2.putText(vid,f'{pinkytip_toMidPIP}',(5,150),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

            if thumb_to_ringpip > 79 and sfl == 0:
                 pydirectinput.keyDown('w')
                 sfl = 1
            elif pinkytip_toMidPIP > 40 and sfl == 0:
                pydirectinput.keyDown('s')
                sfl = 2
            
            elif thumb_to_ringpip < 79 and sfl == 1: 
                pydirectinput.keyUp('w')
                sfl = 0

            elif pinkytip_toMidPIP > 40 and sfl == 2:    
                pydirectinput.keyUp('s')
                sfl = 0

            return 1
      
    else: return 0


async def controls(vid,results,h,w):
    
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

     
        vector_mid_to_Rwrist = right_wrist - right_mid
        vector_mid_to_Lwrist = left_wrist - left_mid
        
        Rwrist_angle = calc(right_mid, vector_mid_to_Rwrist)
        
        Lwrist_angle = calc(left_mid, vector_mid_to_Lwrist)

        cv2.putText(vid,f'Lwrist:{Lwrist_angle:.2f}',(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
        cv2.putText(vid,f'Lwrist:{Lwrist_angle:.2f}',(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.putText(vid,f'Rwrist:{Rwrist_angle:.2f}',(5,85),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
        cv2.putText(vid,f'Rwrist:{Rwrist_angle:.2f}',(5,85),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        global strt_flg
        
        if Rwrist_angle < 159 and Rwrist_angle > 150 and strt_flg == 0: 
        
            pydirectinput.keyDown('a')
            strt_flg = 1
        
        elif Lwrist_angle > 93 and Lwrist_angle < 112 and strt_flg == 0: 
        
            pydirectinput.keyDown('d')
            strt_flg = 2
        
        elif Lwrist_angle < 93 and strt_flg == 1:
             pydirectinput.keyUp('a')
             strt_flg = 0
        
        elif Rwrist_angle < 137 and strt_flg == 2:
             pydirectinput.keyUp('d')
             strt_flg = 0
        
        #else:
        #     pydirectinput.keyUp('a, d')
       
        return 1
             
    else:
         return 0



async def process_controller(vid,results,h,w):
    await asyncio.gather(controls(vid, results, h, w), fing_cont(vid,w,h))
    return 0     

while True:
        
        stat, img = cap.read()
        video = cv2.resize(img, (640,480))
        
        h,w,c = video.shape

        result = pose.process(video) #Provides landmarks and connections

         
        asyncio.run(process_controller(video, result, h, w))
 
        mp_draw.draw_landmarks(video, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
     
        cv2.imshow("Webcam footage", video)
        if cv2.waitKey(1) & 0xFF == 27:
            break
     
cv2.destroyAllWindows