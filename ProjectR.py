import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import pydirectinput, asyncio
from configparser import ConfigParser
from PyQt5.QtWidgets import QMessageBox


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

config = ConfigParser()
config.read("control_config.ini")
config_data = config["CONTROLS"]

thumbs_up = config_data["thumbs_up"]
pinky_finger_up = config_data["pinky_finger_up"]
turn_left = config_data["steer_left"]
turn_right = config_data["steer_right"]

strt_flg = 0
sfl = 0

exit_flag = 0

def pop_up_box(message_in_a_bottle):
        msg = QMessageBox()
        msg.setWindowTitle("Warning!")
        if message_in_a_bottle == "Person not Detected!":
            msg.setText(message_in_a_bottle)
            msg.setInformativeText("Ensure that surroundings are well lit and the subject is in frame at an appropriate distance")
        elif message_in_a_bottle == "Hands not Detected!":
            msg.setText(message_in_a_bottle)
            msg.setInformativeText("Open your palm such that it is facing the camera")
        else: msg.setText(message_in_a_bottle)
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(msg.Ok)
        x = msg.exec_()

def calc(vector_b,vector_a):
        
        # Calculate angle (in degrees) using dot product and arccosine
        dot_product = np.dot(vector_b, vector_a)
        magnitude_product = np.linalg.norm(vector_b) * np.linalg.norm(vector_a)
        angle_rad = np.arccos(dot_product / magnitude_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

def euc_dis(point, base):
            return int(hypot(point[0]-base[0],point[1]-base[1]))


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

            if thumb_to_ringpip > 100 and sfl == 0:
                 pydirectinput.keyDown(thumbs_up)
                 sfl = 1
            elif pinkytip_toMidPIP > 60 and sfl == 0:
                pydirectinput.keyDown(pinky_finger_up)
                sfl = 2
            
            elif thumb_to_ringpip < 100 and sfl == 1: 
                pydirectinput.keyUp(thumbs_up)
                sfl = 0

            elif pinkytip_toMidPIP > 60 and sfl == 2:    
                pydirectinput.keyUp(pinky_finger_up)
                sfl = 0

            return 1
      
    else: return 0


async def controls(vid,mp_pose, results,h,w):
    
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

        #cv2.putText(vid,f'Lwrist:{Lwrist_angle:.2f}',(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
        #cv2.putText(vid,f'Lwrist:{Lwrist_angle:.2f}',(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        #cv2.putText(vid,f'Rwrist:{Rwrist_angle:.2f}',(5,85),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)
        #cv2.putText(vid,f'Rwrist:{Rwrist_angle:.2f}',(5,85),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        global strt_flg
        
        if Rwrist_angle < 159 and Rwrist_angle > 150 and strt_flg == 0: 
        
            pydirectinput.keyDown(turn_left)
            strt_flg = 1
        
        elif Lwrist_angle > 93 and Lwrist_angle < 112 and strt_flg == 0: 
        
            pydirectinput.keyDown(turn_right)
            strt_flg = 2
        
        elif Lwrist_angle < 93 and strt_flg == 1:
             pydirectinput.keyUp(turn_left)
             strt_flg = 0
        
        elif Rwrist_angle < 137 and strt_flg == 2:
             pydirectinput.keyUp(turn_right)
             strt_flg = 0
        
        #else:
        #     pydirectinput.keyUp('a, d')
       
        return 1
             
    else:
         return 0



async def process_controller(vid,mp_pose, results,h,w):
    await asyncio.gather(controls(vid, mp_pose, results, h, w), fing_cont(vid,w,h))
    return 0 
        


"""
     def show_msg(vid, results, mp_pose, mp_draw, toshow):
    mp_draw.draw_landmarks(vid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if toshow == 0:
        cv2.putText(vid,'Detection failed',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,0),4)
        cv2.putText(vid,"Detection failed",(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)
    else:
        cv2.putText(vid,"Detection successful",(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,0),4)
        cv2.putText(vid,"Detection successful",(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,255,0),2)
    cv2.imshow("Webcam footage",vid)"""
     

def end_program():
   global exit_flag
   exit_flag = 1
   

def start_program():

    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose #Solution which performs pose detection
    mp_draw = mp.solutions.drawing_utils #Solution which draws detected pose

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7) #Contains the actual algorithm for pose detection, instance of solutions pose class created and stored in pose variable

    #palm_shown = 0

    global exit_flag
    if exit_flag == 1:
         exit_flag = 0

    
    try:    
     while exit_flag!=1:
     
        stat, img = cap.read(0)

        if img.all() == None:
            raise Exception('Camera disconnected')

        vid = cv2.resize(img, (640,480))

        h,w,c = vid.shape
       
        results = pose.process(vid) #provides landmarks and connections
             
       
        if results.pose_landmarks!=None:
                #show_msg(vid, results, mp_pose, mp_draw, 1)
                mp_draw.draw_landmarks(vid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                asyncio.run(process_controller(vid, mp_pose, results, h, w))

                key = cv2.waitKey(1)
                if key == 127:  # 'DEL' key
                    break
            
        else:
                 #show_msg(vid, results, mp_pose, mp_draw, 0)
                 cv2.putText(vid,'Detection failed',(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,0),4)
                 cv2.putText(vid,"Detection failed",(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.57,(0,0,255),2)
                 
                 key = cv2.waitKey(1)
                 if key == 127:  # 'DEL' key
                    break
            
        """elif results.pose_landmarks==None and hand_results.multi_hand_landmarks==None:
                cv2.destroyAllWindows
                pop_up_box("Person not Detected!")
            
            elif results.pose_landmarks!=None and hand_results.multi_hand_landmarks==None:
                cv2.destroyAllWindows()
                pop_up_box("Hands not Detected!")"""
            
            #else: raise Exception
        cv2.imshow("Web footage",vid)

        key = cv2.waitKey(1)
        if key == 127:  # 'DEL' key
                break

    except cv2.error:
                pop_up_box("Camera missing")              
                exit_flag = 0
                #palm_shown = 0
                cap.release()
                cv2.destroyAllWindows()
                         
     
    except Exception as e: 
                
                exit_flag = 0
                pop_up_box(str(e))
                exit_flag = 0
                #palm_shown = 0           
                
       
    exit_flag = 0
    #palm_shown = 0
    cap.release()
    cv2.destroyAllWindows()