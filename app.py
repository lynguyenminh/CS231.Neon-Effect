import numpy as np
import cv2
import streamlit as st
import mediapipe as mp


# load model
@st.cache(allow_output_mutation=True)
def load_mediapose():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    return mp_drawing, mp_drawing_styles, mp_pose

with st.spinner("Loading Model...."):
    mp_drawing, mp_drawing_styles, mp_pose=load_mediapose()


# global vairable
FRAME_WINDOW = st.image([])

# read effect 1 video
effect_frame = cv2.VideoCapture('effect-video/ball_energy.mp4')

# read effect 2 video
effect_frame_plasma_ball = cv2.VideoCapture('effect-video/Plasma-Ball.mp4')
effect_frame_punch = cv2.VideoCapture('effect-video/punch.mp4')

# read webcam
cap = cv2.VideoCapture(0)


effect_name = 'ball_energy'


with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)

        if effect_name == "ball_energy":
            success, effect = effect_frame.read()
            if success:
                height_frame, width_frame, _ = image.shape
                height_effect, width_effect, _ = effect.shape
                try:
                    results = pose.process(image)
                    left_hand = results.pose_landmarks.landmark[18]
                    right_hand = results.pose_landmarks.landmark[17]

                except: 
                    result = image

                if left_hand.y >=1 or right_hand.y >=1: 
                    result = image
                else: 
                    left_hand = [int(left_hand.x * width_frame), int(left_hand.y * height_frame)]
                    right_hand = [int(right_hand.x * width_frame), int(right_hand.y * height_frame)]

                    distance = np.linalg.norm(np.array(left_hand) - np.array(right_hand))

                    # resize anh
                    width_effect_final, height_effect_final = int(width_effect * distance / height_effect), int(distance)
                    effect = cv2.resize(effect, (width_effect_final, height_effect_final), interpolation=cv2.INTER_AREA)


                    # 1. mo rong theo 4 huong tao anh 2000x2000 sao cho tam qua cau o 1000x1000.
                    top = bot = 1000 - effect.shape[0]//2
                    left = right = 1000 - effect.shape[1]//2
                    background = cv2.copyMakeBorder(src=effect, top=top, bottom=bot, left=left, right=right, borderType=cv2.BORDER_REPLICATE) 

                    # 2. Tim center
                    x_center = (left_hand[0] + right_hand[0])//2
                    y_center = (left_hand[1] + right_hand[1])//2

                    # 3. cat anh 480x640 sao cho tam qua cau trung voi center
                    background = background[1000-y_center:1000-y_center+480, 1000-x_center:1000-x_center+640, :]

                    # 4. merge voi frame
                    result = cv2.addWeighted(image, 0.3, background, 0.7, 0)
                
            else: 
                result = image
                effect_name = 'plasma_ball'
        
        elif effect_name == 'plasma_ball': 
            success1, effect_plasma_ball = effect_frame_plasma_ball.read()
            success2, effect_punch = effect_frame_punch.read()
            if success1 and success2: 
                height_frame, width_frame, _ = image.shape

                try:
                    results = pose.process(image)

                    left_pinky = results.pose_landmarks.landmark[17]
                    left_elbow = results.pose_landmarks.landmark[13]
                    left_shoulder = results.pose_landmarks.landmark[11]
                except: 
                    continue

                if left_pinky.y >=1 or left_elbow.y >=1 and left_shoulder.y >=1: 
                    result = image
                else: 
                    left_pinky = [int(left_pinky.x * width_frame), int(left_pinky.y * height_frame)]
                    left_elbow = [int(left_elbow.x * width_frame), int(left_elbow.y * height_frame)]
                    left_shoulder = [int(left_shoulder.x * width_frame), int(left_shoulder.y * height_frame)]

                    distance = np.linalg.norm(np.array(left_pinky) - np.array(left_elbow))

                    if left_elbow[1] - left_pinky[1] > 50 and left_elbow[1] - left_shoulder[1] > 50: 
                        # 1. resize cho nho lai
                        new_height = int(0.5 * distance)
                        new_width = int(new_height * 1280 / 720)
                        effect_plasma_ball = cv2.resize(effect_plasma_ball, (new_width, int(0.7 * distance)), interpolation=cv2.INTER_AREA)

                        # 2. expand effect_plasma_ball 
                        effect_plasma_ball = effect_plasma_ball[5:-5, 5:-5]
                        top = bot = 1000 - effect_plasma_ball.shape[0]//2
                        left = right = 1000 - effect_plasma_ball.shape[1]//2
                        
                        background = cv2.copyMakeBorder(src=effect_plasma_ball, top=top, bottom=bot, left=left, right=right, borderType=cv2.BORDER_CONSTANT) 
                        # 3. cat anh
                        background = background[1000-left_pinky[1]:1000-left_pinky[1]+480, 1000-left_pinky[0]:1000-left_pinky[0]+640, :]

                    else: 
                        background = cv2.resize(effect_punch, (640, 480), interpolation=cv2.INTER_AREA)

                    result = cv2.addWeighted(image, 0.3, background, 0.7, 0)
            else: 
                effect_name='none'
                result = image
        else: 
            result = image
            st.write('End code')
            break
        
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(result)
