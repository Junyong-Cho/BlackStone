import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import preprocessing as pc

MIN_VISIBILITY_THRESHOLD = 0.6

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

cap = cv2.VideoCapture(0)
count = 1

if not cap.isOpened() :
    print('No Webcam')
    exit()
    
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
record = cv2.VideoWriter('output_pose.avi', fourcc, 20.0, (640, 480)) # 웹캠 해상도에 맞게 조정
    
# window_name = 'motion_capturing'

# cv2.nameWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 800, 600)

pose_data_frames = []
is_capturing = False
action_label = 'PT'

angle_columns = [name for _,_,_,name in pc.ANG_LANDMARKS]
angle_columns.append('leg_spread_angle')

countdown_start_time = 0
countdown_active = False
countdown_num = 0

print('start/pause : s, stop : q')

while cap.isOpened() :
    ret, frame = cap.read()
    if not ret :
        break
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if result.pose_landmarks :
        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        landmarks = result.pose_landmarks.landmark
        
        if is_capturing and not countdown_active:
            current_frame_landmarks_data = []
            for id, lm in enumerate(landmarks) :
                if lm.visibility < MIN_VISIBILITY_THRESHOLD :
                    current_frame_landmarks_data.extend(([np.nan]*3)+[lm.visibility])
                else :
                    current_frame_landmarks_data.extend([lm.x, lm.y, lm.z, lm.visibility])
            calc_angles = pc.get_all_angles(landmarks)
            
            for name in angle_columns :
                current_frame_landmarks_data.append(calc_angles[name])
            
            pose_data_frames.append(current_frame_landmarks_data)
            
    if countdown_active :
        time_flow = time.time() - countdown_start_time
        current_countdown_num = 3 - int(time_flow)
        
        if current_countdown_num > 0 :
            text = str(current_countdown_num)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)[0]
            text_x = int((image.shape[1]-text_size[0])/2)
            text_y = int((image.shape[0]+text_size[1])/2)
            
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 5, cv2.LINE_AA)
        else :
            text = 'GO'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)[0]
            text_x = int((image.shape[1]-text_size[0])/2)
            text_y = int((image.shape[0]+text_size[1])/2)
            
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 5, cv2.LINE_AA)
            if time_flow > 3.5 :
                countdown_active = False
                is_capturing = True
        
    status_text = "CAPTURING" if is_capturing else "IDLE"
    cv2.putText(image, f'Status: {status_text}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    if is_capturing :
        cv2.putText(image, f'label: {action_label}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Pose', image)
    if is_capturing :
        record.write(image)
    
    key = cv2.waitKey(5)&0xFF
    if key == ord('s') :
        if not is_capturing and not countdown_active:
            countdown_active = True
            countdown_start_time = time.time()
            current_countdown_num = 3
        elif is_capturing and not countdown_active :
            is_capturing = False
            print(f'캡쳐 중지. {len(pose_data_frames)} 프레임 수집.')
            if pose_data_frames :
                
                data = np.array(pose_data_frames)
                
                imputed_data_frames = pc.impute_pose_data(data)
                cols = []
                for i in range(33) :
                    cols.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z', f'lm{i}_visibility'])
                
                cols.extend(angle_columns)
                
                df = pd.DataFrame(imputed_data_frames, columns=cols)
                df['label'] = action_label
                timestamp = int(time.time())
                df.to_csv(f'{action_label}/{action_label}_{count}.csv', index=False)
                count += 1
                print(f'데이터 저장 성공')
                pose_data_frames = []
    elif key==ord('q') :
        print('종료')
        break

record.release()
pose.close()
cap.release()
cv2.destroyAllWindows()