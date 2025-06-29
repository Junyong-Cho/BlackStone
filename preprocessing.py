# 데이터 전처리 코드

import numpy as np
import mediapipe

pl = mediapipe.solutions.pose.PoseLandmark

# 랜드마크 좌표 기준 각도 계산 함수
def calculate_angle(a_coords, b_coords, c_coords) :
    
    if a_coords is None or b_coords is None or c_coords is None :
        return np.nan
    
    a = np.array(a_coords)
    b = np.array(b_coords)
    c = np.array(c_coords)
    
    ba = a-b
    bc = c-b
    
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    if magnitude_ba*magnitude_bc==0.0 :
        return np.nan
    
    dot_product = np.dot(ba,bc)
    
    cosine_angle = dot_product/(magnitude_ba*magnitude_bc)
    cosine_angle = np.clip(cosine_angle, -1.0,1.0)
    
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# 프레임 당 랜드마크 배열에서 특정 랜드마크 배열의 x,y,z 좌표 추출 함수
def get_landmark_coords(landmarks, enum, min_visibility_threshold = 0.6) :
    
    if enum.value>= len(landmarks) or landmarks[enum.value] is None :
        return [np.nan]*3
    
    lm = landmarks[enum.value]
    
    if lm.visibility < min_visibility_threshold :
        return [np.nan]*3
    
    return [lm.x, lm.y, lm.z]

# mediapipe 랜드마크의 각도 계산을 위한 튜플 리스트
ANG_LANDMARKS = [
    (pl.RIGHT_HIP, pl.RIGHT_SHOULDER, pl.RIGHT_ELBOW,"right_shoulder_angle"),
    (pl.RIGHT_SHOULDER, pl.RIGHT_ELBOW, pl.RIGHT_WRIST,"right_elbow_angle"),
    (pl.RIGHT_SHOULDER, pl.RIGHT_HIP, pl.RIGHT_KNEE,"right_hip_angle"),
    (pl.RIGHT_HIP, pl.RIGHT_KNEE, pl.RIGHT_ANKLE,"right_knee_angle"),
    (pl.LEFT_HIP, pl.LEFT_SHOULDER, pl.LEFT_ELBOW,"left_shoulder_angle"),
    (pl.LEFT_SHOULDER, pl.LEFT_ELBOW, pl.LEFT_WRIST,"left_elbow_angle"),
    (pl.LEFT_SHOULDER, pl.LEFT_HIP, pl.LEFT_KNEE,"left_hip_angle"),
    (pl.LEFT_HIP, pl.LEFT_KNEE, pl.LEFT_ANKLE,"left_knee_angle")
]

# 추출한 랜드마크 배열을 받아서 모든 각도를 계산하여 return하는 함수
def get_all_angles(landmarks) :
    angles = {}
    for a, b, c, angle in ANG_LANDMARKS :
        al = get_landmark_coords(landmarks, a)
        bl = get_landmark_coords(landmarks, b)
        cl = get_landmark_coords(landmarks, c)
        angles[angle] = calculate_angle(al,bl,cl)
    
    al = get_landmark_coords(landmarks, pl.LEFT_KNEE)
    cl = get_landmark_coords(landmarks,pl.RIGHT_KNEE)
    
    left_hip = get_landmark_coords(landmarks,pl.LEFT_HIP)
    right_hip = get_landmark_coords(landmarks, pl.RIGHT_HIP)
    
    if left_hip is None or right_hip is None :
        bl =[None]*3
    else :
        bl = [(left_hip[i]+right_hip[i])/2 for i in range(3)]
    
    angles['leg_spread_angle'] = calculate_angle(al,bl,cl)
    
    return angles

# 누락된 값 보간하는 함수
def fill_nans(arr) :
    nans = np.isnan(arr)
    
    if not np.any(nans) :
        return arr
    
    x_coords = np.arange(len(arr))
    y_coords = arr[~nans]
    x_coords_nonan = x_coords[~nans]
    
    return np.interp(x_coords, x_coords_nonan, y_coords)

# 모든 누락된 값 보간하는 함수
def impute_pose_data(array, num_landmarks=33) :
    
    imputed_array = np.copy(array)
    
    for lm_id in range(num_landmarks) :
        col_x_idx = lm_id*4
        imputed_array[:, col_x_idx] = fill_nans(imputed_array[:, col_x_idx])
        
        col_y_idx = lm_id*4+1
        imputed_array[:,col_y_idx] = fill_nans(imputed_array[:,col_y_idx])
        
        col_z_idx = lm_id*4+2
        imputed_array[:,col_z_idx] = fill_nans(imputed_array[:,col_z_idx])
    
    angle_start_col_idx = num_landmarks*4
    
    for col_idx in range(angle_start_col_idx, imputed_array.shape[1]):
        imputed_array[:,col_idx] = fill_nans(imputed_array[:,col_idx])
        
    return imputed_array