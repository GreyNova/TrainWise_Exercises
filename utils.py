import cv2
import mediapipe as mp
import numpy as np

def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)


    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)

    return img




def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame


def draw_text(
    img,
    msg,
    width = 8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
    box_offset=(20, 10),
):

    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))
    
    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)


    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)), 
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    
    return text_size




def find_angle(p1, p2, ref_pt = np.array([0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
    degree = int(180 / np.pi) * theta

    return int(degree)





def get_landmark_array(pose_landmark, key, frame_width, frame_height):

    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])




def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord   = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord   = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord   = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord   = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord   = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    
    else:
       raise ValueError("feature needs to be either 'nose', 'left' or 'right")



# Import MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys

class LandmarkListShim:
    def __init__(self, landmarks):
        self.landmark = landmarks

class ResultShim:
    def __init__(self, detection_result):
        if detection_result.pose_landmarks:
            self.pose_landmarks = LandmarkListShim(detection_result.pose_landmarks[0])
        else:
            self.pose_landmarks = None

class PoseShim:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        # Determine absolute path to model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'pose_landmarker_full.task')
        
        if not os.path.exists(model_path):
             print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        running_mode = vision.RunningMode.VIDEO
        if static_image_mode:
            running_mode = vision.RunningMode.IMAGE
            
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1
        )
        try:
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.running_mode = running_mode
        except Exception as e:
            print(f"Error creating PoseLandmarker: {e}", file=sys.stderr)
            raise e

        self.timestamp_ms = 0

    def process(self, image):
        try:
            # Convert cv2 image (numpy) to mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # In VIDEO mode, must provide timestamp
            self.timestamp_ms += 33 # approx 30fps increment
            
            # Use detect_for_video
            if self.running_mode == vision.RunningMode.VIDEO:
                 result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
            else:
                 result = self.landmarker.detect(mp_image)
                 
            return ResultShim(result)
        except Exception as e:
            print(f"Error in PoseShim.process: {e}", file=sys.stderr)
            class EmptyResult:
                pose_landmarks = None
            return EmptyResult()

def get_mediapipe_pose(
                        static_image_mode = False, 
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5
                      ):
    pose = PoseShim(
                    static_image_mode = static_image_mode,
                    model_complexity = model_complexity,
                    smooth_landmarks = smooth_landmarks,
                    min_detection_confidence = min_detection_confidence,
                    min_tracking_confidence = min_tracking_confidence
                   )
    return pose