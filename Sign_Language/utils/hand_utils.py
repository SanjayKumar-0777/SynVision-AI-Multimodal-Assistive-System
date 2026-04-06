import numpy as np
import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):

    POSE_LEN = 33 * 4
    FACE_LEN = 34 * 3
    HAND_LEN = 21 * 3
    DIST_LEN = 9

    pose = np.zeros(POSE_LEN)
    face = np.zeros(FACE_LEN)
    lh = np.zeros(HAND_LEN)
    rh = np.zeros(HAND_LEN)
    lh_dist = np.zeros(DIST_LEN)
    rh_dist = np.zeros(DIST_LEN)

    # 1. Pose Landmarks
    if results.pose_landmarks:
        # Translation invariance: use nose (0) as base
        base_x = results.pose_landmarks.landmark[0].x
        base_y = results.pose_landmarks.landmark[0].y
        pose = np.array(
            [[res.x - base_x, res.y - base_y, res.z, res.visibility]
             for res in results.pose_landmarks.landmark]
        ).flatten()

    SELECTED_FACE_INDICES = [
        70,63,105,66,107,336,296,334,293,300,
        33,133,362,263,159,145,386,374,
        61,291,0,17,13,78,308,324,318,402,310,317,87,178,88,95
    ]

    # 2. Face Landmarks
    if results.face_landmarks:
        # Translation invariance: use face center/nose (1 typically) as base
        base_x = results.face_landmarks.landmark[1].x
        base_y = results.face_landmarks.landmark[1].y
        temp = []
        for idx in SELECTED_FACE_INDICES:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                temp.extend([lm.x - base_x, lm.y - base_y, lm.z])
            else:
                temp.extend([0, 0, 0])
        face = np.array(temp)

    # Helper function for Hands
    def process_hand(hand_landmarks):
        if not hand_landmarks:
            return np.zeros(HAND_LEN), np.zeros(DIST_LEN)

        # Translation Invariance: shift relative to wrist (landmark 0)
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z

        hand_points = np.array(
            [[res.x - base_x, res.y - base_y, res.z - base_z]
             for res in hand_landmarks.landmark]
        )

        l = hand_landmarks.landmark
        def d(i,j):
            return np.sqrt((l[i].x-l[j].x)**2 +
                           (l[i].y-l[j].y)**2 +
                           (l[i].z-l[j].z)**2)

        dists = np.array([
            d(0,4), d(0,8), d(0,12), d(0,16), d(0,20),
            d(4,8), d(8,12), d(12,16), d(16,20)
        ])

        # Scale Invariance: Divide by max distance
        max_dist = np.max(dists) if np.max(dists) > 1e-6 else 1.0
        
        hand_points = hand_points / max_dist
        dists = dists / max_dist

        return hand_points.flatten(), dists

    lh, lh_dist = process_hand(results.left_hand_landmarks)
    rh, rh_dist = process_hand(results.right_hand_landmarks)

    return np.concatenate([pose, face, lh, rh, lh_dist, rh_dist])


def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=
        mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=
        mp_drawing_styles.get_default_pose_landmarks_style()
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    return image

def detect_and_process_hand(frame):

    if holistic is None:
        return None, frame

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = holistic.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    annotated_frame = draw_styled_landmarks(image, results)

    keypoints = extract_keypoints(results)

    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return None, annotated_frame

    if results.left_hand_landmarks or results.right_hand_landmarks:
        return keypoints, annotated_frame
    else:
        return None, annotated_frame