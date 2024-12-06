import cv2
import mediapipe as mp
import os

# Anthropometric segment weights and COM locations (Winter's table)
SEGMENT_WEIGHTS = {
    "head": 0.081,
    "torso": 0.497,
    "upper_arm": 0.027,
    "forearm": 0.016,
    "hand": 0.006,
    "thigh": 0.105,
    "shank": 0.047,
    "foot": 0.014,
}
SEGMENT_COMS = {
    "head": 0.5,
    "torso": 0.5,
    "upper_arm": 0.436,
    "forearm": 0.430,
    "hand": 0.506,
    "thigh": 0.433,
    "shank": 0.433,
    "foot": 0.5,
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

def process_video(video_path):
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = os.path.splitext(video_path)[0] + "_COM.mp4"
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Define keypoints of interest
            keypoints = {
                "head": landmarks[mp_pose.PoseLandmark.NOSE],
                "torso": {
                    "x": (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
                    "y": (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2,
                    "visibility": min(landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility),
                },
                "left_upper_arm": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                "left_forearm": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                "left_hand": landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
                "left_thigh": landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                "left_shank": landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                "left_foot": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
                "right_upper_arm": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                "right_forearm": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                "right_hand": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
                "right_thigh": landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                "right_shank": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                "right_foot": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
            }

            # Calculate COM
            total_weight = 0
            com_x, com_y = 0, 0

            for segment, keypoint in keypoints.items():
                if isinstance(keypoint, dict):  # For computed landmarks like "torso"
                    x = keypoint["x"] * frame.shape[1]
                    y = keypoint["y"] * frame.shape[0]
                    visibility = keypoint["visibility"]
                else:  # For regular landmarks
                    x = keypoint.x * frame.shape[1]
                    y = keypoint.y * frame.shape[0]
                    visibility = keypoint.visibility

                if visibility < 0.5:  # Skip low-visibility keypoints
                    continue

                weight = SEGMENT_WEIGHTS.get(segment.split('_')[0], 0) / 2  # Divide by 2 for left/right
                com_x += weight * x
                com_y += weight * y
                total_weight += weight

            if total_weight > 0:
                com_x /= total_weight
                com_y /= total_weight

                # Draw COM on the frame
                com_position = (int(com_x), int(com_y))
                cv2.circle(frame, com_position, 1, (0, 255, 0), -1)  # Green circle for COM
                cv2.putText(frame, "COM", (int(com_x) + 10, int(com_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Write the annotated frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved at {output_path}")

# Example usage
video_path = "static/videos/CMJ2.mp4"  # Replace with your video path
process_video(video_path)
