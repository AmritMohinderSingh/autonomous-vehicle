import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
import torch
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp


class PedestrianTrackerNode(Node):
    def __init__(self):
        super().__init__("pedestrian_tracker_node")

        # Initialize model and tools
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt").to(self.device)
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose()

        # Publisher for tracking data
        self.tracking_pub = self.create_publisher(String, "pedestrian_tracking_data", 10)

        # Open video file
        self.cap = cv2.VideoCapture(r"C:\Users\madha\Videos\pedestrians.mp4")
        self.timer = self.create_timer(0.05, self.process_frame)  # ~20 FPS

        self.get_logger().info("Pedestrian Tracker Node Started")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video reached.")
            self.cap.release()
            self.destroy_node()
            return

        results = self.model(frame)[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0 and score > 0.4:  # person class
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

        tracked_objects = self.tracker.update_tracks(detections, frame=frame)

        data_to_publish = []

        for track in tracked_objects:
            if track.is_confirmed():
                bbox = track.to_ltrb()
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, bbox)
                person_roi = frame[y1:y2, x1:x2]
                person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                results_pose = self.pose.process(person_roi_rgb)

                key_points = {}
                if results_pose.pose_landmarks:
                    for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                        key_x = int(landmark.x * (x2 - x1)) + x1
                        key_y = int(landmark.y * (y2 - y1)) + y1
                        key_points[idx] = [key_x, key_y]

                track_data = {
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "keypoints": key_points
                }

                data_to_publish.append(track_data)

                # Optional visualization (comment out if headless)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for _, (kx, ky) in key_points.items():
                    cv2.circle(frame, (kx, ky), 3, (0, 0, 255), -1)

        # Publish JSON data as String
        json_msg = String()
        json_msg.data = json.dumps(data_to_publish)
        self.tracking_pub.publish(json_msg)

        # Optional display
        cv2.imshow("Pedestrian Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cap.release()
            self.destroy_node()
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianTrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
