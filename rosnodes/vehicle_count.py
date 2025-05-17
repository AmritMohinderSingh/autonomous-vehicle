import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

import cv2
import torch
from ultralytics import YOLO


class VehicleCounterNode(Node):
    def __init__(self):
        super().__init__("vehicle_counter_node")

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load YOLOv8 model
        self.model = YOLO("yolov8n.pt").to(self.device)

        # Publisher for vehicle count
        self.vehicle_count_pub = self.create_publisher(Int32, "vehicle_count", 10)

        # Initialize video capture
        self.cap = cv2.VideoCapture("traffic1.mp4")
        self.timer = self.create_timer(0.05, self.process_frame)  # ~20 FPS

        # Frame resize dimensions
        self.frame_width = 640
        self.frame_height = 480

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video or cannot read frame.")
            self.cap.release()
            self.destroy_node()
            return

        frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        results = self.model(frame, device=self.device)
        vehicle_count = len(results[0].boxes)

        # Publish vehicle count
        msg = Int32()
        msg.data = vehicle_count
        self.vehicle_count_pub.publish(msg)

        # Optional display (useful for debugging)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Traffic Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cap.release()
            self.destroy_node()
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = VehicleCounterNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
