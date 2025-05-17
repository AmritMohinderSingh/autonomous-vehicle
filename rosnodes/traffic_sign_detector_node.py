
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np
import time
import random

class TrafficSignDetector(Node):
    def __init__(self):
        super().__init__('traffic_sign_detector')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/detected_signs/image', 10)
        self.bridge = CvBridge()
        self.model = YOLO('/path/to/best.pt')  # Replace with actual path
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust for your environment
        self.category_images = {
            "Amber Light": "images/amber_light.jpg",
            "Green Light": "images/green_light.png",
            "Red Light": "images/red_light.png",
            "Speed Limit 30": "images/speed_30.png",
            "Speed Limit 60": "images/speed_60.png",
            "Speed Limit 80": "images/speed_80.png",
        }
        self.detection_list = []

    def perform_ocr(self, cropped_img):
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        extracted_text = pytesseract.image_to_string(thresh, config='--psm 6')
        extracted_digits = ''.join(filter(str.isdigit, extracted_text))
        return extracted_digits

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, conf=0.3)

        for r in results:
            for box in r.boxes:
                cls = self.model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_sign = frame[y1:y2, x1:x2]

                if "Speed Limit" in cls:
                    ocr_result = self.perform_ocr(cropped_sign)
                    if ocr_result in ["30", "60", "80"]:
                        cls = f"Speed Limit {ocr_result}"

                lateral_distance = round(random.uniform(1, 5), 2)
                longitudinal_distance = round(random.uniform(5, 30), 2)
                self.detection_list.insert(0, {
                    "class": cls,
                    "lateral": lateral_distance,
                    "longitudinal": longitudinal_distance,
                    "time": time.time()
                })

        if len(self.detection_list) > 50:
            self.detection_list.pop()

        annotated_frame = results[0].plot()
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.publisher.publish(out_msg)
        self.get_logger().info(f"Published detection with {len(results[0].boxes)} boxes.")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
