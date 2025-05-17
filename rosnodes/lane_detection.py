import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection_node")
        self.publisher = self.create_publisher(Image, "lane_detection/image", 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture("project_video.mp4")  # Change this as needed
        self.timer = self.create_timer(0.05, self.timer_callback)  # ~20 FPS
        self.get_logger().info("Lane Detection Node started.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video stream.")
            self.cap.release()
            self.destroy_node()
            return

        processed = self.process_frame(frame)
        msg = self.bridge.cv2_to_imgmsg(processed, encoding="bgr8")
        self.publisher.publish(msg)

        # Optional display
        cv2.imshow("Lane Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            self.destroy_node()
            cv2.destroyAllWindows()

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def canny_edge(self, image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        mask = np.zeros_like(image)
        region = np.array([[
            (width * 0.1, height),
            (width * 0.45, height * 0.6),
            (width * 0.55, height * 0.6),
            (width * 0.9, height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, region, 255)
        return cv2.bitwise_and(image, mask)

    def hough_lines(self, image):
        return cv2.HoughLinesP(image, 1, np.pi/180, 50, minLineLength=100, maxLineGap=160)

    def draw_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        return cv2.addWeighted(image, 0.8, line_image, 1, 0)

    def process_frame(self, frame):
        gray = self.grayscale(frame)
        blurred = self.gaussian_blur(gray)
        edges = self.canny_edge(blurred)
        roi = self.region_of_interest(edges)
        lines = self.hough_lines(roi)
        return self.draw_lines(frame, lines)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
