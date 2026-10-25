import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2  # 追加

class YoloTracker(Node):
    def __init__(self):
        super().__init__('yolo_tracker')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8s.pt')
        self.target_class = 'sports ball' 

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image, verbose=False)
        
        # YOLOの認識結果を画面に表示する処理を追加
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)
        cv2.waitKey(1)

        twist = Twist()
        target_detected = False

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if self.model.names[int(cls)] == self.target_class:
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    error_x = (cv_image.shape[1] / 2) - float(center_x)
                    twist.angular.z = error_x * 0.002
                    twist.linear.x = 0.15
                    target_detected = True
                    break
            if target_detected:
                break

        if not target_detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.3 

        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = YoloTracker()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
