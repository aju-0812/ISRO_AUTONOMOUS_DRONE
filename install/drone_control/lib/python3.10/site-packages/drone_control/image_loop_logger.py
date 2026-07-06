#!/usr/bin/env python3
"""
image_loop_logger.py
────────────────────
ROS 2 Node that captures downward webcam frames every 3 seconds in a loop.
Files are saved in ~/drone_survey_dump/ named explicitly as (x,y,z).jpg
using real-time MAVROS position matrix feeds.
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

import cv2
import numpy as np


class ImageLoopLogger(Node):

    def __init__(self):
        super().__init__('image_loop_logger')
        self.get_logger().info("📸 Drone Image Loop Logger Node Initialized!")

        # ── Setup Directory Storage ───────────────────────────
        # Automatically resolves to /home/ajendra/drone_survey_dump
        self.target_dir = os.path.expanduser('~/drone_survey_dump')
        os.makedirs(self.target_dir, exist_ok=True)
        self.get_logger().info(f"📁 Target storage folder set to: {self.target_dir}")

        # ── QoS Configuration Profile Mapping ─────────────────
        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── State Coordinate Registers ────────────────────────
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 0.0
        self.pose_received = False
        self.latest_frame = None

        # ── Subscribers ───────────────────────────────────────
        # Track position coordinates from MAVROS
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            telemetry_qos
        )

        # Track downward camera images from the webcam topic layout
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            cam_qos
        )

        # ── 3-Second Loop Timer Execution Block ───────────────
        self.timer_period = 3.0  # seconds
        self.capture_timer = self.create_timer(self.timer_period, self.capture_loop_callback)

    def pose_callback(self, msg: PoseStamped):
        self.cx = msg.pose.position.x
        self.cy = msg.pose.position.y
        self.cz = msg.pose.position.z
        self.pose_received = True

    def image_callback(self, msg: Image):
        self.latest_frame = msg

    def capture_loop_callback(self):
        # 1. Protection safety gates
        if self.latest_frame is None:
            self.get_logger().warn("⏳ Waiting for incoming video frames on /image_raw...")
            return

        if not self.pose_received:
            self.get_logger().warn("⏳ Waiting for valid telemetry vectors on /mavros/local_position/pose...")
            return

        # 2. Parse ROS Image Message frame array into native OpenCV matrix
        msg = self.latest_frame
        try:
            # Determine channels (grayscale mono vs raw color BGR packing arrays)
            channels = 1 if msg.encoding in ('mono8', '8UC1') else 3
            raw_data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            
            if channels == 1:
                img_matrix = raw_data.reshape(msg.height, msg.width)
                bgr_frame = cv2.cvtColor(img_matrix, cv2.COLOR_GRAY2BGR)
            else:
                img_matrix = raw_data.reshape(msg.height, msg.width, 3)
                # Convert standard RGB stream to native OpenCV storage layout (BGR)
                bgr_frame = cv2.cvtColor(img_matrix, cv2.COLOR_RGB2BGR)

            # 3. Format coordinates and build file output string string name
            # Replaces negative symbols with clear strings for clean parsing if coordinate drifts negative
            def format_coord(val):
                return f"n{abs(val):.2f}" if val < 0 else f"{val:.2f}"

            file_name = f"({format_coord(self.cx)},{format_coord(self.cy)},{format_coord(self.cz)}).jpg"
            full_write_path = os.path.join(self.target_dir, file_name)

            # 4. Write image file array directly to user home disk workspace
            cv2.imwrite(full_write_path, bgr_frame)
            self.get_logger().info(f"📸 Snapshot logged -> Name: {file_name}")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to parse or save camera frame snapshot: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageLoopLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Image capture node stopped manually via keyboard interrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
