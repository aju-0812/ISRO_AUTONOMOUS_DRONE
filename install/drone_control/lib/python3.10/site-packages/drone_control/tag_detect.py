#!/usr/bin/env python3
"""
apriltag_detector_node_sim.py
────────────────────────────
Standalone AprilTag detector node — SIM VERSION (Gazebo / PX4 SITL).

CHANGES vs apriltag_detector_node.py (real-drone version):

  1. IMAGE_TOPIC now points at the Gazebo x500_depth mono camera:
         /world/forest/model/x500_depth_0/model/mono_cam/link/camera_link/sensor/camera/image
     (matches your sim world/model naming — change WORLD_NAME / MODEL_NAME
     below if you switch worlds or vehicles, instead of hand-editing the
     full topic string).

  2. NEW: camera intrinsics are no longer hardcoded placeholders. The real
     file's FX/FY/CX/CY were guesses meant to be replaced by real-camera
     calibration; in sim, Gazebo publishes a CameraInfo topic alongside the
     image with the *exact* intrinsics the sensor plugin is using. This
     node now subscribes to that (auto-derived by swapping 'image' ->
     'camera_info' in IMAGE_TOPIC) and uses it once received, instead of
     guessing. Detection is held off until the first CameraInfo arrives so
     a wrong-FOV guess never silently produces a bad tag_size/pose.

  3. NEW: image_callback also accepts 'rgb8' encoding directly (Gazebo's
     ros_gz_bridge commonly publishes rgb8 for mono/color sim cameras,
     unlike real USB cams which are more often mono8/bgr8 already). Falls
     back through mono8 -> rgb8 -> bgr8.

  Everything else (tag family, pupil_apriltags pose estimation, /tag_pose
  + /tag_detector_alive contract, heartbeat-independent-of-frames design)
  is UNCHANGED from the real-drone version — flight_manager doesn't care
  which detector node is feeding it, the contract is identical.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import cv2
from pupil_apriltags import Detector


# ── EDIT THESE for your sim world / model ──────────────────────────
WORLD_NAME    = 'forest'
MODEL_NAME    = 'x500_depth_0'
CAMERA_PATH   = 'model/mono_cam/link/camera_link/sensor/camera'

IMAGE_TOPIC   = f'/world/{WORLD_NAME}/model/{MODEL_NAME}/{CAMERA_PATH}/image'
# CameraInfo is published by the same sim camera sensor, same path, just
# 'camera_info' instead of 'image' -- this is the gz sim convention.
CAMERA_INFO_TOPIC = f'/world/{WORLD_NAME}/model/{MODEL_NAME}/{CAMERA_PATH}/camera_info'

TAG_FAMILY    = 'tag36h11'
TARGET_TAG_ID = 38         # must match TARGET_TAG_ID in flight_manager
TAG_SIZE      = 0.40       # metres, edge length of your printed/sim marker
                            # (matches vision_smart_land.py's confirmed-working value)

HEARTBEAT_HZ     = 2.0
DETECT_EVERY_N   = 1
# ─────────────────────────────────────────────────────────────────


class AprilTagDetectorNodeSim(Node):

    def __init__(self):
        super().__init__('apriltag_detector_node_sim')
        self.get_logger().info(
            f"AprilTag detector (SIM) online — watching '{IMAGE_TOPIC}' "
            f"for tag id={TARGET_TAG_ID}, waiting for CameraInfo on "
            f"'{CAMERA_INFO_TOPIC}' before detecting")

        self.bridge = CvBridge()
        self.frame_count = 0

        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
        )

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── NEW: real intrinsics from sim CameraInfo, not guessed ──────
        self.fx = self.fy = self.cx_intr = self.cy_intr = None
        self.camera_info_received = False

        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, best_effort_qos)
        # NOTE: image subscription uses reliable_qos here, not best_effort_qos --
        # matches the ros_gz_bridge publisher's actual QoS (RELIABLE,
        # KEEP_LAST(10), confirmed via `ros2 topic info -v`) and matches
        # vision_smart_land.py's confirmed-working subscription.
        self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, reliable_qos)

        self.pose_pub  = self.create_publisher(PoseStamped, '/tag_pose', reliable_qos)
        self.alive_pub = self.create_publisher(Bool, '/tag_detector_alive', reliable_qos)

        # Heartbeat independent of image/camera_info arrival -- same reasoning
        # as the real node: proves the *node* (not just frames) is alive.
        self.create_timer(1.0 / HEARTBEAT_HZ, self.heartbeat_callback)

        self.last_detection_log = 0.0
        self.last_no_intrinsics_warn = 0.0

    def heartbeat_callback(self):
        msg = Bool()
        msg.data = True
        self.alive_pub.publish(msg)

    def camera_info_callback(self, msg: CameraInfo):
        # msg.k is the 3x3 intrinsic matrix, row-major: [fx 0 cx, 0 fy cy, 0 0 1]
        if not self.camera_info_received:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx_intr, self.cy_intr = msg.k[2], msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f"📐 CameraInfo received -> fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx_intr:.1f} cy={self.cy_intr:.1f} (using real sim intrinsics)")

    def image_callback(self, msg: Image):
        if not self.camera_info_received:
            now = time.time()
            if now - self.last_no_intrinsics_warn > 2.0:
                self.last_no_intrinsics_warn = now
                self.get_logger().warn(
                    f"⏳ No CameraInfo yet on '{CAMERA_INFO_TOPIC}' — skipping "
                    f"detection until intrinsics arrive (check topic name/sim is running)")
            return

        self.frame_count += 1
        if DETECT_EVERY_N > 1 and (self.frame_count % DETECT_EVERY_N) != 0:
            return

        frame = None
        for enc, convert in (
            ('mono8', None),
            ('rgb8', cv2.COLOR_RGB2GRAY),
            ('bgr8', cv2.COLOR_BGR2GRAY),
        ):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=enc)
                frame = cv2.cvtColor(img, convert) if convert is not None else img
                break
            except Exception:
                continue

        if frame is None:
            self.get_logger().warn("image convert failed (tried mono8/rgb8/bgr8)")
            return

        detections = self.detector.detect(
            frame,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx_intr, self.cy_intr),
            tag_size=TAG_SIZE,
        )

        # DEBUG: log every detection seen, regardless of target ID, so we
        # can tell "no tag visible at all" apart from "tag visible but
        # wrong id/family" apart from "marker isn't a tag36h11 AprilTag".
        if detections:
            seen_ids = [d.tag_id for d in detections]
            if seen_ids != getattr(self, '_last_seen_ids', None):
                self._last_seen_ids = seen_ids
                self.get_logger().info(
                    f"🔍 DEBUG: {len(detections)} tag(s) detected this frame, "
                    f"ids={seen_ids} (looking for id={TARGET_TAG_ID})")
        else:
            now_dbg = time.time()
            if now_dbg - getattr(self, '_last_zero_log', 0) > 3.0:
                self._last_zero_log = now_dbg
                self.get_logger().info(
                    "🔍 DEBUG: 0 tags detected this frame (no tag36h11 pattern "
                    "found in image at all)")

        tag = None
        for d in detections:
            if d.tag_id == TARGET_TAG_ID:
                tag = d
                break

        if tag is None:
            return  # silence = no tag, flight_manager already handles this

        t = tag.pose_t.flatten()

        vp = PoseStamped()
        vp.header.stamp = self.get_clock().now().to_msg()
        vp.header.frame_id = 'camera_optical_frame'
        vp.pose.position.x = float(t[0])
        vp.pose.position.y = float(t[1])
        vp.pose.position.z = float(t[2])
        # orientation left as identity -- flight_manager only uses position
        self.pose_pub.publish(vp)

        now = time.time()
        if now - self.last_detection_log > 2.0:
            self.last_detection_log = now
            self.get_logger().info(
                f"🏷️ tag {TARGET_TAG_ID} @ cam-frame "
                f"x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNodeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("AprilTag detector (sim) shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
