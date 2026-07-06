#!/usr/bin/env python3
"""
navigation_manager_mavros.py
─────────────────────────────
Waypoint navigation manager for a REAL drone using MAVROS.

Topics used:
  SUB  /mavros/local_position/odom   — current position (ENU)
  SUB  /mavros/battery               — battery state
  SUB  /mavros/state                 — FCU state (mode, connected)
  SUB  /image_raw                    — downward camera (captured at each WP)
  PUB  /drone_target                 — consumed by flight_manager_mavros

Coverage area : 7.62 m (X/East) × 10.66 m (Y/North)
Step spacing  : 1.5 m × 1.5 m
Flight altitude: 1.5 m AGL (home-relative)
Pattern       : Boustrophedon (snake) sweep
Hover per WP  : 5 s  (image captured at hover mid-point)

Failsafes
  Battery ≤ 25 %   → RTH
  FCU connection lost → RTH
  Offboard mode lost (after mission start) → HOLD
"""

import math
import os
import enum

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, Image
from mavros_msgs.msg import State

# ── Mission tuning ────────────────────────────────────────────
AREA_X         = 7.62   # m  East extent
AREA_Y         = 10.66  # m  North extent
STEP_X         = 1.5    # m  column spacing
STEP_Y         = 1.5    # m  row spacing
FLIGHT_ALT     = 2.0    # m  altitude above home

HOVER_SECONDS      = 5.0   # s    dwell at each waypoint
REACHED_THRESHOLD  = 0.3   # m    3-D distance → WP reached
TIMER_DT           = 0.1   # s    navigation loop period (10 Hz)
BATTERY_RTH_PCT    = 25.0  # %    threshold for RTH

HOVER_TICKS = int(HOVER_SECONDS / TIMER_DT)

# ── Camera ────────────────────────────────────────────────────
IMAGE_TOPIC = '/image_raw'
SAVE_DIR    = os.path.expanduser('~/survey_images')


# ── Mission state machine ─────────────────────────────────────
class State_(enum.Enum):
    WAIT_HOME          = 0
    GENERATE_WAYPOINTS = 1
    FLY_WP             = 2
    HOVER_WP           = 3
    RETURN_HOME        = 4
    HOLD               = 5


class NavigationManagerMavros(Node):

    def __init__(self):
        super().__init__('navigation_manager_mavros')
        self.get_logger().info(
            f"🧭 Navigation Manager (MAVROS) started  "
            f"area={AREA_X}×{AREA_Y} m  step={STEP_X}×{STEP_Y} m  "
            f"alt={FLIGHT_ALT} m  hover={HOVER_SECONDS} s"
        )

        # ── Save dir ──────────────────────────────────────────
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.get_logger().info(f"📁 Survey images → {SAVE_DIR}")

        # ── QoS profiles ──────────────────────────────────────
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Publisher ─────────────────────────────────────────
        self.target_pub = self.create_publisher(
            PoseStamped, '/drone_target', reliable_qos
        )

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            best_effort_qos
        )
        self.create_subscription(
            BatteryState,
            '/mavros/battery',
            self.battery_callback,
            best_effort_qos
        )
        self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            best_effort_qos
        )
        self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            cam_qos
        )

        # ── Current position (ENU) ────────────────────────────
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 0.0

        # ── Home ──────────────────────────────────────────────
        self.home_captured = False
        self.home_x        = 0.0
        self.home_y        = 0.0
        self.home_z        = 0.0

        # ── Failsafe state ────────────────────────────────────
        self.battery_pct  = 100.0
        self.fcu_connected = True
        self.mission_started = False   # set True once first WP published

        # ── Waypoints & mission ───────────────────────────────
        self.waypoints       = []
        self.current_waypoint = 0
        self.hover_count     = 0

        # ── Camera state ──────────────────────────────────────
        self.latest_image   = None   # most recent cached frame
        self.image_captured = False  # per-WP flag, reset each time we start hovering

        # ── State machine ─────────────────────────────────────
        self.state = State_.WAIT_HOME

        # ── Navigation timer ──────────────────────────────────
        self.create_timer(TIMER_DT, self.navigation_loop)

    # ──────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.cx = p.x
        self.cy = p.y
        self.cz = p.z

        if not self.home_captured:
            self.home_x = self.cx
            self.home_y = self.cy
            self.home_z = self.cz
            self.home_captured = True
            self.get_logger().info(
                f"🏠 Home captured  "
                f"({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f})"
            )
            self.state = State_.GENERATE_WAYPOINTS

    def battery_callback(self, msg: BatteryState):
        self.battery_pct = msg.percentage * 100.0
        if self.battery_pct <= BATTERY_RTH_PCT:
            if self.state not in (State_.RETURN_HOME, State_.HOLD):
                self.get_logger().warn(
                    f"⚠️  Battery {self.battery_pct:.1f}% ≤ {BATTERY_RTH_PCT}% → RTH"
                )
                self._trigger_rth()

    def state_callback(self, msg: State):
        self.fcu_connected = msg.connected

        # FCU connection lost → RTH
        if not msg.connected:
            if self.state not in (State_.RETURN_HOME, State_.HOLD):
                self.get_logger().error("📡 FCU connection lost → RTH")
                self._trigger_rth()
            return

        # Offboard lost after mission has started → HOLD
        if self.mission_started and msg.mode != "OFFBOARD":
            if self.state not in (State_.HOLD,):
                self.get_logger().error(
                    f"⚠️  Offboard mode lost (mode={msg.mode}) → HOLD"
                )
                self.state = State_.HOLD

    def image_callback(self, msg: Image):
        """Cache latest frame — saved only at hover mid-point."""
        self.latest_image = msg

    # ──────────────────────────────────────────────────────────
    # Image save
    # ──────────────────────────────────────────────────────────

    def _save_image(self):
        """Save cached frame with filename (x,y,z).jpg using capture-time pose."""
        if self.latest_image is None:
            self.get_logger().warn("📷 No frame cached yet — skipping")
            return

        msg = self.latest_image
        try:
            if msg.encoding in ('rgb8', 'bgr8'):
                channels = 3
            elif msg.encoding == 'mono8':
                channels = 1
            else:
                self.get_logger().warn(
                    f"📷 Unsupported encoding '{msg.encoding}' — skipping"
                )
                return

            img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                msg.height, msg.width, channels)

            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # bgr8 / mono8 need no conversion for cv2.imwrite

            # Format: (x,y,z).jpg  — negative values use 'n' prefix
            def fmt(v):
                return f"n{abs(v):.3f}" if v < 0 else f"{v:.3f}"

            fname = f"({fmt(self.cx)},{fmt(self.cy)},{fmt(self.cz)}).jpg"
            path  = os.path.join(SAVE_DIR, fname)
            cv2.imwrite(path, img)
            self.get_logger().info(f"📸 Saved: {fname}")

        except Exception as e:
            self.get_logger().error(f"📷 Save failed: {e}")

    # ──────────────────────────────────────────────────────────
    # Waypoint generation
    # ──────────────────────────────────────────────────────────

    def generate_waypoints(self):
        """Generate boustrophedon (snake) coverage waypoints."""
        self.waypoints = []
        x         = 0.0
        direction = 1          # +1 = North, -1 = South

        while x <= AREA_X + 0.01:
            if direction == 1:
                y = 0.0
                while y <= AREA_Y + 0.01:
                    self.waypoints.append([
                        self.home_x + x,
                        self.home_y + y,
                        FLIGHT_ALT
                    ])
                    y += STEP_Y
            else:
                # Start at last valid y on this strip
                y = 0.0
                col_pts = []
                while y <= AREA_Y + 0.01:
                    col_pts.append([
                        self.home_x + x,
                        self.home_y + y,
                        FLIGHT_ALT
                    ])
                    y += STEP_Y
                self.waypoints.extend(reversed(col_pts))

            x         += STEP_X
            direction *= -1

        self.get_logger().info(
            f"✅ Generated {len(self.waypoints)} coverage waypoints"
        )

    # ──────────────────────────────────────────────────────────
    # Navigation loop  (10 Hz)
    # ──────────────────────────────────────────────────────────

    def navigation_loop(self):

        # ── WAIT_HOME ─────────────────────────────────────────
        if self.state == State_.WAIT_HOME:
            # Nothing to do — waiting for first odom message
            return

        # ── GENERATE_WAYPOINTS ────────────────────────────────
        if self.state == State_.GENERATE_WAYPOINTS:
            self.generate_waypoints()
            self.current_waypoint = 0
            self.hover_count      = 0
            self.state            = State_.FLY_WP
            self.mission_started  = True
            return

        # ── HOLD ──────────────────────────────────────────────
        if self.state == State_.HOLD:
            self._publish_target(self.home_x, self.home_y, FLIGHT_ALT)
            return

        # ── RETURN_HOME ───────────────────────────────────────
        if self.state == State_.RETURN_HOME:
            self._publish_target(self.home_x, self.home_y, FLIGHT_ALT)
            dist = math.sqrt(
                (self.cx - self.home_x) ** 2 +
                (self.cy - self.home_y) ** 2
            )
            self.get_logger().info(
                f"🏠 Returning home  dist={dist:.2f} m",
                throttle_duration_sec=1.0
            )
            if dist < REACHED_THRESHOLD:
                self.get_logger().info("🏠 Home reached → HOLD")
                self.state = State_.HOLD
            return

        # ── Coverage mission (FLY_WP / HOVER_WP) ─────────────
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info(
                "✅ All coverage waypoints complete → Return Home"
            )
            self.state = State_.RETURN_HOME
            return

        wp = self.waypoints[self.current_waypoint]

        # ── HOVER_WP ──────────────────────────────────────────
        if self.state == State_.HOVER_WP:
            self._publish_target(wp[0], wp[1], wp[2])
            self.hover_count += 1
            remaining = (HOVER_TICKS - self.hover_count) * TIMER_DT

            # Capture once at hover mid-point (drone stable)
            if self.hover_count == HOVER_TICKS // 2 and not self.image_captured:
                self._save_image()
                self.image_captured = True

            self.get_logger().info(
                f"⏳ WP {self.current_waypoint}/{len(self.waypoints)-1}  "
                f"hover {remaining:.1f} s remaining",
                throttle_duration_sec=1.0
            )
            if self.hover_count >= HOVER_TICKS:
                self.get_logger().info(
                    f"➡️  Hover done → WP {self.current_waypoint + 1}"
                )
                self.hover_count      = 0
                self.current_waypoint += 1
                self.state            = State_.FLY_WP
            return

        # ── FLY_WP ────────────────────────────────────────────
        if self.state == State_.FLY_WP:
            distance = math.sqrt(
                (wp[0] - self.cx) ** 2 +
                (wp[1] - self.cy) ** 2 +
                (wp[2] - self.cz) ** 2
            )
            self._publish_target(wp[0], wp[1], wp[2])
            self.get_logger().info(
                f"🎯 WP {self.current_waypoint}/{len(self.waypoints)-1}  "
                f"target=({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f})  "
                f"pos=({self.cx:.1f},{self.cy:.1f},{self.cz:.1f})  "
                f"dist={distance:.2f} m",
                throttle_duration_sec=0.5
            )
            if distance < REACHED_THRESHOLD:
                self.get_logger().info(
                    f"📍 WP {self.current_waypoint} reached → hovering {HOVER_SECONDS} s"
                )
                self.hover_count    = 0
                self.image_captured = False
                self.state          = State_.HOVER_WP
            return

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _publish_target(self, x: float, y: float, z: float):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        self.target_pub.publish(msg)

    def _trigger_rth(self):
        """Jump directly to Return-Home state."""
        self.state = State_.RETURN_HOME


# ── Entry point ───────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = NavigationManagerMavros()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Navigation Manager shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
