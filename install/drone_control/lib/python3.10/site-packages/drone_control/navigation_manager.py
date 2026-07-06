#!/usr/bin/env python3
"""
navigation_manager_mavros.py
─────────────────────────────
Boustrophedon coverage navigation manager for MAVROS (real drone).

Arena layout
  East  (X / forward / drone heading) : 11.0 m
  South (Y / lateral)                 : 7.0 m
  Flight altitude                     : handled entirely by flight_manager_mavros

Sweep pattern
  Drone head faces East  → forward axis  = +X (ENU East)
  Side axis              = -Y (ENU South, i.e. -Y in ENU)

  Strips run East–West (forward axis).
  After each strip, step 2 m South and reverse.
  Step sizes: 2 m forward × 2 m side.

  Strip samples  :  x = 0, 2, 4, 6, 8, 10  (6 points → 6 columns)
  Side positions :  Δy = 0, −2, −4, −6      (4 side steps → 4 strips)
  Total WPs      :  6 × 4 = 24

ENU convention
  +X = East   +Y = North   +Z = Up
  "South" in the arena = −Y in ENU

Topics
  SUB  /mavros/local_position/odom   — position (ENU)
  SUB  /mavros/battery               — battery state
  SUB  /mavros/state                 — FCU state
  SUB  /image_raw                    — downward camera
  PUB  /drone_target                 — consumed by flight_manager_mavros

Failsafes
  Battery ≤ 25 %        → RTH
  FCU connection lost   → RTH
  Offboard mode lost    → HOLD
  WP timeout (30 s)     → skip WP and continue
"""

import math
import os
import enum
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, Image
from mavros_msgs.msg import State

# ══════════════════════════════════════════════════════════════
#  Mission parameters
# ══════════════════════════════════════════════════════════════

# Arena  (drone-frame: forward = East, side = South)
ARENA_EAST_M      = 10.0  # m  East extent   (+X ENU)
ARENA_SOUTH_M     = 7.00  # m  South extent  (−Y ENU)

# Step sizes
STEP_EAST_M       =  1.6  # m  spacing along East strips  (forward)
STEP_SOUTH_M      =  2.85 # m  spacing between strips     (side)

# NOTE: Altitude (Z) is NOT managed here — flight_manager_mavros owns it entirely.
#       This node passes home_z as the Z target so the flight manager's
#       altitude controller is the single authority on height.

# Waypoint control
HOVER_SECONDS      =  5.0  # s    dwell at each WP
REACHED_THRESHOLD  =  0.15 # m    3-D distance → WP reached  (tightened from 0.3)
TIMER_DT           =  0.1  # s    nav-loop period (10 Hz)
WP_TIMEOUT_SEC     = 30.0  # s    max time to reach a WP before skipping
BATTERY_RTH_PCT    = 25.0  # %    low-battery RTH threshold

# RTH: seconds to hover over home before transitioning to HOLD
# Gives the Flight Manager time to fully settle on the home position.
HOME_SETTLE_SECONDS = 4.0
HOME_SETTLE_TICKS   = int(HOME_SETTLE_SECONDS / TIMER_DT)  # 40 ticks @ 10 Hz

# EKF settle: ignore first N odom messages before locking home
ODOM_HOME_SETTLE_COUNT = 20   # ~2 s at 10 Hz

HOVER_TICKS = int(HOVER_SECONDS / TIMER_DT)

# Camera
IMAGE_TOPIC = '/image_raw'
SAVE_DIR    = os.path.expanduser('~/survey2_images')


# ══════════════════════════════════════════════════════════════
#  State machine
# ══════════════════════════════════════════════════════════════
class MissionState(enum.Enum):
    WAIT_HOME          = 0
    GENERATE_WAYPOINTS = 1
    FLY_WP             = 2
    HOVER_WP           = 3
    RETURN_HOME        = 4
    HOME_SETTLE        = 5   # NEW: hover over home for a few seconds before HOLD
    HOLD               = 6


# ══════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════
class NavigationManagerMavros(Node):

    def __init__(self):
        super().__init__('navigation_manager_mavros')

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.get_logger().info(
            f"\n"
            f"  🧭  Navigation Manager (MAVROS) started\n"
            f"  Arena  : East {ARENA_EAST_M} m  ×  South {ARENA_SOUTH_M} m\n"
            f"  Steps  : {STEP_EAST_M} m East  ×  {STEP_SOUTH_M} m South\n"
            f"  Alt    : managed by flight_manager_mavros\n"
            f"  Hover  : {HOVER_SECONDS} s / WP\n"
            f"  RTH settle : {HOME_SETTLE_SECONDS} s over home before HOLD\n"
            f"  Images : {SAVE_DIR}\n"
        )

        # ── QoS ───────────────────────────────────────────────
        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        rel_qos = QoSProfile(
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
            PoseStamped, '/drone_target', rel_qos
        )

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            Odometry, '/mavros/local_position/odom',
            self.odom_callback, be_qos
        )
        self.create_subscription(
            BatteryState, '/mavros/battery',
            self.battery_callback, be_qos
        )
        self.create_subscription(
            State, '/mavros/state',
            self.fcu_state_callback, be_qos
        )
        self.create_subscription(
            Image, IMAGE_TOPIC,
            self.image_callback, cam_qos
        )

        # ── Position (ENU) ────────────────────────────────────
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 0.0

        # ── Home ──────────────────────────────────────────────
        self.home_captured   = False
        self.home_x          = 0.0
        self.home_y          = 0.0
        self.home_z          = 0.0
        self._odom_count     = 0

        # ── Failsafe ──────────────────────────────────────────
        self.battery_pct     = 100.0
        self.fcu_connected   = True
        self.mission_started = False

        # ── Mission ───────────────────────────────────────────
        self.waypoints       = []        # list of (x, y, z) ENU tuples
        self.current_wp_idx  = 0
        self.hover_count     = 0
        self.wp_start_time   = None      # monotonic time when FLY_WP began

        # ── RTH settling ──────────────────────────────────────
        self.home_settle_count = 0       # ticks spent in HOME_SETTLE

        # ── Camera ────────────────────────────────────────────
        self.latest_image    = None
        self.image_captured  = False

        # ── State ─────────────────────────────────────────────
        self.state = MissionState.WAIT_HOME

        self.create_timer(TIMER_DT, self.navigation_loop)

    # ──────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.cx = p.x
        self.cy = p.y
        self.cz = p.z

        if self.home_captured:
            return

        # Wait for EKF to converge before locking home
        self._odom_count += 1
        if self._odom_count < ODOM_HOME_SETTLE_COUNT:
            return

        self.home_x = self.cx
        self.home_y = self.cy
        self.home_z = self.cz
        self.home_captured = True

        self.get_logger().info(
            f"🏠 Home locked (EKF settled)  "
            f"ENU=({self.home_x:.3f}, {self.home_y:.3f}, {self.home_z:.3f})"
        )
        self.state = MissionState.GENERATE_WAYPOINTS

    def battery_callback(self, msg: BatteryState):
        raw = msg.percentage
        # BatteryState.percentage: 0.0–1.0 per spec; guard against non-compliant FW
        self.battery_pct = raw * 100.0 if raw <= 1.0 else raw
        if self.battery_pct <= BATTERY_RTH_PCT:
            if self.state not in (MissionState.RETURN_HOME,
                                  MissionState.HOME_SETTLE,
                                  MissionState.HOLD):
                self.get_logger().warn(
                    f"⚠️  Battery {self.battery_pct:.1f}% ≤ {BATTERY_RTH_PCT}% → RTH"
                )
                self._trigger_rth()

    def fcu_state_callback(self, msg: State):
        self.fcu_connected = msg.connected

        if not msg.connected:
            if self.state not in (MissionState.RETURN_HOME,
                                  MissionState.HOME_SETTLE,
                                  MissionState.HOLD):
                self.get_logger().error("📡 FCU disconnected → RTH")
                self._trigger_rth()
            return

        if self.mission_started and msg.mode != "OFFBOARD":
            if self.state != MissionState.HOLD:
                self.get_logger().error(
                    f"⚠️  Offboard lost (mode={msg.mode}) → HOLD"
                )
                self.state = MissionState.HOLD

    def image_callback(self, msg: Image):
        self.latest_image = msg

    # ──────────────────────────────────────────────────────────
    # Waypoint generation
    # ──────────────────────────────────────────────────────────

    def _generate_waypoints(self):
        """
        Boustrophedon sweep — drone faces East.

        Forward axis : East  = +X ENU
        Side axis    : South = −Y ENU

        Altitude is owned by flight_manager_mavros.
        This node publishes home_z as the Z target so that the flight
        manager's altitude hold loop is the single source of truth for height.

        Strip layout (top-view, North up):

          Home(0,0)
            ┌──────────────────────────────────────┐  South=0
            │ WP00→ WP01→ WP02→ WP03→ WP04→ WP05 │  strip 0
            │ WP11← WP10← WP09← WP08← WP07← WP06 │  strip 1 (−2 m South)
            │ WP12→ WP13→ WP14→ WP15→ WP16→ WP17 │  strip 2 (−4 m South)
            │ WP23← WP22← WP21← WP20← WP19← WP18 │  strip 3 (−6 m South)
            └──────────────────────────────────────┘  South=7
               E=0   2    4    6    8   10
        """
        x_offsets = np.arange(0.0, ARENA_EAST_M  + STEP_EAST_M  / 2.0, STEP_EAST_M)
        y_offsets = np.arange(0.0, ARENA_SOUTH_M + STEP_SOUTH_M / 2.0, STEP_SOUTH_M)
        y_enu     = -y_offsets   # South → negative ENU Y

        # Z target = home_z only; flight_manager owns actual altitude AGL
        target_z  = self.home_z

        waypoints = []
        for strip_idx, dy in enumerate(y_enu):
            # Even strip → fly East (x increasing); Odd strip → fly West (x decreasing)
            xs = x_offsets if strip_idx % 2 == 0 else x_offsets[::-1]
            for dx in xs:
                waypoints.append((
                    self.home_x + dx,
                    self.home_y + dy,
                    target_z
                ))

        self.waypoints = waypoints

        self.get_logger().info(
            f"\n"
            f"  ✅  Waypoints generated\n"
            f"  Strips : {len(y_enu)}  WP/strip : {len(x_offsets)}  "
            f"Total : {len(waypoints)}\n"
            f"  East   : {x_offsets[0]:.1f} → {x_offsets[-1]:.1f} m\n"
            f"  South  : 0 → {y_offsets[-1]:.1f} m  "
            f"(ENU Y : {y_enu[0]:.1f} → {y_enu[-1]:.1f})\n"
            f"  Z      : home_z={self.home_z:.3f} (alt managed by flight_manager)\n"
        )
        for i, wp in enumerate(waypoints[:8]):
            self.get_logger().info(
                f"    WP[{i:02d}]  E={wp[0]:.1f}  S={-(wp[1]-self.home_y):.1f}  "
                f"Z={wp[2]:.2f}"
            )
        if len(waypoints) > 8:
            self.get_logger().info(f"    ... ({len(waypoints) - 8} more)")

    # ──────────────────────────────────────────────────────────
    # Image save
    # ──────────────────────────────────────────────────────────

    def _save_image(self, wp_idx: int):
        if self.latest_image is None:
            self.get_logger().warn(
                f"📷 WP[{wp_idx:02d}] No cached frame — skipping"
            )
            return
        msg = self.latest_image
        try:
            enc = msg.encoding
            if enc in ('rgb8', 'bgr8'):
                ch = 3
            elif enc == 'mono8':
                ch = 1
            else:
                self.get_logger().warn(
                    f"📷 WP[{wp_idx:02d}] Unsupported encoding '{enc}'"
                )
                return

            img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                msg.height, msg.width, ch)
            if enc == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            def fmt(v):
                return f"n{abs(v):.3f}" if v < 0 else f"{v:.3f}"

            fname = (
                f"wp{wp_idx:02d}"
                f"_E{fmt(self.cx - self.home_x)}"
                f"_S{fmt(self.home_y - self.cy)}"
                f"_Z{fmt(self.cz)}.jpg"
            )
            cv2.imwrite(os.path.join(SAVE_DIR, fname), img)
            self.get_logger().info(f"📸 WP[{wp_idx:02d}] → {fname}")

        except Exception as e:
            self.get_logger().error(f"📷 WP[{wp_idx:02d}] save failed: {e}")

    # ──────────────────────────────────────────────────────────
    # Navigation loop  (10 Hz)
    # ──────────────────────────────────────────────────────────

    def navigation_loop(self):

        # ── WAIT_HOME ─────────────────────────────────────────
        if self.state == MissionState.WAIT_HOME:
            return

        # ── GENERATE_WAYPOINTS ────────────────────────────────
        if self.state == MissionState.GENERATE_WAYPOINTS:
            self._generate_waypoints()
            self.current_wp_idx  = 0
            self.hover_count     = 0
            self.wp_start_time   = None
            self.mission_started = True
            self.state           = MissionState.FLY_WP
            return

        # ── HOLD ──────────────────────────────────────────────
        if self.state == MissionState.HOLD:
            # Keep publishing home so the drone holds the correct final position,
            # not wherever it happened to be when HOLD was entered.
            self._publish_target(self.home_x, self.home_y, self.home_z)
            return

        # ── HOME_SETTLE ───────────────────────────────────────
        if self.state == MissionState.HOME_SETTLE:
            # Continue publishing exact home coords so the flight manager
            # can fully converge before we declare HOLD.
            self._publish_target(self.home_x, self.home_y, self.home_z)
            self.home_settle_count += 1

            remaining = (HOME_SETTLE_TICKS - self.home_settle_count) * TIMER_DT
            self.get_logger().info(
                f"🏠 Settling over home … {remaining:.1f} s left  "
                f"pos=({self.cx:.2f},{self.cy:.2f},{self.cz:.2f})",
                throttle_duration_sec=1.0
            )

            if self.home_settle_count >= HOME_SETTLE_TICKS:
                self.get_logger().info("🏠 Home settled → HOLD")
                self.state = MissionState.HOLD
            return

        # ── RETURN_HOME ───────────────────────────────────────
        if self.state == MissionState.RETURN_HOME:
            # Altitude is managed by flight_manager; publish home_z so the
            # flight manager's altitude loop targets the correct height.
            # No extra Z offset added here — that's the flight manager's job.
            self._publish_target(self.home_x, self.home_y, self.home_z)

            # 3-D distance check — ensures we're aligned in X, Y, AND Z
            dist = math.sqrt(
                (self.cx - self.home_x) ** 2 +
                (self.cy - self.home_y) ** 2 +
                (self.cz - self.home_z) ** 2
            )
            self.get_logger().info(
                f"🏠 RTH  dist_3d={dist:.2f} m  "
                f"pos=({self.cx:.2f},{self.cy:.2f},{self.cz:.2f})",
                throttle_duration_sec=1.0
            )

            if dist < REACHED_THRESHOLD:
                self.get_logger().info(
                    f"🏠 Home within {REACHED_THRESHOLD} m → settling "
                    f"{HOME_SETTLE_SECONDS:.0f} s"
                )
                self.home_settle_count = 0
                self.state = MissionState.HOME_SETTLE
            return

        # ── Mission complete ───────────────────────────────────
        if self.current_wp_idx >= len(self.waypoints):
            self.get_logger().info(
                f"✅ All {len(self.waypoints)} WPs done → RTH"
            )
            self._trigger_rth()
            return

        wp = self.waypoints[self.current_wp_idx]

        # ── HOVER_WP ──────────────────────────────────────────
        if self.state == MissionState.HOVER_WP:
            self._publish_target(*wp)
            self.hover_count += 1

            # Capture at hover mid-point (drone settled)
            if self.hover_count == HOVER_TICKS // 2 and not self.image_captured:
                self._save_image(self.current_wp_idx)
                self.image_captured = True

            remaining = (HOVER_TICKS - self.hover_count) * TIMER_DT
            self.get_logger().info(
                f"⏳ WP[{self.current_wp_idx:02d}/{len(self.waypoints)-1}]  "
                f"hover {remaining:.1f} s left",
                throttle_duration_sec=1.0
            )

            if self.hover_count >= HOVER_TICKS:
                self.get_logger().info(
                    f"➡️  WP[{self.current_wp_idx:02d}] done"
                )
                self.hover_count     = 0
                self.image_captured  = False
                self.current_wp_idx += 1
                self.wp_start_time   = None
                self.state           = MissionState.FLY_WP
            return

        # ── FLY_WP ────────────────────────────────────────────
        if self.state == MissionState.FLY_WP:
            now = time.monotonic()
            if self.wp_start_time is None:
                self.wp_start_time = now

            elapsed = now - self.wp_start_time

            # WP timeout → skip and continue mission
            if elapsed > WP_TIMEOUT_SEC:
                self.get_logger().warn(
                    f"⏰ WP[{self.current_wp_idx:02d}] timed out "
                    f"({WP_TIMEOUT_SEC:.0f} s) → skip"
                )
                self.current_wp_idx += 1
                self.wp_start_time   = None
                return

            dist = math.sqrt(
                (wp[0] - self.cx) ** 2 +
                (wp[1] - self.cy) ** 2 +
                (wp[2] - self.cz) ** 2
            )
            self._publish_target(*wp)

            self.get_logger().info(
                f"🎯 WP[{self.current_wp_idx:02d}/{len(self.waypoints)-1}]  "
                f"tgt=({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f})  "
                f"pos=({self.cx:.1f},{self.cy:.1f},{self.cz:.1f})  "
                f"dist={dist:.2f} m  t={elapsed:.0f}/{WP_TIMEOUT_SEC:.0f} s",
                throttle_duration_sec=0.5
            )

            if dist < REACHED_THRESHOLD:
                self.get_logger().info(
                    f"📍 WP[{self.current_wp_idx:02d}] reached → "
                    f"hover {HOVER_SECONDS:.0f} s"
                )
                self.hover_count    = 0
                self.image_captured = False
                self.state          = MissionState.HOVER_WP
            return

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _publish_target(self, x: float, y: float, z: float):
        """
        Publish target pose.
        Orientation = identity quaternion → yaw = 0 → drone faces East (+X ENU).
        Z is passed as-is; altitude management is flight_manager_mavros's responsibility.
        """
        msg                    = PoseStamped()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = 'map'
        msg.pose.position.x    = float(x)
        msg.pose.position.y    = float(y)
        msg.pose.position.z    = float(z)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0    # valid identity quaternion → yaw=0 (East)
        self.target_pub.publish(msg)

    def _trigger_rth(self):
        self.state = MissionState.RETURN_HOME


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

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

