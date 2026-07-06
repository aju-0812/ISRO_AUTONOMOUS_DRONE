#!/usr/bin/env python3
"""
flight_manager_mavros_v4.py
────────────────────────────
Velocity-controlled flight manager for a REAL drone using MAVROS.

CHANGES vs prior v4 (hold-only + battery failsafe, ported from the sim
flight_manager_v3_tag_hold.py fixes):

  1. FIX: /drone_target watchdog no longer false-triggers when no nav
     node (vision_smart_land.py) is running at all. Previously, entering
     NAVIGATE seeded last_target_time as if a target had just arrived,
     so the watchdog started counting down immediately even with zero
     external publishers -- guaranteed failsafe-land on any flight test
     without the nav node running. Now: the watchdog only arms once an
     ACTUAL /drone_target message has been received (self.target_active).
     Until then, the drone just holds at its takeoff position
     indefinitely -- no nav node, no failsafe. Once a real nav node
     starts publishing (and could later crash/hang), the watchdog
     behaves exactly as before: HOLD at TARGET_HOLD_TIMEOUT, AUTO.LAND
     at TARGET_FAILSAFE_TIMEOUT.

  2. NEW: Battery failsafe. Subscribes to sensor_msgs/BatteryState on
     /mavros/battery. If percentage drops at/below BATTERY_LOW_THRESHOLD,
     triggers the same AUTO.LAND failsafe path used by the target
     watchdog and the HOLD-phase drift safety net -- regardless of
     current phase (TAKEOFF/HOLD/NAVIGATE), since _trigger_failsafe_land
     sets self.phase directly and the FSM is phase-string-driven (no
     restructuring needed here, unlike the sim version's flag-based loop).

Everything else (tag fusion gating, drift safety net, target/detector
watchdogs) is UNCHANGED from the prior v4.

Coordinate frame: MAVROS ENU. z is POSITIVE upward.
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from sensor_msgs.msg import BatteryState
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


# ── Tuning: Takeoff ──────────────────────────────────────────
TAKEOFF_HEIGHT   = 3.0   # m   ENU, positive = up
CLIMB_SPEED      = 1.0   # m/s vertical speed during climb
SETTLE_SECONDS   = 2.0   # s   hover at takeoff height before navigating
DT               = 0.05  # s   control loop period (20 Hz)

# ── Tuning: Navigation (speed-limited P controller) ──────────
NAV_SPEED        = 0.6   # m/s  max horizontal cruise speed
SLOW_RADIUS      = 0.8   # m    start braking within this radius of target
MIN_SPEED        = 0.05  # m/s  minimum speed when very close (avoid stall)
VERT_SPEED       = 0.5   # m/s  max vertical correction speed
KP_XY            = 1.0   # proportional gain, horizontal
KP_Z             = 1.0   # proportional gain, vertical

# ── Dead zones ───────────────────────────────────────────────
REACHED_XY       = 0.10  # m
REACHED_Z        = 0.08  # m

# ── Mode / arm retry ─────────────────────────────────────────
MODE_RETRY_SEC   = 2.0
STREAM_SECONDS   = 3.0

# ── AprilTag drift correction ───────────────────────────────
TAG_TIMEOUT_SEC    = 1.0      # if no fresh /tag_pose within this, treat as "no tag"
TAG_FUSION_PHASES  = ('TAKEOFF', 'HOLD')   # hard limit -- fusion NEVER runs outside these

# ── Static camera mount transform (EDIT to match your physical mount) ──
CAM_ROLL  = -math.pi / 2
CAM_PITCH = 0.0
CAM_YAW   = 0.0
TAG_WORLD_Z = 0.0

# ── Watchdog / failsafe tuning ─────────────────────────────────
TAG_DETECTOR_HEARTBEAT_TIMEOUT = 5.0   # s -- warn if /tag_detector_alive goes silent
HOLD_MAX_DRIFT_XY              = 2.0   # m -- HOLD failsafe radius from home
TARGET_HOLD_TIMEOUT            = 2.0   # s -- /drone_target stale -> hold position
TARGET_FAILSAFE_TIMEOUT        = 6.0   # s -- /drone_target stale -> trigger AUTO.LAND
FAILSAFE_MODE_RETRY_SEC        = 2.0

# ── NEW: Battery failsafe tuning ─────────────────────────────
BATTERY_LOW_THRESHOLD = 0.20   # remaining fraction (0.0-1.0); <= this -> AUTO.LAND
BATTERY_WARN_MARGIN   = 0.10   # early heads-up this far above the hard threshold


class FlightManagerMavros(Node):

    def __init__(self):
        super().__init__('flight_manager_mavros')
        self.get_logger().info("MAVROS Flight Manager v4 (hold-only + battery failsafe) started")

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

        # ── Publishers ──────────────────────────────────────
        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', reliable_qos)

        self.vision_pose_pub = self.create_publisher(
            PoseStamped, '/mavros/vision_pose/pose', reliable_qos)

        # ── Subscribers ─────────────────────────────────────
        self.create_subscription(State, '/mavros/state', self.state_callback, best_effort_qos)
        self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_callback, best_effort_qos)
        self.create_subscription(PoseStamped, '/drone_target', self.target_callback, reliable_qos)
        self.create_subscription(PoseStamped, '/tag_pose', self.tag_callback, best_effort_qos)
        self.create_subscription(Bool, '/tag_detector_alive', self.tag_heartbeat_callback, reliable_qos)
        # NEW: battery status
        self.create_subscription(BatteryState, '/mavros/battery', self.battery_callback, best_effort_qos)

        # ── Service clients ─────────────────────────────────
        self.arming_client   = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode,     '/mavros/set_mode')

        # ── State ───────────────────────────────────────────
        self.current_state = State()
        self.cx = self.cy = self.cz = 0.0

        self.tx = 0.0
        self.ty = 0.0
        self.tz = TAKEOFF_HEIGHT

        self.home_x = 0.0
        self.home_y = 0.0

        self.phase      = 'WAIT_CONNECTION'
        self.phase_time = time.time()

        self.last_arm_time  = 0.0
        self.last_mode_time = 0.0
        self.last_failsafe_mode_time = 0.0

        self.settle_ticks     = 0
        self.settle_ticks_req = int(SETTLE_SECONDS / DT)

        self.loop_count = 0

        self._cam_rot = self._euler_to_rot(CAM_ROLL, CAM_PITCH, CAM_YAW)

        self.last_tag_time    = 0.0
        self.tag_fusion_count = 0

        self.last_heartbeat_time   = time.time()
        self.heartbeat_warned      = False
        self.last_target_time      = time.time()
        self.target_stale_warned   = False
        # NEW: watchdog only arms once a REAL external target message has
        # ever been received -- prevents false failsafe when no nav node
        # is running at all (e.g. pure takeoff/hold testing).
        self.target_active         = False
        self.failsafe_reason       = None

        # NEW: battery state
        self.battery_remaining = 1.0   # assume full until first message
        self.battery_warned    = False

        self.create_timer(DT, self.control_loop)

    # ── Math helper (no external tf_transformations dependency) ─

    @staticmethod
    def _euler_to_rot(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr
        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr
        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr
        return ((r00, r01, r02), (r10, r11, r12), (r20, r21, r22))

    @staticmethod
    def _mat_vec(mat, vec):
        return (
            mat[0][0]*vec[0] + mat[0][1]*vec[1] + mat[0][2]*vec[2],
            mat[1][0]*vec[0] + mat[1][1]*vec[1] + mat[1][2]*vec[2],
            mat[2][0]*vec[0] + mat[2][1]*vec[1] + mat[2][2]*vec[2],
        )

    # ── Callbacks ───────────────────────────────────────────

    def state_callback(self, msg: State):
        self.current_state = msg

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.cx, self.cy, self.cz = p.x, p.y, p.z

    def target_callback(self, msg: PoseStamped):
        self.tx = msg.pose.position.x
        self.ty = msg.pose.position.y
        self.tz = msg.pose.position.z
        self.last_target_time    = time.time()
        self.target_stale_warned = False
        self.target_active       = True   # NEW: a real nav node is now talking to us

    def tag_heartbeat_callback(self, msg: Bool):
        self.last_heartbeat_time = time.time()
        self.heartbeat_warned    = False

    # ── NEW: battery callback ────────────────────────────────
    def battery_callback(self, msg: BatteryState):
        # BatteryState.percentage is 0.0-1.0 (or NaN if unknown -- guard for that)
        if msg.percentage != msg.percentage:  # NaN check
            return
        self.battery_remaining = msg.percentage
        if msg.percentage <= BATTERY_LOW_THRESHOLD and self.phase != 'FAILSAFE_LAND':
            self._trigger_failsafe_land(
                f"battery low: {msg.percentage * 100:.0f}% remaining "
                f"(threshold {BATTERY_LOW_THRESHOLD * 100:.0f}%)")
        elif not self.battery_warned and msg.percentage <= BATTERY_LOW_THRESHOLD + BATTERY_WARN_MARGIN:
            self.battery_warned = True
            self.get_logger().warn(
                f"🔋 Battery getting low: {msg.percentage * 100:.0f}% remaining")

    def tag_callback(self, msg: PoseStamped):
        """
        Tag pose arrives already in CAMERA OPTICAL FRAME xyz.
        HARD GATE: only fused into vision_pose during TAG_FUSION_PHASES
        (TAKEOFF/HOLD). NAVIGATE phase ignores it completely.
        """
        if self.phase not in TAG_FUSION_PHASES:
            return

        self.last_tag_time = time.time()

        p = msg.pose.position
        cam_xyz = (p.x, p.y, p.z)
        body_xyz = self._mat_vec(self._cam_rot, cam_xyz)

        vp = PoseStamped()
        vp.header.stamp    = self.get_clock().now().to_msg()
        vp.header.frame_id = 'map'
        vp.pose.position.x = -body_xyz[0]
        vp.pose.position.y = -body_xyz[1]
        vp.pose.position.z = -body_xyz[2] + TAG_WORLD_Z

        self.vision_pose_pub.publish(vp)
        self.tag_fusion_count += 1

    def _tag_is_fresh(self) -> bool:
        return (time.time() - self.last_tag_time) < TAG_TIMEOUT_SEC

    # ── Phase helpers ───────────────────────────────────────

    def set_phase(self, phase: str):
        self.phase      = phase
        self.phase_time = time.time()
        self.get_logger().info(f"======= PHASE: {phase} =======")

    def elapsed(self) -> float:
        return time.time() - self.phase_time

    def _send_arm(self, value: bool):
        now = time.time()
        if now - self.last_arm_time < MODE_RETRY_SEC:
            return
        self.last_arm_time = now
        req = CommandBool.Request()
        req.value = value
        self.arming_client.call_async(req)
        self.get_logger().info(f"[ARM] Sending arm={value}")

    def _send_mode(self, mode: str):
        now = time.time()
        if now - self.last_mode_time < MODE_RETRY_SEC:
            return
        self.last_mode_time = now
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_client.call_async(req)
        self.get_logger().info(f"[MODE] Requesting: {mode}")

    # ── Failsafe trigger ────────────────────────────────────

    def _trigger_failsafe_land(self, reason: str):
        if self.phase == 'FAILSAFE_LAND':
            return
        self.failsafe_reason = reason
        self.get_logger().error(f"🚨 FAILSAFE: {reason} -> commanding AUTO.LAND")
        self.set_phase('FAILSAFE_LAND')

    # ── Velocity publisher ──────────────────────────────────

    def _publish_velocity(self, vx: float, vy: float, vz: float):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = float(vx)
        msg.twist.linear.y  = float(vy)
        msg.twist.linear.z  = float(vz)
        self.vel_pub.publish(msg)

    def _publish_zero(self):
        self._publish_velocity(0.0, 0.0, 0.0)

    # ── Speed-limited P controller (smooth braking taper) ──

    def _velocity_toward(self, tx, ty, tz):
        ex = tx - self.cx
        ey = ty - self.cy
        ez = tz - self.cz

        dist_xy = math.sqrt(ex ** 2 + ey ** 2)

        if dist_xy > REACHED_XY:
            vx_raw = ex * KP_XY
            vy_raw = ey * KP_XY
            speed_raw = math.sqrt(vx_raw ** 2 + vy_raw ** 2)

            if dist_xy < SLOW_RADIUS:
                scale = dist_xy / SLOW_RADIUS
                speed_limit = MIN_SPEED + (NAV_SPEED - MIN_SPEED) * scale
            else:
                speed_limit = NAV_SPEED

            if speed_raw > speed_limit and speed_raw > 0:
                ratio = speed_limit / speed_raw
                vx, vy = vx_raw * ratio, vy_raw * ratio
            else:
                vx, vy = vx_raw, vy_raw
        else:
            vx, vy = 0.0, 0.0

        if abs(ez) > REACHED_Z:
            vz_raw = ez * KP_Z
            vz = math.copysign(min(abs(vz_raw), VERT_SPEED), vz_raw)
        else:
            vz = 0.0

        return vx, vy, vz

    # ── Watchdog checks (run every tick, cheap) ────────

    def _check_detector_heartbeat(self):
        if self.phase not in TAG_FUSION_PHASES:
            return
        age = time.time() - self.last_heartbeat_time
        if age > TAG_DETECTOR_HEARTBEAT_TIMEOUT and not self.heartbeat_warned:
            self.heartbeat_warned = True
            self.get_logger().warn(
                f"⚠️ No /tag_detector_alive heartbeat for {age:.1f}s — "
                f"apriltag_detector_node may be down (not just 'no tag visible')")

    # ── Main FSM ────────────────────────────────────────────

    def control_loop(self):
        self.loop_count += 1
        self._check_detector_heartbeat()

        if self.phase == 'WAIT_CONNECTION':
            self._publish_zero()
            if self.current_state.connected:
                self.get_logger().info("MAVROS connected to FC")
                self.set_phase('STREAM_SETPOINTS')
            elif self.loop_count % 60 == 0:
                self.get_logger().warn("Waiting for MAVROS connection...")
            return

        if self.phase == 'STREAM_SETPOINTS':
            self._publish_zero()
            if self.elapsed() >= STREAM_SECONDS:
                self.set_phase('SET_OFFBOARD')
            return

        if self.phase == 'SET_OFFBOARD':
            self._publish_zero()
            if self.current_state.mode != 'OFFBOARD':
                self._send_mode('OFFBOARD')
            else:
                self.set_phase('WAIT_OFFBOARD')
            if self.elapsed() > 10.0:
                self.get_logger().warn("OFFBOARD timeout — retrying")
                self.last_mode_time = 0.0
            return

        if self.phase == 'WAIT_OFFBOARD':
            self._publish_zero()
            if self.current_state.mode == 'OFFBOARD':
                self.get_logger().info("OFFBOARD confirmed")
                self.set_phase('ARM')
            elif self.elapsed() > 5.0:
                self.get_logger().warn("Mode not confirmed — retrying SET_OFFBOARD")
                self.set_phase('SET_OFFBOARD')
            return

        if self.phase == 'ARM':
            self._publish_zero()
            if not self.current_state.armed:
                self._send_arm(True)
            else:
                self.get_logger().info("ARMED")
                self.home_x, self.home_y = self.cx, self.cy
                self.set_phase('TAKEOFF')
            return

        if self.phase == 'TAKEOFF':
            vx, vy, vz = 0.0, 0.0, 0.0
            ez = TAKEOFF_HEIGHT - self.cz
            if abs(ez) > REACHED_Z:
                vz = math.copysign(min(CLIMB_SPEED, abs(ez)), ez)

            self._publish_velocity(vx, vy, vz)

            if self.loop_count % 20 == 0:
                tag_status = "TAG-LOCKED" if self._tag_is_fresh() else "no tag (flow-only)"
                self.get_logger().info(
                    f"Climbing z={self.cz:.2f}/{TAKEOFF_HEIGHT:.1f}m [{tag_status}]")

            if self.cz >= TAKEOFF_HEIGHT - REACHED_Z:
                self.set_phase('HOLD')
            return

        if self.phase == 'HOLD':
            drift = math.hypot(self.cx - self.home_x, self.cy - self.home_y)
            if drift > HOLD_MAX_DRIFT_XY:
                self._trigger_failsafe_land(
                    f"HOLD drift {drift:.2f}m exceeded {HOLD_MAX_DRIFT_XY}m limit "
                    f"(tag lock: {'fresh' if self._tag_is_fresh() else 'stale/none'})")
                return

            ez = TAKEOFF_HEIGHT - self.cz
            vz = 0.0
            if abs(ez) > REACHED_Z:
                vz = math.copysign(min(abs(ez) * KP_Z, VERT_SPEED), ez)
            self._publish_velocity(0.0, 0.0, vz)

            self.settle_ticks += 1
            if self.loop_count % 20 == 0:
                tag_status = "TAG-ASSISTED HOLD" if self._tag_is_fresh() else "FLOW-ONLY HOLD (no tag)"
                self.get_logger().info(
                    f"Settling {self.settle_ticks}/{self.settle_ticks_req} "
                    f"Pos({self.cx:.2f},{self.cy:.2f},{self.cz:.2f}) "
                    f"drift={drift:.2f}m [{tag_status}]")

            if self.settle_ticks >= self.settle_ticks_req:
                self.get_logger().info(
                    f"Takeoff/hold complete (tag fused {self.tag_fusion_count} times) — entering NAVIGATE "
                    f"(holding here until a nav node sends /drone_target)")
                self.tx, self.ty, self.tz = self.cx, self.cy, self.cz
                # NOTE: last_target_time intentionally NOT reset here anymore --
                # the watchdog only starts caring once target_active flips True
                # via a real /drone_target message (see target_callback).
                self.set_phase('NAVIGATE')
            return

        if self.phase == 'NAVIGATE':
            # NEW: watchdog only active once a real nav node has spoken
            if self.target_active:
                target_age = time.time() - self.last_target_time

                if target_age > TARGET_FAILSAFE_TIMEOUT:
                    self._trigger_failsafe_land(
                        f"/drone_target stale for {target_age:.1f}s "
                        f"(vision_smart_land.py may have crashed/hung)")
                    return

                if target_age > TARGET_HOLD_TIMEOUT:
                    if not self.target_stale_warned:
                        self.target_stale_warned = True
                        self.get_logger().warn(
                            f"⚠️ /drone_target stale ({target_age:.1f}s) — holding position, "
                            f"will failsafe-land at {TARGET_FAILSAFE_TIMEOUT:.0f}s")
                    self._publish_zero()
                    return
            # else: no nav node has ever connected -- just hold at the
            # position captured when HOLD completed, no watchdog, no failsafe.

            vx, vy, vz = self._velocity_toward(self.tx, self.ty, self.tz)
            self._publish_velocity(vx, vy, vz)

            if self.loop_count % 20 == 0:
                dist_xy = math.sqrt((self.tx - self.cx) ** 2 + (self.ty - self.cy) ** 2)
                self.get_logger().info(
                    f"Tgt({self.tx:.2f},{self.ty:.2f},{self.tz:.2f}) "
                    f"Pos({self.cx:.2f},{self.cy:.2f},{self.cz:.2f}) "
                    f"dXY={dist_xy:.2f}m Vel({vx:.2f},{vy:.2f},{vz:.2f})")
            return

        if self.phase == 'FAILSAFE_LAND':
            if self.current_state.mode != 'AUTO.LAND':
                now = time.time()
                if now - self.last_failsafe_mode_time >= FAILSAFE_MODE_RETRY_SEC:
                    self.last_failsafe_mode_time = now
                    self._send_mode('AUTO.LAND')
            if self.loop_count % 40 == 0:
                self.get_logger().error(
                    f"🚨 FAILSAFE_LAND active — reason: {self.failsafe_reason} "
                    f"| mode={self.current_state.mode} alt={self.cz:.2f}m "
                    f"batt={self.battery_remaining*100:.0f}%")
            return


def main(args=None):
    rclpy.init(args=args)
    node = FlightManagerMavros()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Flight Manager shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[flight_manager_mavros_v4] Stopped")


if __name__ == '__main__':
    main()
