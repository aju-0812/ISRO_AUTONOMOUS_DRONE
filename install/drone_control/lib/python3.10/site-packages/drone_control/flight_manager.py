#!/usr/bin/env python3
"""
flight_manager_mavros.py
────────────────────────
Velocity-controlled flight manager for a REAL drone using MAVROS.

Topics used (from `ros2 topic list` on the real drone):
  SUB  /mavros/state                          — armed / mode / connected
  SUB  /mavros/local_position/odom            — current position (ENU)
  SUB  /mavros/local_position/pose            — fallback position source
  SUB  /drone_target                          — target from navigation_manager
  PUB  /mavros/setpoint_velocity/cmd_vel_unstamped — velocity commands
  SRV  /mavros/cmd/arming                     — arm / disarm
  SRV  /mavros/set_mode                       — mode switching

Coordinate frame: MAVROS uses ENU (East-North-Up), z is POSITIVE upward.
  takeoff_height = +1.5 m  (positive = up in ENU)

Flow:
  WAIT_CONNECTION → STREAM_SETPOINTS → SET_OFFBOARD → WAIT_OFFBOARD
  → ARM → WAIT_ARMED → TAKEOFF → HOVER → NAVIGATE
"""

import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

import time

# ── Tuning ────────────────────────────────────────────────────
MAX_SPEED        = 0.6   # m/s  horizontal speed cap
VERT_SPEED       = 1.5   # m/s  vertical speed cap
TAKEOFF_HEIGHT   = 3.0   # m    ENU (positive = up)
SETTLE_SECONDS   = 2.0   # s    hover at takeoff height before navigating
DT               = 0.05  # s    control loop period (20 Hz)
REACHED_XY       = 0.15  # m    horizontal dead-zone
REACHED_Z        = 0.10  # m    vertical dead-zone
MODE_RETRY_SEC   = 2.0   # s    how often to retry mode / arm commands
STREAM_SECONDS   = 3.0   # s    pre-arm setpoint streaming period


class FlightManagerMavros(Node):

    def __init__(self):
        super().__init__('flight_manager_mavros')
        self.get_logger().info("🚀 MAVROS Flight Manager started")

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

        # ── Publishers ────────────────────────────────────────
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            reliable_qos
        )

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            best_effort_qos
        )
        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            best_effort_qos
        )
        self.create_subscription(
            PoseStamped,
            '/drone_target',
            self.target_callback,
            reliable_qos
        )

        # ── Service clients ───────────────────────────────────
        self.arming_client   = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode,     '/mavros/set_mode')

        # ── State ─────────────────────────────────────────────
        self.current_state = State()

        # Current position (ENU)
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 0.0

        # Target position (ENU) — default is hover in place
        self.tx = 0.0
        self.ty = 0.0
        self.tz = TAKEOFF_HEIGHT

        # FSM
        self.phase      = 'WAIT_CONNECTION'
        self.phase_time = time.time()

        # Arm / mode cooldowns
        self.last_arm_time  = 0.0
        self.last_mode_time = 0.0

        # Takeoff settle counter
        self.settle_ticks     = 0
        self.settle_ticks_req = int(SETTLE_SECONDS / DT)

        # Loop counter for debug throttle
        self.loop_count = 0

        self.create_timer(DT, self.control_loop)

    # ── Callbacks ─────────────────────────────────────────────

    def state_callback(self, msg: State):
        self.current_state = msg

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.cx = p.x
        self.cy = p.y
        self.cz = p.z

    def target_callback(self, msg: PoseStamped):
        """
        navigation_manager publishes in its own frame.
        MAVROS ENU: x=East, y=North, z=Up.
        We accept the target directly — navigation_manager
        must publish ENU-consistent coordinates.
        """
        self.tx = msg.pose.position.x
        self.ty = msg.pose.position.y
        self.tz = msg.pose.position.z

    # ── Phase helpers ─────────────────────────────────────────

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

    # ── Velocity publisher ────────────────────────────────────

    def _publish_velocity(self, vx: float, vy: float, vz: float):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x  = float(vx)
        msg.twist.linear.y  = float(vy)
        msg.twist.linear.z  = float(vz)
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.vel_pub.publish(msg)

    def _publish_zero(self):
        self._publish_velocity(0.0, 0.0, 0.0)

    # ── Velocity toward a 3-D target ─────────────────────────

    def _velocity_toward(self, tx, ty, tz):
        ex = tx - self.cx
        ey = ty - self.cy
        ez = tz - self.cz   # ENU: positive = up

        dist_xy = math.sqrt(ex ** 2 + ey ** 2)

        # Horizontal velocity
        if dist_xy > REACHED_XY:
            scale = min(MAX_SPEED / dist_xy, MAX_SPEED)
            vx = ex * scale
            vy = ey * scale
            speed = math.sqrt(vx ** 2 + vy ** 2)
            if speed > MAX_SPEED:
                vx = vx / speed * MAX_SPEED
                vy = vy / speed * MAX_SPEED
        else:
            vx, vy = 0.0, 0.0

        # Vertical velocity
        if abs(ez) > REACHED_Z:
            vz = math.copysign(min(VERT_SPEED, abs(ez)), ez)
        else:
            vz = 0.0

        return vx, vy, vz

    # ── Main FSM ──────────────────────────────────────────────

    def control_loop(self):
        self.loop_count += 1

        # ── PHASE 1: Wait for MAVROS → FC connection ──────────
        if self.phase == 'WAIT_CONNECTION':
            self._publish_zero()
            if self.current_state.connected:
                self.get_logger().info("✅ MAVROS connected to FC!")
                self.set_phase('STREAM_SETPOINTS')
            else:
                if self.loop_count % 60 == 0:
                    self.get_logger().warn("⏳ Waiting for MAVROS connection...")
            return

        # ── PHASE 2: Stream zero-vel for STREAM_SECONDS ───────
        # OFFBOARD needs a pre-existing setpoint stream before
        # it will accept the mode switch.
        if self.phase == 'STREAM_SETPOINTS':
            self._publish_zero()
            if self.elapsed() >= STREAM_SECONDS:
                self.set_phase('SET_OFFBOARD')
            return

        # ── PHASE 3: Request OFFBOARD mode ───────────────────
        if self.phase == 'SET_OFFBOARD':
            self._publish_zero()
            if self.current_state.mode != 'OFFBOARD':
                self._send_mode('OFFBOARD')
            else:
                self.set_phase('WAIT_OFFBOARD')
            if self.elapsed() > 10.0:
                self.get_logger().warn("⚠️  OFFBOARD timeout — retrying")
                self.last_mode_time = 0.0  # force retry
            return

        if self.phase == 'WAIT_OFFBOARD':
            self._publish_zero()
            if self.current_state.mode == 'OFFBOARD':
                self.get_logger().info("✅ OFFBOARD confirmed")
                self.set_phase('ARM')
            elif self.elapsed() > 5.0:
                self.get_logger().warn("⚠️  Mode not confirmed — back to SET_OFFBOARD")
                self.set_phase('SET_OFFBOARD')
            return

        # ── PHASE 4: Arm ──────────────────────────────────────
        if self.phase == 'ARM':
            self._publish_zero()
            if not self.current_state.armed:
                self.get_logger().info(
                    "⏳ Arming... (ensure RC throttle is at MIN)",
                    throttle_duration_sec=3.0
                )
                self._send_arm(True)
            else:
                self.get_logger().info("✅ ARMED")
                self.set_phase('TAKEOFF')
            return

        # ── PHASE 5: Climb to TAKEOFF_HEIGHT ─────────────────
        if self.phase == 'TAKEOFF':
            vx, vy, vz = self._velocity_toward(0.0, 0.0, TAKEOFF_HEIGHT)
            self._publish_velocity(vx, vy, vz)

            if self.loop_count % 20 == 0:
                self.get_logger().info(
                    f"🛫 Climbing  z={self.cz:.2f}m / target={TAKEOFF_HEIGHT:.1f}m"
                )

            if self.cz >= TAKEOFF_HEIGHT - REACHED_Z:
                self.settle_ticks += 1
                if self.settle_ticks >= self.settle_ticks_req:
                    self.get_logger().info("✅ Takeoff complete — entering NAVIGATE")
                    # Initialise target to current hover position
                    self.tx, self.ty, self.tz = self.cx, self.cy, self.cz
                    self.set_phase('NAVIGATE')
            else:
                self.settle_ticks = 0
            return

        # ── PHASE 6: Navigate toward /drone_target ────────────
        if self.phase == 'NAVIGATE':
            vx, vy, vz = self._velocity_toward(self.tx, self.ty, self.tz)
            self._publish_velocity(vx, vy, vz)

            if self.loop_count % 20 == 0:
                dist_xy = math.sqrt(
                    (self.tx - self.cx) ** 2 + (self.ty - self.cy) ** 2
                )
                self.get_logger().info(
                    f"🎯 Tgt({self.tx:.2f},{self.ty:.2f},{self.tz:.2f})  "
                    f"Pos({self.cx:.2f},{self.cy:.2f},{self.cz:.2f})  "
                    f"dXY={dist_xy:.2f}m  "
                    f"Vel({vx:.2f},{vy:.2f},{vz:.2f})"
                )
            return


# ── Entry point ───────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FlightManagerMavros()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Flight Manager shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[flight_manager_mavros] ✅ Stopped")


if __name__ == '__main__':
    main()
