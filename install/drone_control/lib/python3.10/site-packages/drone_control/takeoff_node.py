#!/usr/bin/env python3
"""
flight_manager_mavros.py  (FIXED — stable no-drift takeoff)
────────────────────────────────────────────────────────────
Key fixes over original:
  1. Home XY locked at ARM time → takeoff climbs straight up, no XY drift
  2. Takeoff target uses home_x/home_y instead of absolute (0,0)
  3. Settle logic uses abs(cz - target) band, not one-sided >= check
  4. Proportional velocity controller (P-gain) replaces bang-bang scaler
     → smooth deceleration near target, no oscillation
  5. nav_manager waypoints are IGNORED until NAVIGATE phase
     (target stays at home XY during takeoff & settle)
  6. Publisher topic fixed: cmd_vel_unstamped → cmd_vel (TwistStamped)

Topics:
  SUB  /mavros/state                          — armed / mode / connected
  SUB  /mavros/local_position/odom            — current position (ENU)
  SUB  /drone_target                          — target from navigation_manager
  PUB  /mavros/setpoint_velocity/cmd_vel      — velocity commands (TwistStamped)
  SRV  /mavros/cmd/arming                     — arm / disarm
  SRV  /mavros/set_mode                       — mode switching

FSM:
  WAIT_CONNECTION → STREAM_SETPOINTS → SET_OFFBOARD → WAIT_OFFBOARD
  → ARM → TAKEOFF → SETTLE → NAVIGATE
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# ══════════════════════════════════════════════════════════════
#  Tuning parameters
# ══════════════════════════════════════════════════════════════

TAKEOFF_HEIGHT   = 2.0    # m AGL (ENU +Z)
SETTLE_SECONDS   = 2.0    # s  hover at takeoff height before navigating
STREAM_SECONDS   = 3.0    # s  pre-arm setpoint streaming period
DT               = 0.05   # s  control loop period (20 Hz)
MODE_RETRY_SEC   = 2.0    # s  cooldown between arm/mode retries

# Proportional gains (tune on bench with props OFF first)
P_XY             = 0.6    # horizontal: vel = P * error  (clipped to MAX_SPEED)
P_Z              = 0.5    # vertical  : vel = P * error  (clipped to VERT_SPEED)

MAX_SPEED        = 0.5    # m/s  horizontal speed cap
VERT_SPEED       = 0.4    # m/s  vertical speed cap

# Dead-zones (stop commanding when closer than this)
REACHED_XY       = 0.12   # m
REACHED_Z        = 0.10   # m

# Takeoff settle band: must stay within ±SETTLE_BAND of target Z
SETTLE_BAND_Z    = 0.12   # m


# ══════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════

class FlightManagerMavros(Node):

    def __init__(self):
        super().__init__('flight_manager_mavros')
        self.get_logger().info("🚀 MAVROS Flight Manager (fixed) started")

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

        # ── Publishers ────────────────────────────────────────
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',   # TwistStamped topic
            rel_qos
        )

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(State,    '/mavros/state',
                                 self.state_callback,  be_qos)
        self.create_subscription(Odometry, '/mavros/local_position/odom',
                                 self.odom_callback,   be_qos)
        self.create_subscription(PoseStamped, '/drone_target',
                                 self.target_callback, rel_qos)

        # ── Service clients ───────────────────────────────────
        self.arming_client   = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode,     '/mavros/set_mode')

        # ── Internal state ────────────────────────────────────
        self.current_state = State()

        # Current ENU position
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 0.0

        # Home position — locked at ARM time (FIX #1)
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_locked = False

        # Navigation target — only used in NAVIGATE phase (FIX #5)
        self.nav_tx = None   # None = not yet received
        self.nav_ty = None
        self.nav_tz = None

        # FSM
        self.phase      = 'WAIT_CONNECTION'
        self.phase_time = time.time()

        # Cooldowns
        self.last_arm_time  = 0.0
        self.last_mode_time = 0.0

        # Settle counter
        self.settle_ticks     = 0
        self.settle_ticks_req = int(SETTLE_SECONDS / DT)

        self.loop_count = 0

        self.create_timer(DT, self.control_loop)

    # ──────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────

    def state_callback(self, msg: State):
        self.current_state = msg

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.cx = p.x
        self.cy = p.y
        self.cz = p.z

    def target_callback(self, msg: PoseStamped):
        """
        Cache nav_manager targets.  They are only applied in NAVIGATE phase
        so takeoff is never disturbed by incoming waypoints. (FIX #5)
        """
        self.nav_tx = msg.pose.position.x
        self.nav_ty = msg.pose.position.y
        self.nav_tz = msg.pose.position.z

    # ──────────────────────────────────────────────────────────
    # Phase helpers
    # ──────────────────────────────────────────────────────────

    def set_phase(self, phase: str):
        self.phase      = phase
        self.phase_time = time.time()
        self.get_logger().info(f"═══════ PHASE → {phase} ═══════")

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

    # ──────────────────────────────────────────────────────────
    # Velocity publisher
    # ──────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────
    # Proportional velocity controller  (FIX #4)
    # ──────────────────────────────────────────────────────────

    def _velocity_toward(self, tx: float, ty: float, tz: float):
        """
        Pure proportional controller.
        vel = P * error, clipped to max speeds.
        Produces smooth deceleration near target (no bang-bang oscillation).
        """
        ex = tx - self.cx
        ey = ty - self.cy
        ez = tz - self.cz

        # Horizontal
        dist_xy = math.sqrt(ex**2 + ey**2)
        if dist_xy > REACHED_XY:
            raw_vx = ex * P_XY
            raw_vy = ey * P_XY
            # Clip to MAX_SPEED while preserving direction
            speed  = math.sqrt(raw_vx**2 + raw_vy**2)
            if speed > MAX_SPEED:
                raw_vx = raw_vx / speed * MAX_SPEED
                raw_vy = raw_vy / speed * MAX_SPEED
            vx, vy = raw_vx, raw_vy
        else:
            vx, vy = 0.0, 0.0

        # Vertical
        if abs(ez) > REACHED_Z:
            vz = max(-VERT_SPEED, min(VERT_SPEED, ez * P_Z))
        else:
            vz = 0.0

        return vx, vy, vz

    # ──────────────────────────────────────────────────────────
    # FSM
    # ──────────────────────────────────────────────────────────

    def control_loop(self):
        self.loop_count += 1
        dbg = (self.loop_count % 20 == 0)   # ~1 Hz debug prints

        # ── WAIT_CONNECTION ───────────────────────────────────
        if self.phase == 'WAIT_CONNECTION':
            self._publish_zero()
            if self.current_state.connected:
                self.get_logger().info("✅ MAVROS connected to FC")
                self.set_phase('STREAM_SETPOINTS')
            elif dbg:
                self.get_logger().warn("⏳ Waiting for MAVROS connection...")
            return

        # ── STREAM_SETPOINTS ──────────────────────────────────
        # Stream zero-vel so FC accepts OFFBOARD mode switch
        if self.phase == 'STREAM_SETPOINTS':
            self._publish_zero()
            if self.elapsed() >= STREAM_SECONDS:
                self.set_phase('SET_OFFBOARD')
            return

        # ── SET_OFFBOARD ──────────────────────────────────────
        if self.phase == 'SET_OFFBOARD':
            self._publish_zero()
            if self.current_state.mode == 'OFFBOARD':
                self.set_phase('WAIT_OFFBOARD')
            else:
                self._send_mode('OFFBOARD')
                if self.elapsed() > 10.0:
                    self.get_logger().warn("⚠️  OFFBOARD timeout — retrying")
                    self.last_mode_time = 0.0
            return

        # ── WAIT_OFFBOARD ─────────────────────────────────────
        if self.phase == 'WAIT_OFFBOARD':
            self._publish_zero()
            if self.current_state.mode == 'OFFBOARD':
                self.get_logger().info("✅ OFFBOARD confirmed")
                self.set_phase('ARM')
            elif self.elapsed() > 5.0:
                self.get_logger().warn("⚠️  Mode not confirmed → retry SET_OFFBOARD")
                self.set_phase('SET_OFFBOARD')
            return

        # ── ARM ───────────────────────────────────────────────
        if self.phase == 'ARM':
            self._publish_zero()
            if not self.current_state.armed:
                if dbg:
                    self.get_logger().info("⏳ Arming...")
                self._send_arm(True)
            else:
                # FIX #1 — lock home XY exactly when motors arm
                if not self.home_locked:
                    self.home_x    = self.cx
                    self.home_y    = self.cy
                    self.home_z    = self.cz
                    self.home_locked = True
                    self.get_logger().info(
                        f"🏠 Home locked at ENU "
                        f"({self.home_x:.3f}, {self.home_y:.3f}, {self.home_z:.3f})"
                    )
                self.get_logger().info("✅ ARMED → TAKEOFF")
                self.set_phase('TAKEOFF')
            return

        # Target takeoff Z in absolute ENU
        takeoff_z = self.home_z + TAKEOFF_HEIGHT

        # ── TAKEOFF ───────────────────────────────────────────
        if self.phase == 'TAKEOFF':
            # FIX #2 — hold home XY while climbing (no horizontal drift)
            vx, vy, vz = self._velocity_toward(
                self.home_x, self.home_y, takeoff_z
            )
            self._publish_velocity(vx, vy, vz)

            if dbg:
                self.get_logger().info(
                    f"🛫 Climbing  z={self.cz:.2f} / target={takeoff_z:.2f} m  "
                    f"vel=({vx:.2f},{vy:.2f},{vz:.2f})"
                )

            # FIX #3 — use band check, not one-sided >=
            if abs(self.cz - takeoff_z) < SETTLE_BAND_Z:
                self.settle_ticks += 1
                if self.settle_ticks >= self.settle_ticks_req:
                    self.get_logger().info(
                        f"✅ Takeoff complete — settled at z={self.cz:.2f} m → SETTLE"
                    )
                    self.settle_ticks = 0
                    self.set_phase('SETTLE')
            else:
                self.settle_ticks = 0   # reset if Z drifts out of band
            return

        # ── SETTLE ────────────────────────────────────────────
        # Extra hover at home XY to confirm stability before nav takes over
        if self.phase == 'SETTLE':
            vx, vy, vz = self._velocity_toward(
                self.home_x, self.home_y, takeoff_z
            )
            self._publish_velocity(vx, vy, vz)

            if dbg:
                self.get_logger().info(
                    f"⏳ Settling  z={self.cz:.2f}  "
                    f"XY err=({self.home_x - self.cx:.3f},"
                    f"{self.home_y - self.cy:.3f}) m"
                )

            if self.elapsed() >= SETTLE_SECONDS:
                self.get_logger().info("✅ Settled → NAVIGATE")
                self.set_phase('NAVIGATE')
            return

        # ── NAVIGATE ──────────────────────────────────────────
        if self.phase == 'NAVIGATE':
            # FIX #5 — only apply nav targets here, not during takeoff
            if self.nav_tx is None:
                # nav_manager hasn't published yet — hold home
                vx, vy, vz = self._velocity_toward(
                    self.home_x, self.home_y, takeoff_z
                )
                if dbg:
                    self.get_logger().warn("⏳ Waiting for /drone_target ...")
            else:
                vx, vy, vz = self._velocity_toward(
                    self.nav_tx, self.nav_ty, self.nav_tz
                )
                if dbg:
                    dist_xy = math.sqrt(
                        (self.nav_tx - self.cx)**2 + (self.nav_ty - self.cy)**2
                    )
                    self.get_logger().info(
                        f"🎯 Tgt({self.nav_tx:.2f},{self.nav_ty:.2f},{self.nav_tz:.2f})  "
                        f"Pos({self.cx:.2f},{self.cy:.2f},{self.cz:.2f})  "
                        f"dXY={dist_xy:.2f} m  "
                        f"Vel({vx:.2f},{vy:.2f},{vz:.2f})"
                    )

            self._publish_velocity(vx, vy, vz)
            return


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

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
