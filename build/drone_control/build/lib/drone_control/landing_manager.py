#!/usr/bin/env python3
"""
landing_mavros.py
──────────────────
Safe landing sequence for a REAL drone using MAVROS.

Based on the proven landing_node.py that works on the real drone,
extended with:
  • Graceful velocity-zeroing before mode switch
  • Configurable land mode (AUTO.LAND for PX4, LAND for ArduPilot)
  • Force-disarm backup with timeout
  • Clean node shutdown after landing confirmed

Topics used:
  SUB  /mavros/state                       — armed / mode / connected
  PUB  /mavros/setpoint_position/local     — keep setpoint stream alive
  SRV  /mavros/set_mode                    — trigger AUTO.LAND
  SRV  /mavros/cmd/arming                  — backup force-disarm

Phase flow:
  WAIT_CONNECTION → CHECK_ARMED → ZERO_VELOCITY
  → SET_LAND_MODE → WAIT_TOUCHDOWN → [FORCE_DISARM] → FINISHED
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool

import time

# ── Tuning ────────────────────────────────────────────────────
LAND_MODE          = 'AUTO.LAND'   # PX4 — change to 'LAND' for ArduPilot
ZERO_VEL_SECONDS   = 1.5          # s  — publish zero setpoint before mode switch
TOUCHDOWN_TIMEOUT  = 30.0         # s  — force-disarm if not landed in this time
MODE_RETRY_SEC     = 2.0          # s  — retry cooldown for service calls
LOOP_HZ            = 20           # Hz — control loop rate
LOOP_DT            = 1.0 / LOOP_HZ


class LandingMavros(Node):

    def __init__(self):
        super().__init__('landing_mavros')
        self.get_logger().info(
            f"🛬 Landing Node started  land_mode={LAND_MODE}"
        )

        # ── QoS ───────────────────────────────────────────────
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Publisher ─────────────────────────────────────────
        # Keep publishing setpoints so OFFBOARD does not cut out
        # before we switch to AUTO.LAND
        self.pos_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            10
        )

        # ── Subscriber ────────────────────────────────────────
        self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            best_effort_qos
        )

        # ── Service clients ───────────────────────────────────
        self.set_mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.arming_client   = self.create_client(CommandBool, '/mavros/cmd/arming')

        # ── State ─────────────────────────────────────────────
        self.current_state     = State()
        self.phase             = 'WAIT_CONNECTION'
        self.phase_time        = time.time()
        self.last_mode_time    = 0.0
        self.last_disarm_time  = 0.0

        # ── Timer ─────────────────────────────────────────────
        self.timer = self.create_timer(LOOP_DT, self.control_loop)

    # ── Callbacks ─────────────────────────────────────────────

    def state_callback(self, msg: State):
        self.current_state = msg

    # ── Helpers ───────────────────────────────────────────────

    def set_phase(self, phase: str):
        self.phase      = phase
        self.phase_time = time.time()
        self.get_logger().info(f"======= PHASE: {phase} =======")

    def elapsed(self) -> float:
        return time.time() - self.phase_time

    def _publish_zero_setpoint(self):
        """Publish a zero position setpoint to keep the setpoint stream alive."""
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0
        self.pos_pub.publish(msg)

    def _send_mode(self, mode: str):
        now = time.time()
        if now - self.last_mode_time < MODE_RETRY_SEC:
            return
        self.last_mode_time = now
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_client.call_async(req)
        self.get_logger().info(f"[MODE] Requesting: {mode}")

    def _send_disarm(self):
        now = time.time()
        if now - self.last_disarm_time < MODE_RETRY_SEC:
            return
        self.last_disarm_time = now
        req = CommandBool.Request()
        req.value = False   # False = disarm
        self.arming_client.call_async(req)
        self.get_logger().warn("[ARM] Force-disarm sent")

    # ── FSM ───────────────────────────────────────────────────

    def control_loop(self):

        # PHASE 1: Wait for MAVROS → FC connection
        if self.phase == 'WAIT_CONNECTION':
            if self.current_state.connected:
                self.get_logger().info("✅ MAVROS connected to FC!")
                self.set_phase('CHECK_ARMED')
            else:
                self.get_logger().info(
                    "⏳ Waiting for MAVROS connection...",
                    throttle_duration_sec=3.0
                )
            return

        # PHASE 2: Verify the drone is actually armed / airborne
        if self.phase == 'CHECK_ARMED':
            if self.current_state.armed:
                self.get_logger().info("✅ Drone is armed — beginning landing sequence")
                self.set_phase('ZERO_VELOCITY')
            else:
                self.get_logger().warn(
                    "⚠️  Drone is already DISARMED — nothing to land",
                    throttle_duration_sec=3.0
                )
            return

        # PHASE 3: Publish zero setpoints briefly before mode switch
        # This prevents OFFBOARD from dropping out mid-flight during the
        # mode change, which could cause an uncontrolled descent.
        if self.phase == 'ZERO_VELOCITY':
            self._publish_zero_setpoint()
            if self.elapsed() >= ZERO_VEL_SECONDS:
                self.set_phase('SET_LAND_MODE')
            return

        # PHASE 4: Switch to AUTO.LAND
        if self.phase == 'SET_LAND_MODE':
            self._publish_zero_setpoint()
            if self.current_state.mode != LAND_MODE:
                self._send_mode(LAND_MODE)
            else:
                self.get_logger().info(f"✅ Mode confirmed: {LAND_MODE}")
                self.set_phase('WAIT_TOUCHDOWN')
            return

        # PHASE 5: Wait for FC to land and auto-disarm
        if self.phase == 'WAIT_TOUCHDOWN':
            self._publish_zero_setpoint()

            if not self.current_state.armed:
                self.get_logger().info("🛬 Touchdown confirmed! Drone auto-disarmed.")
                self.set_phase('FINISHED')
                return

            self.get_logger().info(
                f"⬇️  Descending... mode={self.current_state.mode} | "
                f"elapsed={self.elapsed():.1f}s / timeout={TOUCHDOWN_TIMEOUT:.0f}s",
                throttle_duration_sec=2.0
            )

            if self.elapsed() > TOUCHDOWN_TIMEOUT:
                self.get_logger().warn(
                    "⚠️  Touchdown timeout reached — forcing disarm"
                )
                self.set_phase('FORCE_DISARM')
            return

        # PHASE 6: Force disarm (backup)
        if self.phase == 'FORCE_DISARM':
            self._publish_zero_setpoint()
            if self.current_state.armed:
                self._send_disarm()
                self.get_logger().warn(
                    "⚠️  Waiting for force-disarm to take effect...",
                    throttle_duration_sec=2.0
                )
            else:
                self.get_logger().info("✅ Force-disarm successful")
                self.set_phase('FINISHED')
            return

        # PHASE 7: Done — cancel timer so this node idles cleanly
        if self.phase == 'FINISHED':
            self.get_logger().info(
                "✅ Landing sequence complete. Safe to kill node.",
                throttle_duration_sec=5.0
            )
            self.timer.cancel()
            return


# ── Entry point ───────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LandingMavros()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Landing node interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
