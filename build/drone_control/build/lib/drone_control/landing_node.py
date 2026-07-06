#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped
import time


class LandingNode(Node):

    def __init__(self):
        super().__init__('landing_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers & Publishers
        self.state_sub = self.create_subscription(
            State, '/mavros/state',
            self.state_callback, qos
        )

        self.local_pos_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local', 10
        )

        # Clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.disarm_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        # State Variables
        self.current_state = State()
        self.phase = 'WAIT_CONNECTION'
        self.phase_time = time.time()
        self.last_disarm_time = 0.0
        
        # CHANGED: PX4 uses 'AUTO.LAND' instead of ArduPilot's 'LAND'
        self.land_mode_string = 'AUTO.LAND' 

        # Control Loop (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Landing node started — waiting for MAVROS...')

    def state_callback(self, msg):
        self.current_state = msg

    def publish_zero_setpoint(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        self.local_pos_pub.publish(pose)

    def send_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_client.call_async(req)
        self.get_logger().info(f'[MODE] Requesting: {mode}')

    def send_disarm(self, value):
        now = time.time()
        if now - self.last_disarm_time < 2.0:
            return
        self.last_disarm_time = now
        req = CommandBool.Request()
        req.value = value
        self.disarm_client.call_async(req)
        self.get_logger().info(f'[ARM] Sending arm={value} (Disarming)')

    def set_phase(self, phase):
        self.phase = phase
        self.phase_time = time.time()
        self.get_logger().info(f'======= PHASE: {phase} =======')

    def elapsed(self):
        return time.time() - self.phase_time

    def control_loop(self):
        # PHASE 1: Wait for connection
        if self.phase == 'WAIT_CONNECTION':
            if self.current_state.connected:
                self.get_logger().info('MAVROS connected to FC!')
                self.set_phase('CHECK_ARMED')
            return

        # PHASE 2: Ensure drone is actually flying/armed before trying to land
        if self.phase == 'CHECK_ARMED':
            if self.current_state.armed:
                self.get_logger().info('Drone is armed. Initiating Land Procedure...')
                self.set_phase('SET_LAND_MODE')
            else:
                self.get_logger().warn('Drone is already DISARMED. Nothing to land.', throttle_duration_sec=3.0)
            return

        # PHASE 3: Trigger Auto Land
        if self.phase == 'SET_LAND_MODE':
            self.publish_zero_setpoint()
            if self.current_state.mode != self.land_mode_string:
                self.send_mode(self.land_mode_string)
            else:
                self.set_phase('WAIT_FOR_TOUCHDOWN')
            return

        # PHASE 4: Wait for the Flight Controller to touch ground and auto-disarm
        if self.phase == 'WAIT_FOR_TOUCHDOWN':
            self.publish_zero_setpoint() 
            
            if not self.current_state.armed:
                self.get_logger().info('Touchdown confirmed! Drone Disarmed.')
                self.set_phase('FINISHED')
            else:
                self.get_logger().info(
                    f'Descending... Mode: {self.current_state.mode} | Waiting for touchdown auto-disarm',
                    throttle_duration_sec=2.0
                )
                
                if self.elapsed() > 20.0:
                    self.get_logger().warn('Landing timeout reached. Forcing backup disarm sequence...')
                    self.set_phase('FORCE_DISARM')
            return

        # PHASE 5: Emergency backup disarm
        if self.phase == 'FORCE_DISARM':
            if self.current_state.armed:
                self.send_disarm(False)
            else:
                self.set_phase('FINISHED')
            return

        # PHASE 6: Done
        if self.phase == 'FINISHED':
            self.get_logger().info('Landing sequence complete. Safe to kill node.')
            self.timer.cancel()
            return


def main(args=None):
    rclpy.init(args=args)
    node = LandingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
