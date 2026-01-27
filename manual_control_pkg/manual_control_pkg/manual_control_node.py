# manual_control_pkg/manual_control_pkg/manual_control_node.py
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from pynput import keyboard
import threading
from tf_transformations import quaternion_about_axis, quaternion_multiply
import numpy as np

class ManualControlNode(Node):
    def __init__(self):
        super().__init__('manual_control_node')

        # Declare and initialize parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('position_increment', 0.025),
                ('rotation_increment', 0.05),
                ('gripper_increment', 0.8),
                ('end_effector_frame', 'robotiq_85_base_link'),
                ('base_frame', 'base_link'),
                ('timer_period', 0.1),
                ('bicep2base', [0.0, -5.4 / 1000, 284.8 / 1000]),
                ('min_radius', 410.0 / 1000),
                ('max_radius', 873 / 1000), 
                ('gripper_v_minus', -20.0),
                ('gripper_v_plus', 20.0),
                ('gripper_increment', 0.05)
            ]
        )
        self.position_increment = self.get_parameter('position_increment').get_parameter_value().double_value
        self.rotation_increment = self.get_parameter('rotation_increment').get_parameter_value().double_value
        self.gripper_increment = self.get_parameter('gripper_increment').get_parameter_value().double_value
        self.end_effector_frame = self.get_parameter('end_effector_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.bicep2base = np.array(self.get_parameter('bicep2base').get_parameter_value().double_array_value)
        self.min_radius = self.get_parameter('min_radius').get_parameter_value().double_value
        self.max_radius = self.get_parameter('max_radius').get_parameter_value().double_value
        self.gripper_v_minus = self.get_parameter('gripper_v_minus').get_parameter_value().double_value
        self.gripper_v_plus = self.get_parameter('gripper_v_plus').get_parameter_value().double_value
        self.gripper_increment = self.get_parameter('gripper_increment').get_parameter_value().double_value

        # publisher
        self.publisher_ = self.create_publisher(Pose, 'processed_data/desired_pose', 10)
        self.gripper_publish = self.create_publisher(JointState, 'joint_command', 10)

        # Initialize tf2 to get the current pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize target and current states
        self.current_position = np.zeros(3)
        self.current_orientation = np.zeros(4)
        self.target_position = np.zeros(3)
        self.target_orientation = np.zeros(4)

        # For tracking the first valid transform
        self.flag_t = True

        # Variables for space and gripper state
        self.space_pressed = False
        self.left_finger_p = 0.0
        self.right_finger_p = 0.0
        self.left_finger_v = 0.0
        self.right_finger_v = 0.0

        # keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener_thread = threading.Thread(target=self.listener.start)
        self.listener_thread.start()

        # timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('Manual Control Node has been started.')

    def timer_callback(self):
        self.update_transform()
        self.publish_pose()
        self.gripper_command_publish()

    def update_transform(self):
        """Acquire the current pose of the end effector from TF once it’s available.
           If first time, initialize target pose = current pose."""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                now
            )

            self.current_position = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            self.current_orientation = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])

            # Initialize target to current on first valid reading
            if self.flag_t:
                self.target_position = np.copy(self.current_position)
                self.target_orientation = np.copy(self.current_orientation)
                self.flag_t = False

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            self.get_logger().warn('Transform not available yet.')

    def on_press(self, key):
        """Handle key presses for position increment, rotation, and gripper commands."""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = True

            if hasattr(key, 'char') and key.char is not None:
                c = key.char.lower()

                # Store old pose in case we need to revert
                old_position = np.copy(self.target_position)
                old_orientation = np.copy(self.target_orientation)

                # Handle position or orientation updates
                if c == 'n':  # +X or Rotate around X negative
                    if self.space_pressed:
                        self.rotate_end_effector('x', self.rotation_increment)
                    else:
                        self.target_position[0] += self.position_increment

                elif c == 'v':  # -X or Rotate around X positive
                    if self.space_pressed:
                        self.rotate_end_effector('x', -self.rotation_increment)
                    else:
                        self.target_position[0] -= self.position_increment

                elif c == 'f':  # -Y or Rotate around Y
                    if self.space_pressed:
                        self.rotate_end_effector('y', self.rotation_increment)
                    else:
                        self.target_position[1] += self.position_increment

                elif c == 'h':  # +Y or Rotate around Y
                    if self.space_pressed:
                        self.rotate_end_effector('y', -self.rotation_increment)
                    else:
                        self.target_position[1] -= self.position_increment

                elif c == 'g':  # +Z or Rotate around Z
                    if self.space_pressed:
                        self.rotate_end_effector('z', self.rotation_increment)
                    else:
                        self.target_position[2] += self.position_increment

                elif c == 'b':  # -Z or Rotate around Z
                    if self.space_pressed:
                        self.rotate_end_effector('z', -self.rotation_increment)
                    else:
                        self.target_position[2] -= self.position_increment

                elif c == 'r':  # Close gripper
                    self.left_finger_p = min(0.8, max(0, self.left_finger_p + self.gripper_increment))
                    self.right_finger_p = min(0.0, max(-0.8, self.right_finger_p - self.gripper_increment))
                    self.get_logger().info(f"Gripper command is {self.left_finger_p} and {self.right_finger_p}")

                elif c == 'y':  # Open gripper
                    self.left_finger_p = min(0.8, max(0, self.left_finger_p - self.gripper_increment))
                    self.right_finger_p = min(0.0, max(-0.8, self.right_finger_p + self.gripper_increment))
                    self.get_logger().info(f"Gripper command is {self.left_finger_p} and {self.right_finger_p}")

                # After applying move or rotate, check range
                if not self.is_within_range(self.target_position):
                    # Revert to old valid pose
                    self.get_logger().warn("Target out of range! Reverting to previous valid pose.")
                    self.target_position = old_position
                    self.target_orientation = old_orientation

        except AttributeError:
            self.get_logger().error(f"Error happens in the keyboard control {AttributeError}")

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = False
        elif key == keyboard.Key.esc:
            return False

    def is_within_range(self, position):
        """Check if a given position is within min/max radius from self.bicep2base."""
        end_effector2bicep = position - self.bicep2base
        dist = np.linalg.norm(end_effector2bicep)
        self.get_logger().info(f"distance is {dist}")
        return (self.min_radius < dist < self.max_radius)

    def rotate_end_effector(self, axis, angle):
        """
        Rotate the target_orientation around X/Y/Z by a given angle (in radians)
        using quaternion multiplication to avoid gimbal lock issues.
        """

        # Pick the rotation axis
        if axis == 'x':
            rot_axis = (1.0, 0.0, 0.0)
        elif axis == 'y':
            rot_axis = (0.0, 1.0, 0.0)
        elif axis == 'z':
            rot_axis = (0.0, 0.0, 1.0)
        else:
            self.get_logger().warn(f"Invalid axis '{axis}' for rotate_end_effector. No rotation applied.")
            return

        # Create the incremental rotation quaternion for the given axis & angle
        inc_q = quaternion_about_axis(angle, rot_axis)

        # Typically for local rotation about the end-effector's own X/Y/Z:
        new_q = quaternion_multiply(self.target_orientation, inc_q)

        # Update the target orientation
        self.target_orientation = new_q

    def publish_pose(self):
        """Publish the current valid target pose."""
        target_pose = Pose()

        # Fill orientation
        target_pose.orientation.x = self.target_orientation[0]
        target_pose.orientation.y = self.target_orientation[1]
        target_pose.orientation.z = self.target_orientation[2]
        target_pose.orientation.w = self.target_orientation[3]

        # Convert position to millimeters
        target_pose.position.x = self.target_position[0] * 1000
        target_pose.position.y = self.target_position[1] * 1000
        target_pose.position.z = self.target_position[2] * 1000

        self.publisher_.publish(target_pose)

    def gripper_command_publish(self):
        """Publish the gripper joint velocities."""
        gripper = JointState()
        gripper.name = ['robotiq_85_left_knuckle_joint', 'robotiq_85_right_knuckle_joint']
        gripper.position = [float(self.left_finger_p), float(self.right_finger_p)]
        # gripper.velocity = [float(self.left_finger_v), float(self.right_finger_v)]
        self.gripper_publish.publish(gripper)

    def destroy_node(self):
        """Cleanly stop the listener before destroying the node."""
        super().destroy_node()
        self.listener.stop()
        self.listener_thread.join()

def main(args=None):
    rclpy.init(args=args)
    node = ManualControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

