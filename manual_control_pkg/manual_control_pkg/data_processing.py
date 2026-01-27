import rclpy
from rclpy.node import Node
import tf2_ros

from geometry_msgs.msg import PoseStamped, TwistStamped, QuaternionStamped, Vector3Stamped
from sensor_msgs.msg import JointState, Imu

class DataProcessing(Node):
    def __init__(self):
        super().__init__('data_processing_node')
        
        # Declare tf parameters    
        self.declare_parameter('timer_period', 0.1)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('shoulder_frame', 'shoulder_link')
        self.declare_parameter('bicep_frame', 'bicep_link')
        self.declare_parameter('forearm_frame', 'forearm_link')
        self.declare_parameter('spherical_wrist_1_frame', 'spherical_wrist_1_link')
        self.declare_parameter('spherical_wrist_2_frame', 'spherical_wrist_2_link')
        self.declare_parameter('end_effector_frame', 'robotiq_85_base_link')
        self.declare_parameter('wheelchair_root', 'wheelchair_root')

        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        shoulder_frame = self.get_parameter('shoulder_frame').get_parameter_value().string_value
        bicep_frame = self.get_parameter('bicep_frame').get_parameter_value().string_value
        forearm_frame = self.get_parameter('forearm_frame').get_parameter_value().string_value
        sw1_frame = self.get_parameter('spherical_wrist_1_frame').get_parameter_value().string_value
        sw2_frame = self.get_parameter('spherical_wrist_2_frame').get_parameter_value().string_value
        ee_frame = self.get_parameter('end_effector_frame').get_parameter_value().string_value
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value

        self.frame_list = [
            {
                'name': 'wheelchair',
                'parent': 'house',
                'child': 'wheelchair_root'
                
            },
            {
                'name': 'shoulder',
                'parent': base_frame,
                'child': shoulder_frame
            },
            {
                'name': 'bicep',
                'parent': base_frame,
                'child': bicep_frame
            },
            {
                'name': 'forearm',
                'parent': base_frame,
                'child': forearm_frame
            },
            {
                'name': 'spherical_wrist_1',
                'parent': base_frame,
                'child': sw1_frame
            },
            {
                'name': 'spherical_wrist_2',
                'parent': base_frame,
                'child': sw2_frame
            },
            {
                'name': 'end_effector',
                'parent': base_frame,
                'child': ee_frame
            },
        ]

        # Bookkeeping dictionaries
        self.previous_times = {}
        self.previous_positions = {}
        self.pose_pubs = {}
        self.twist_pubs = {}

        # Cartesian Pose Publisers
        for f in self.frame_list:
            name = f['name']
            self.previous_times[name] = None
            self.previous_positions[name] = None

            # Pose publisher
            pose_topic = f'processed_data/cartesian_pose/{name}'
            self.pose_pubs[name] = self.create_publisher(PoseStamped, pose_topic, 10)

            # Twist publisher
            twist_topic = f'processed_data/cartesian_velocity/{name}'
            self.twist_pubs[name] = self.create_publisher(TwistStamped, twist_topic, 10)

        # Storage for joint states
        self.joint_states = {}

        # Transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Joint State and Imu Subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_states_callback,
            10
        )
        self.imu_subscriber = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10
        )

        # IMU Publishers
        self.orientation_pub = self.create_publisher(
            QuaternionStamped,
            'processed_data/imu/orientation',
            10
        )
        self.lin_acc_pub = self.create_publisher(
            Vector3Stamped,
            'processed_data/imu/linear_acceleration',
            10
        )
        self.ang_vel_pub = self.create_publisher(
            Vector3Stamped,
            'processed_data/imu/angular_velocity',
            10
        )

        # Timer 
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    # -------------------- IMU Callback --------------------
    def imu_callback(self, msg):
        # Orientation
        omsg = QuaternionStamped()
        omsg.header = msg.header
        omsg.quaternion = msg.orientation
        self.orientation_pub.publish(omsg)

        # Linear acceleration
        lamsg = Vector3Stamped()
        lamsg.header = msg.header
        lamsg.vector.x = msg.linear_acceleration.x
        lamsg.vector.y = msg.linear_acceleration.y
        lamsg.vector.z = msg.linear_acceleration.z
        self.lin_acc_pub.publish(lamsg)

        # Angular velocity
        avmsg = Vector3Stamped()
        avmsg.header = msg.header
        avmsg.vector.x = msg.angular_velocity.x
        avmsg.vector.y = msg.angular_velocity.y
        avmsg.vector.z = msg.angular_velocity.z
        self.ang_vel_pub.publish(avmsg)

    # -------------------- Joint States Callback --------------------
    def joint_states_callback(self, msg):
        stamp = msg.header.stamp  
        for i, joint_name in enumerate(msg.name):
            position = msg.position[i] if i < len(msg.position) else 0.0
            velocity = 0.0
            if msg.velocity and i < len(msg.velocity):
                velocity = msg.velocity[i]

            self.joint_states[joint_name] = {'position': position, 'velocity': velocity}

            js_out = JointState()
            js_out.header.stamp = stamp
            js_out.name = [joint_name]
            js_out.position = [position]
            js_out.velocity = [velocity]
            js_out.effort = []

            # Dynamic publisher creation
            topic_name = f'processed_data/joint_state/{joint_name}'
            pub = getattr(self, f'{joint_name}_jointstate_pub', None)
            if pub is None:
                pub = self.create_publisher(JointState, topic_name, 10)
                setattr(self, f'{joint_name}_jointstate_pub', pub)

            pub.publish(js_out)

    # -------------------- Timer Callback --------------------
    def timer_callback(self):
        for f in self.frame_list:
            name = f['name']
            parent_frame = f['parent']
            child_frame = f['child']

            try:
                transform = self.tf_buffer.lookup_transform(
                    parent_frame,
                    child_frame,
                    rclpy.time.Time()  
                )
                current_tf_time = rclpy.time.Time.from_msg(transform.header.stamp)

                # dt from last time
                prev_tf_time = self.previous_times[name]
                if prev_tf_time is not None:
                    dt = (current_tf_time - prev_tf_time).nanoseconds * 1e-9
                else:
                    dt = 0.0
                self.previous_times[name] = current_tf_time

                # PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = transform.header.stamp
                pose_msg.header.frame_id = parent_frame
                pose_msg.pose.position.x = transform.transform.translation.x
                pose_msg.pose.position.y = transform.transform.translation.y
                pose_msg.pose.position.z = transform.transform.translation.z
                pose_msg.pose.orientation = transform.transform.rotation

                self.pose_pubs[name].publish(pose_msg)

                # TwistStamped (velocity)
                twist_msg = TwistStamped()
                twist_msg.header.stamp = transform.header.stamp
                twist_msg.header.frame_id = parent_frame

                prev_pos = self.previous_positions[name]
                if prev_pos is not None and dt > 1e-9:
                    dx = pose_msg.pose.position.x - prev_pos['x']
                    dy = pose_msg.pose.position.y - prev_pos['y']
                    dz = pose_msg.pose.position.z - prev_pos['z']
                    twist_msg.twist.linear.x = dx / dt
                    twist_msg.twist.linear.y = dy / dt
                    twist_msg.twist.linear.z = dz / dt
                else:
                    twist_msg.twist.linear.x = 0.0
                    twist_msg.twist.linear.y = 0.0
                    twist_msg.twist.linear.z = 0.0

                self.previous_positions[name] = {
                    'x': pose_msg.pose.position.x,
                    'y': pose_msg.pose.position.y,
                    'z': pose_msg.pose.position.z
                }

                self.twist_pubs[name].publish(twist_msg)

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f"Transform not available from {parent_frame} to {child_frame}: {e}"
                )

def main(args=None):
    rclpy.init(args=args)
    node = DataProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
