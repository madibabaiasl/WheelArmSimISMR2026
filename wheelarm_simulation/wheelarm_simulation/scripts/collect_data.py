import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/group2/.local/lib/python3.10/site-packages')
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, QuaternionStamped, Vector3Stamped, Pose
from sensor_msgs.msg import JointState, Image
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64
from openpyxl import load_workbook
import threading
import cv2

try:
    import h5py
    print("h5py imported successfully")
except ModuleNotFoundError as e:
    print(f"Error importing h5py: {e}")
    print("Current Python Path:", sys.path)
    print("Current Directory:", os.getcwd())
    raise


class SaveData(Node):
    def __init__(self, labels, instructions):
        super().__init__('Save')

        # ---------- UI / Label Data ----------
        self.labels = labels
        self.instructions = instructions
        self.ui_data_str = ["instructions"]
        self.ui_data = [self.instructions]

        # ---------- Topics to Subscribe ----------
        # JointState topics (one per joint)
        joint_names = [
            "back_left_wheel_joint", "back_right_wheel_joint",
            "front_left_wheel_joint", "front_right_wheel_joint",
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
            "left_inner_knuckle_finger_tip_joint",
            "right_inner_knuckle_finger_tip_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_finger_tip_joint",
            "robotiq_85_right_knuckle_joint"
        ]
        joint_state_topics = [f"/processed_data/joint_state/{jn}" for jn in joint_names]

        # Robot Frames (PoseStamped / TwistStamped)
        frame_names = [
            "wheelchair",
            "shoulder",
            "bicep",
            "forearm",
            "spherical_wrist_1",
            "spherical_wrist_2",
            "end_effector"
        ]
        pose_topics = [f"/processed_data/cartesian_pose/{fn}" for fn in frame_names]
        twist_topics = [f"/processed_data/cartesian_velocity/{fn}" for fn in frame_names]

        # IMU topics
        imu_topics = [
            "/processed_data/imu/orientation",
            "/processed_data/imu/linear_acceleration",
            "/processed_data/imu/angular_velocity"
        ]

        # Camera topics
        camera_topics = [
            "/processed_data/cameras/rgb/gen3",
            "/processed_data/cameras/depth/gen3",
            "/processed_data/cameras/rgb/wheelchair",
            "/processed_data/cameras/depth/wheelchair"
        ]

        # Additional topics
        additional_topics = [
            "/processed_data/timesteps",      # Clock
            "/processed_data/desired_pose"    # Pose
        ]

        # Combine them all
        self.subscribed_topics_str = (
            joint_state_topics
            + pose_topics
            + twist_topics
            + imu_topics
            + camera_topics
            + additional_topics
        )

        # For convenience, store them in self.topic_name as well (with UI fields)
        self.topic_name = self.ui_data_str + self.subscribed_topics_str

        # ---------- Prepare HDF5 File ----------
        self.data_dict = {}

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)  # Ensure the 'data' directory exists
        self.file_path = os.path.join(data_dir, f"{labels}.hdf5")

        self.setup_hdf5_file()

        # ---------- Create Subscribers ----------
        for topic in self.subscribed_topics_str:
            msg_type = self.get_message_type_for_topic(topic)
            self.create_subscription(
                msg_type,
                topic,
                lambda msg, t=topic: self.listener_callback(msg, t),
                10
            )

        self.shutdown_event = threading.Event()

    def get_message_type_for_topic(self, topic):
        """Helper to decide which msg type to use for a given topic."""
        if topic.startswith("/processed_data/joint_state/"):
            return JointState
        elif topic.startswith("/processed_data/cartesian_pose/"):
            return PoseStamped
        elif topic.startswith("/processed_data/cartesian_velocity/"):
            return TwistStamped
        elif topic.startswith("/processed_data/imu/orientation"):
            return QuaternionStamped
        elif topic.startswith("/processed_data/imu/linear_acceleration"):
            return Vector3Stamped
        elif topic.startswith("/processed_data/imu/angular_velocity"):
            return Vector3Stamped
        elif topic.startswith("/processed_data/cameras/rgb/") or topic.startswith("/processed_data/cameras/depth/"):
            return Image
        elif topic == "/processed_data/timesteps":
            return Clock
        elif topic == "/processed_data/desired_pose":
            return Pose
        else:
            return Float64

    def setup_hdf5_file(self):
        try:
            self.hdf5_file = h5py.File(self.file_path, 'w')
            self.hdf5_file.create_group('data')
            self.get_logger().info(f"HDF5 file created at: {self.file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to create HDF5 file: {e}")
            raise

    # -------------------------------------------------
    # listener_callback (Main Subscription Handler)
    # -------------------------------------------------
    def listener_callback(self, msg, topic_name):
        """
        Store each incoming message in self.data_dict. For JointState, separate
        the string (joint_name) from numeric data (position, velocity).
        Once all *subscribed topics* are present at least once, call append_data().
        """

        # 1) JOINT STATES
        if topic_name.startswith("/processed_data/joint_state/"):
            # Single-element JointState => name[0], position[0], velocity[0]
            if len(msg.name) > 0 and len(msg.position) > 0:
                joint_name = msg.name[0]
                pos_val = msg.position[0] if len(msg.position) > 0 else 0.0
                vel_val = msg.velocity[0] if len(msg.velocity) > 0 else 0.0

                # Store numeric data only in main dict key
                # => no string/float mixing
                timestamp =  msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                self.data_dict[topic_name] = [timestamp, pos_val, vel_val]

                # Store the joint name as a separate key => purely string
                # So it doesn't break h5py's numeric dataset
                name_key = f"{topic_name}_joint_name"
                self.data_dict[name_key] = joint_name
            else:
                self.data_dict[topic_name] = []

        # 2) CARTESIAN POSE (PoseStamped)
        elif topic_name.startswith("/processed_data/cartesian_pose/"):
            p = msg.pose.position
            o = msg.pose.orientation
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.data_dict[topic_name] = [timestamp, p.x, p.y, p.z, o.x, o.y, o.z, o.w]

        # 3) CARTESIAN VELOCITY (TwistStamped)
        elif topic_name.startswith("/processed_data/cartesian_velocity/"):
            lin = msg.twist.linear
            # optionally store angular
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.data_dict[topic_name] = [timestamp, lin.x, lin.y, lin.z]

        # 4) IMU ORIENTATION (QuaternionStamped)
        elif topic_name == "/processed_data/imu/orientation":
            q = msg.quaternion
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.data_dict[topic_name] = [timestamp, q.x, q.y, q.z, q.w]

        # 5) IMU LINEAR ACC (Vector3Stamped)
        elif topic_name == "/processed_data/imu/linear_acceleration":
            v = msg.vector
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.data_dict[topic_name] = [timestamp, v.x, v.y, v.z]

        # 6) IMU ANGULAR VEL (Vector3Stamped)
        elif topic_name == "/processed_data/imu/angular_velocity":
            v = msg.vector
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.data_dict[topic_name] = [timestamp, v.x, v.y, v.z]

        # 7) CAMERA (Image)
        elif topic_name.startswith("/processed_data/cameras/rgb/") or topic_name.startswith("/processed_data/cameras/depth/"):
            image_list = list(msg.data)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.data_dict[topic_name+"time"] = timestamp
            self.data_dict[topic_name] = image_list

        # 8) TIMESTEPS (Clock)
        elif topic_name == "/processed_data/timesteps":
            # The Clock message has builtin_interfaces/Time named "clock".
            sec = msg.clock.sec
            nsec = msg.clock.nanosec
            float_time = sec + nsec * 1e-9
            self.data_dict[topic_name] = float_time

        # 9) DESIRED POSE (Pose)
        elif topic_name == "/processed_data/desired_pose":
            p = msg.position
            o = msg.orientation
            self.data_dict[topic_name] = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

        # ELSE: fallback
        else:
            self.data_dict[topic_name] = str(msg)

        # Also attach UI data every time
        for key, value in zip(self.ui_data_str, self.ui_data):
            self.data_dict[key] = value

        # Check if all subscribed topics have arrived at least once
        if self.have_all_topics():
            # We have at least one message for each subscribed topic => append to file
            self.append_data(self.data_dict)
            # Clear the dictionary so we can catch the next 'batch'
            self.data_dict = {}

    def have_all_topics(self):
        """
        Return True if *all* topics in self.subscribed_topics_str
        are present in self.data_dict.
        (Ignore extra keys like '..._joint_name')
        """
        for t in self.subscribed_topics_str:
            if t not in self.data_dict:
                # print("t is", t)
                return False
        return True

    # -------------------------------------------------
    # append_data (Store in HDF5)
    # -------------------------------------------------
    def append_data(self, data_dict):
        """
        Actually writes the collected data into the HDF5 file.
        data_dict may have extra keys (like joint_name).
        """
        try:
            # Also attach UI data (instructions, etc.) one more time
            for key, value in zip(self.ui_data_str, self.ui_data):
                data_dict[key] = value

            data_group = self.hdf5_file['data']
            group_index = str(len(data_group) + 1)  # Start from 1, increment for each new group
            group = data_group.create_group(group_index)

            for topic_name, value in data_dict.items():
                dataset_name = topic_name.replace("/", "_").strip("_")

                # If value is a list or array
                if isinstance(value, list):
                    # Attempt to store as numeric array if possible
                    # (But if it's all strings, it becomes a string array)
                    arr = np.array(value)
                    group.create_dataset(dataset_name, data=arr, compression="gzip")

                elif np.isscalar(value):
                    # single numeric value (or bool, int, float)
                    group.create_dataset(dataset_name, data=value)

                elif isinstance(value, str):
                    # store as attribute
                    group.attrs[dataset_name] = value

                else:
                    self.get_logger().warning(
                        f"Unsupported data type for key {topic_name}: {type(value)}"
                    )

            self.hdf5_file.flush()
            self.get_logger().info(f"Data saved to group {group_index}")

        except Exception as e:
            self.get_logger().error(f"Failed to append data to HDF5: {e}")

    def stop_data_collection(self):
        try:
            self.hdf5_file.close()
            self.get_logger().info("Data collection stopped and all data saved to HDF5.")
        except Exception as e:
            self.get_logger().error(f"Failed during shutdown: {e}")

    def shutdown_thread(self):
        if rclpy.ok():
            rclpy.shutdown()


def start_ros_spin(data_logger_instance, shutdown_event):
    try:
        while not shutdown_event.is_set():
            rclpy.spin_once(data_logger_instance, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        data_logger_instance.destroy_node()


def start_save_data_in_thread(labels, instructions):
    if not rclpy.ok():
        rclpy.init()
    try:
        data_logger_instance = SaveData(labels, instructions)
        shutdown_event = threading.Event()
        ros_thread = threading.Thread(
            target=start_ros_spin,
            args=(data_logger_instance, shutdown_event),
            daemon=True
        )
        ros_thread.start()
        return data_logger_instance, ros_thread, shutdown_event
    except Exception as e:
        print(f"Failed to start SaveData: {e}")
        if rclpy.ok():
            rclpy.shutdown()
        return None, None, None