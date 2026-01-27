#!/usr/bin/env python3
# manual_control_pkg/manual_control_pkg/inverse_kinematics_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from tf_transformations import quaternion_matrix
from scipy.linalg import logm, expm
import numpy as np


class InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__('inverse_kinematics_node')

        self.subscription = self.create_subscription(
            Pose,
            'processed_data/desired_pose',
            self.pose_callback,
            10)
        
        # publish joint commands
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_command', 10)

        # desired pose
        self.desired_p = []
        self.desired_q = []

        # IK with screw theory
        self.S = np.transpose(np.array([[0, 0, -1, 0, 0, 0],
                            [0, 1, 0, -284.4, 0, 0],
                            [0, -1, 0, 694.8, 0, 0],
                            [0, 0, -1, -1, 0, 0],
                            [0, -1, 0, 1009.1, 0, 0],
                            [0, 0, -1, -1, 0, 0]]))
        self.M = np.array([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 1176.5],
                            [0, 0, 0, 1]])
        self.a = [[0, 0, 156.4],[0, -5.4, 284.4],[0, -5.4, 694.8],[0, 1, 903.2],[0, 1, 1009.1], [0, 1, 1115]]
        
        self.Tsd = np.eye((4))
        self.initial_Guess = np.array([0.0, 0.261, 2.27, 0.0, -0.43, -1.5708]) # home positions

        self.bounds = [(-np.pi, np.pi),
                       (-2.2497294, 2.2497294),
                       (-2.5795966, 2.5795966),
                       (-np.pi, np.pi),
                       (-2.0996311, 2.0996311),
                       (-np.pi, np.pi)]

    def skew_symmetry(self, vec):
        skew_matrix = np.array([[ 0, -vec[2], vec[1]],
                                [vec[2], 0, -vec[0]],
                                [-vec[1], vec[0], 0]])
        return skew_matrix
    
    def skew_matrix(self, rot):
        vec = np.array([rot[2,1], rot[0,2], rot[1,0]])
        return vec

    def pose_callback(self, msg):
        # Extract translation
        position = msg.position
        self.desired_p = np.array([[position.x], [position.y], [position.z]])

        # Extract orientation (quaternion) and convert to a rotation matrix
        orientation = msg.orientation
        self.desired_q = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        desired_rot = quaternion_matrix(self.desired_q)[:3, :3]
        
        # Expected homogeneous transformation matrix
        self.Tsd = np.block([[desired_rot, self.desired_p], [0, 0, 0, 1]])

        # inverse kinematics
        NewGuess = self.compute_ik(self.Tsd, self.initial_Guess)
        self.initial_Guess = NewGuess
        joint_state = JointState()
        joint_state.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        joint_state.position = NewGuess
        self.joint_command_publisher.publish(joint_state)
        self.get_logger().info(f'IK solution published. {joint_state}')

    def screw_axis_to_transformation_matrix(self, screw_axis, angle):
        # screw_axis should be np.array([sw1 sw2, sw3, sv1, sv2, sv3])
        assert len(screw_axis) == 6, "Input screw axis must have six components"

        # Extract rotational and translational components from the screw axis
        Sw = screw_axis[:3]
        Sv = screw_axis[3:]

        # Matrix form of the screw axis
        screw_matrix = np.zeros((4, 4))
        screw_matrix[:3, :3] = self.skew_symmetry(Sw)
        screw_matrix[:3, 3] = Sv

        # Exponential map to get the transformation matrix
        exponential_map = expm(angle * screw_matrix)
        
        return exponential_map
    
    def twist_vector_from_twist_matrix(self, twist_matrix):
        assert twist_matrix.shape == (4, 4), "Input matrix must be 4x4"

        w = np.array([twist_matrix[2, 1], twist_matrix[0, 2], twist_matrix[1, 0]])
        v = twist_matrix[:3, 3]

        return np.concatenate((w, v))
    
    def body_jacobian(self, JS, Tb_s):
        R = Tb_s[:3, :3]
        r = Tb_s[:3, 3]
        p = self.skew_symmetry(r)
        pR= p @ R
        ad_tbs = np.block([[R, np.zeros((3, 3))],
                        [pR, R]])
        JB = ad_tbs @ JS
        
        return JB
    
    def Space_Jacobian(self, angles):
        a = self.a
        q = angles
        n = len(q)
        T = np.eye(4,4)
        Ts_b = np.eye(4,4)
        JS = np.zeros((6, n))
        rot = np.array([self.S[:3, 0], self.S[:3, 1], self.S[:3, 2], self.S[:3, 3], self.S[:3, 4], self.S[:3, 5]])
        ad = {}
        tf = {}
        
        for ii in range(n-1,-1,-1):
            rot_hat = self.skew_symmetry(rot[ii])
            e_rot_hat=np.eye(3,3)+rot_hat*np.sin(q[ii])+rot_hat@rot_hat*(1-np.cos(q[ii]))

            if ii>0:
                Sv = -np.transpose(np.cross(rot[ii][:], a[ii][:]))
            elif ii==0:
                Sv = [0, 0, 0]
            
            p = ((np.eye((3))*q[ii]+(1-np.cos(q[ii]))*rot_hat+(q[ii]-np.sin(q[ii]))*rot_hat@rot_hat)@Sv).reshape(3,1)
            e_zai = np.block([[e_rot_hat, p], [0, 0, 0, 1]])
            T = e_zai@T

            tf[ii+1] = e_zai

        Ts_b = T@self.M # from end_effector to the base
        Tb_s = np.linalg.inv(Ts_b)

        for num in range(n):
            tf_cumu = np.eye(4)
            for num_tf in range(num+1):
                tf_cumu = tf_cumu @ tf[num_tf+1]
            R = tf_cumu[:3, :3]
            p = tf_cumu[:3, 3]
            p_hat = self.skew_symmetry(p)

            ad[num] = np.block([
                [R,              np.zeros((3, 3))],
                [p_hat @ R,      R]
            ])
            
        for num_Js in range(n):
            if num_Js == 0:
               JS[:, [num_Js]] = self.S[:, [num_Js]]
            else:
               JS[:, [num_Js]] = ad[num_Js] @ self.S[:, [num_Js]]
                
        return JS, Ts_b, Tb_s
    
    def joints_limit(self, value, lower, upper):
        if lower <= value <= upper:
            return value
        if value < 0:
            return 2 * np.pi * round(value/(2*np.pi)) - value
        return value - 2 * np.pi * round(value/(2*np.pi))

    def compute_ik(self, Tsd, InitGuess):
        for i in range(100):
                    # Calculate the end-effector transform (Tsb) evaluated at the InitGuess using the helper functions that you wrote at the beginning.
                    JS, Ts_b, Tb_s = self.Space_Jacobian(InitGuess) # s is base, b is end-effectorrotation_matrix
                    Tbd = np.linalg.inv(Ts_b) @ Tsd
                    matrix_Vb = logm(Tbd)
                    Vb = self.twist_vector_from_twist_matrix(matrix_Vb).reshape((6, 1))
 
                    # Compute new angles
                    JB = self.body_jacobian(JS, Tb_s)
                    JB_pseudoinv = np.linalg.pinv(JB)
                    
                    NewGuess = InitGuess+(JB_pseudoinv@ Vb).T[0]
                    # print(f"Iteration number: {i} and angles {NewGuess}\n")
                    
                    # Check if you're done and update initial guess
                    # self.get_logger().info(f"error is {np.linalg.norm(abs(NewGuess-InitGuess))}")
                    if(np.linalg.norm(abs(NewGuess-InitGuess)) <= 0.001):
                        for i, (lower, upper) in enumerate(self.bounds):
                            NewGuess[i] = self.joints_limit(NewGuess[i], lower, upper)
                        return [NewGuess[0], NewGuess[1], NewGuess[2], NewGuess[3], NewGuess[4], NewGuess[5]] 
                    else:
                        InitGuess = NewGuess
        print('Numerical solution failed!!')
        NewGuess = self.initial_Guess
        return NewGuess

def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

