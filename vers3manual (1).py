#!/usr/bin/env python3

import math
import time
from typing import List, Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, Float64
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from builtin_interfaces.msg import Duration

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import (
    RobotState,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    WorkspaceParameters,
    MoveItErrorCodes,
    PlanningScene,
    CollisionObject,
    AttachedCollisionObject,
)

from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh

from visualization_msgs.msg import Marker, MarkerArray
from shape_msgs.msg import SolidPrimitive

from cv_bridge import CvBridge
import cv2
import numpy as np
import tkinter as tk
from threading import Thread

from scipy.spatial.transform import Rotation


class DestinationSpot:
    """Represents a destination spot with position and traits"""
    def __init__(self, x: float, y: float, z: float, trait: str):
        self.x = x
        self.y = y
        self.z = z
        self.trait = trait  # 'red' or 'green'
        self.occupied = False
        self.cube_id = None  # Track which cube is placed here


class DetectedCube:
    """Represents a detected cube with position and classification"""
    def __init__(self, x: float, y: float, z: float, pixel_x: int, pixel_y: int, cube_id: int):
        self.x = x
        self.y = y
        self.z = z
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.cube_id = cube_id  # Simple sequential ID
        self.picked = False
        
        # Classify cube based on Y position
        if y >= 0:  # Positive Y side
            self.color_type = 'green'
        else:  # Negative Y side
            self.color_type = 'red'


class XArm7PickPlaceROS2(Node):
    """
    Pick and Place controller with Manual Gripper
    """
    
    def __init__(self):
        super().__init__('xarm7_pick_place_ros2')

        # === xarm_ros2 specific configuration ===
        self.robot_type = self.declare_parameter('robot_type', 'xarm7').value
        
        # MoveIt planning groups (from xarm_ros2 SRDF)
        self.arm_group = self.robot_type  # "xarm7" or "lite6" 
        
        # Frame names (from xarm_ros2 URDF)
        self.base_frame = 'link_base'
        self.ee_link = 'link_eef'
        
        # xArm7 joint names in order
        self.xarm7_joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 
            'joint5', 'joint6', 'joint7'
        ]
        
        # Physical setup parameters
        self.table_height = 1.0  
        self.object_height = 0.1
        self.gripper_length = 0.15  
        
        # Workspace bounds for xArm7 (relative to robot base on 1m table)
        self.workspace = {
            'x_min': 0.15, 'x_max': 0.7,   # Conservative reach
            'y_min': -0.4, 'y_max': 0.4,
            'z_min': -0.8,  # Can reach down below table (relative to base)
            'z_max': 0.6,   # Can reach up above base
        }

        # Motion planning parameters
        self.max_vel_scale = 0.2  # Slower for safety
        self.max_acc_scale = 0.2
        self.planning_time = 8.0  # More time
        self.planning_attempts = 15  # More attempts

        # Camera mounting parameters
        self.camera_offset = {
            'x': 0.0,      # Camera is at gripper position in X
            'y': 0.6,      # Camera is 0.6m behind in Y  
            'z': 2.0,      # Camera is 2.0m above base
        }
        
        # Camera orientation (pointing down)
        self.camera_pitch = 1.57  # 90 degrees down

        # Destination spots with easily editable traits
        self.destination_spots = [
            DestinationSpot(-0.4, -0.05, 0.1, 'red'),
            DestinationSpot(-0.4, 0.05, 0.1, 'green'),
            DestinationSpot(-0.3, -0.05, 0.1, 'red'), 
            DestinationSpot(-0.3, 0.05, 0.1, 'green'),
        ]
        
        # Detection state for multiple cubes
        self.detected_cubes = []  # List of DetectedCube objects
        self.cube_detection_update_time = 0.0
        self.next_cube_id = 1  # Simple sequential ID counter
        
        # State variables
        self.current_joint_state: Optional[JointState] = None
        self.bridge = CvBridge()
        self.latest_image = None
        self.gripper_closed = False  # Track gripper state
        
        # Detection logging control
        self.last_detection_log_time = 0.0
        self.detection_log_interval = 2.0  # Log every 2 seconds max
        self.is_actively_detecting = False  # Only log when we're actively looking

        # Initialize all components
        self._setup_subscribers()
        self._setup_camera()
        self._setup_moveit_services()
        self._setup_action_clients()
        self._setup_gripper_control()
        self._setup_visualization()
        self._setup_planning_scene()

        self.get_logger().info(f'XArm7 Pick&Place initialized for {self.robot_type} with Manual Gripper')
        self._log_connection_status()

    def _setup_subscribers(self):
        """Setup joint state subscriber"""
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

    def _setup_camera(self):
        """Setup camera subscriber with multiple topic fallbacks"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Common camera topics for xarm_ros2 setups
        camera_topics = [
            '/downward_cam/image_raw',      # Updated for Gazebo Classic
            '/downward_camera/image_raw',   # Downward-looking camera
            '/demo_cam/image_raw',          # Demo camera
            '/camera/color/image_raw',      # RealSense
            '/camera/image_raw',            # Generic USB camera
            '/head_camera/rgb/image_raw',   # Head-mounted camera
            '/usb_cam/image_raw',           # USB camera node
        ]
        
        self.camera_sub = None
        for topic in camera_topics:
            try:
                self.get_logger().info(f'Attempting camera topic: {topic}')
                self.camera_sub = self.create_subscription(
                    Image, topic, self._image_callback, qos)
                
                # Test if frames are arriving
                start_time = time.time()
                while time.time() - start_time < 3.0 and self.latest_image is None:
                    rclpy.spin_once(self, timeout_sec=0.1)
                
                if self.latest_image is not None:
                    self.get_logger().info(f'Camera connected: {topic}')
                    break
                else:
                    self.destroy_subscription(self.camera_sub)
                    
            except Exception as e:
                self.get_logger().warn(f'Failed to connect to {topic}: {e}')
        
        if self.latest_image is None:
            self.get_logger().warn('No camera available - using fallback detection')

    def _setup_moveit_services(self):
        """Setup MoveIt2 service clients"""
        self.get_logger().info('Setting up MoveIt2 services...')
        
        # IK solver service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        
        # Motion planning service  
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        # Wait for services
        services_ready = True
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('IK service /compute_ik not available')
            services_ready = False
        else:
            self.get_logger().info('IK service connected')
            
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Planning service /plan_kinematic_path not available')
            services_ready = False
        else:
            self.get_logger().info('Planning service connected')
            
        if not services_ready:
            self.get_logger().warn('Some MoveIt2 services missing - check launch files')

    def _setup_action_clients(self):
        """Setup action clients for arm control"""
        self.get_logger().info('Setting up action clients...')
        
        # Arm controller (xarm_ros2 naming convention)
        arm_action_topics = [
            '/xarm_controller/follow_joint_trajectory',
            f'/{self.robot_type}_traj_controller/follow_joint_trajectory',
            '/follow_joint_trajectory',
        ]
        
        self.arm_action_client = None
        for topic in arm_action_topics:
            try:
                self.get_logger().info(f'Connecting to arm action: {topic}')
                client = ActionClient(self, FollowJointTrajectory, topic)
                if client.wait_for_server(timeout_sec=5.0):
                    self.arm_action_client = client
                    self.get_logger().info(f'Arm controller connected: {topic}')
                    break
            except Exception as e:
                self.get_logger().warn(f'Failed to connect to {topic}: {e}')
        
        if not self.arm_action_client:
            self.get_logger().error('No arm action server available')

    def _setup_gripper_control(self):
        """Setup gripper action client for trajectory control"""
        self.get_logger().info('Setting up gripper action client...')
        
        # Gripper action client for xarm_gripper_traj_controller
        gripper_action_topics = [
            '/xarm_gripper_traj_controller/follow_joint_trajectory',
            '/gripper_controller/follow_joint_trajectory',
            '/gripper/follow_joint_trajectory'
        ]
        
        self.gripper_action_client = None
        for topic in gripper_action_topics:
            try:
                self.get_logger().info(f'Connecting to gripper action: {topic}')
                client = ActionClient(self, FollowJointTrajectory, topic)
                if client.wait_for_server(timeout_sec=5.0):
                    self.gripper_action_client = client
                    self.get_logger().info(f'Gripper controller connected: {topic}')
                    break
            except Exception as e:
                self.get_logger().warn(f'Failed to connect to {topic}: {e}')
        
        if self.gripper_action_client is None:
            self.get_logger().error('No gripper action server available')
        
        # Gripper position values
        self.gripper_open_position = 0.0   # Fully open
        self.gripper_closed_position = 0.5  # Partially closed (adjust as needed)

    def _setup_visualization(self):
        """Setup RViz visualization markers"""
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.marker_timer = self.create_timer(2.0, self._publish_markers)

    def _setup_planning_scene(self):
        """Setup MoveIt planning scene for collision avoidance"""
        self.get_logger().info('Setting up MoveIt planning scene...')
        
        # Planning scene publisher
        self.planning_scene_pub = self.create_publisher(
            PlanningScene, '/planning_scene', 10)
        
        # Planning scene service clients
        self.get_scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene')
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')
        
        # Wait for services
        if not self.get_scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Planning scene service not available')
        else:
            self.get_logger().info('Planning scene service connected')
        
        # Track collision objects
        self.collision_objects = {}  # Dict of object_id -> CollisionObject
        self.attached_objects = {}   # Dict of object_id -> AttachedCollisionObject
        
        # Add static table to scene
        time.sleep(1.0)  # Wait for scene to be ready
        self._add_table_to_scene()

    def _log_connection_status(self):
        """Log the status of all connections"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('CONNECTION STATUS SUMMARY - ACTION CLIENT GRIPPER SYSTEM')
        self.get_logger().info(f'Robot Type:     {self.robot_type}')
        self.get_logger().info(f'Arm Action:     {"OK" if self.arm_action_client else "MISSING"}')
        self.get_logger().info(f'Gripper Action: {"OK" if self.gripper_action_client else "MISSING"}')
        self.get_logger().info(f'IK Service:     {"OK" if self.ik_client.service_is_ready() else "MISSING"}')
        self.get_logger().info(f'Planner:        {"OK" if self.plan_client.service_is_ready() else "MISSING"}')
        self.get_logger().info(f'Camera:         {"OK" if self.latest_image is not None else "FALLBACK"}')
        self.get_logger().info(f'Dest. Spots:    {len(self.destination_spots)} configured')
        self.get_logger().info('='*60 + '\n')


    # DESTINATION SPOT MANAGEMENT


    def update_destination_traits(self, new_traits: List[str]):
        """Update the traits of destination spots easily"""
        if len(new_traits) != len(self.destination_spots):
            self.get_logger().error(f'Expected {len(self.destination_spots)} traits, got {len(new_traits)}')
            return False
        
        for i, trait in enumerate(new_traits):
            if trait not in ['red', 'green']:
                self.get_logger().error(f'Invalid trait "{trait}". Must be "red" or "green"')
                return False
            self.destination_spots[i].trait = trait
        
        self.get_logger().info(f'Updated destination traits: {new_traits}')
        return True

    def get_available_spots_by_trait(self, trait: str) -> List[DestinationSpot]:
        """Get all unoccupied spots with specified trait"""
        return [spot for spot in self.destination_spots 
                if spot.trait == trait and not spot.occupied]

    def get_next_available_spot(self, trait: str) -> Optional[DestinationSpot]:
        """Get the next available spot with specified trait"""
        available_spots = self.get_available_spots_by_trait(trait)
        return available_spots[0] if available_spots else None


    # CALLBACKS


    def _joint_state_callback(self, msg: JointState):
        """Store latest joint state"""
        self.current_joint_state = msg

    def _image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            
            # Only detect when actively looking for objects
            if self.is_actively_detecting:
                self._detect_all_cubes(cv_image)
                
        except Exception as e:
            self.get_logger().error(f'Image processing failed: {e}')


    # ENHANCED OBJECT DETECTION FOR MULTIPLE CUBES


    def _detect_all_cubes(self, image):
        """Detect all cubes regardless of color using edge detection and shape analysis"""
        if image is None:
            return
            
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Clear previous detections
        current_time = time.time()
        if current_time - self.cube_detection_update_time > 0.5:  # Update every 0.5 seconds
            self.detected_cubes.clear()
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (cube should be reasonably sized)
                if area > 300 and area < 5000:  # Adjust these thresholds as needed
                    # Check if contour is roughly rectangular (cube-like)
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    # Look for rectangular shapes (4 corners) or close to it
                    if len(approx) >= 4 and len(approx) <= 8:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Check aspect ratio (cubes should be roughly square from above)
                        aspect_ratio = float(w) / h
                        if 0.5 <= aspect_ratio <= 2.0:  # Allow some variation
                            # Convert to world coordinates
                            world_x, world_y, world_z = self.pixel_to_world_coordinates(center_x, center_y)
                            
                            # Create detected cube object with simple ID
                            detected_cube = DetectedCube(world_x, world_y, world_z, center_x, center_y, self.next_cube_id)
                            self.detected_cubes.append(detected_cube)
                            self.next_cube_id += 1
            
            self.cube_detection_update_time = current_time
            
            # Log detection results occasionally
            if current_time - self.last_detection_log_time > self.detection_log_interval and self.detected_cubes:
                self._log_detection_results()
                self.last_detection_log_time = current_time

    def _log_detection_results(self):
        """Log current detection results"""
        self.get_logger().info(f'Detected {len(self.detected_cubes)} cubes:')
        for cube in self.detected_cubes:
            status = "PICKED" if cube.picked else "AVAILABLE"
            self.get_logger().info(f'  Cube{cube.cube_id} ({cube.color_type}): '
                                 f'({cube.x:.3f}, {cube.y:.3f}, {cube.z:.3f}) - {status}')

    def pixel_to_world_coordinates(self, pixel_x: int, pixel_y: int) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates using pinhole camera model."""
        # Camera parameters
        image_width = 640
        image_height = 480
        camera_x_offset = 0.0
        camera_y_offset = -0.19
        camera_height = 0.9  # meters above table
    
        # Image center
        c_x = image_width / 2
        c_y = image_height / 2
    
        # Focal lengths in pixel units (estimate these!)
        # You need to calibrate by measuring cube size in pixels at center
        f_x = 425.0  # example value, adjust!
        f_y = 490.0  # example value, adjust!
    
        # Convert pixels to world coordinates (perspective scaling)
        world_x = (pixel_x - c_x) * camera_height / f_x
        world_y = (c_y - pixel_y) * camera_height / f_y  # flip Y
        world_z = 0.08  # cube on table
    
        # Rotate to robot frame
        temp_x = world_x
        world_x = world_y + camera_y_offset
        world_y = -temp_x - camera_x_offset
    
        return world_x, world_y, world_z


    # CUBE SELECTION LOGIC


    def get_next_cube_for_trait(self, trait: str) -> Optional[DetectedCube]:
        """Get the next available cube for the specified trait (red/green)"""
        # Filter cubes by trait and availability
        available_cubes = [cube for cube in self.detected_cubes 
                          if cube.color_type == trait and not cube.picked]
        
        if not available_cubes:
            return None
        
        # Sort by X position (descending - highest X first for consistency)
        available_cubes.sort(key=lambda c: c.x, reverse=True)
        
        return available_cubes[0]

    def mark_cube_as_picked(self, cube_id: int):
        """Mark a cube as picked"""
        for cube in self.detected_cubes:
            if cube.cube_id == cube_id:
                cube.picked = True
                self.get_logger().info(f'Marked Cube{cube_id} as picked')
                break


    # MANUAL GRIPPER INTERFACE


    def open_gripper(self) -> bool:
        """Open the gripper using action client"""
        if self.gripper_action_client is None:
            self.get_logger().warn('No gripper action client available')
            return False
        
        try:
            # Create trajectory goal
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = ['drive_joint']
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = [self.gripper_open_position]
            point.time_from_start = Duration(sec=2, nanosec=0)
            goal.trajectory.points = [point]
            
            #Send goal
            self.get_logger().info('Opening gripper...')
            send_goal_future = self.gripper_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=3.0)
        
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper open goal rejected')
                return False
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        
            if result_future.result() is None:
                self.get_logger().error('Gripper open timeout')
                return False
        
            self.gripper_closed = False
            self.get_logger().info('Gripper opened successfully')
            return True
        
        except Exception as e:
            self.get_logger().error(f'Failed to open gripper: {e}')
            return False
    
    def close_gripper(self) -> bool:
        if self.gripper_action_client is None:
            self.get_logger().warn('No gripper action client available')
            return False
    
        try:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = ['drive_joint']
            
            point = JointTrajectoryPoint()
            point.positions = [self.gripper_closed_position]
            point.time_from_start = Duration(sec=3, nanosec=0)
            goal.trajectory.points = [point]
        
            self.get_logger().info('Closing gripper...')
            send_goal_future = self.gripper_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=3.0)
        
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper close goal rejected')
                return False
        
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=8.0)
        
            if result_future.result() is None:
                self.get_logger().error('Gripper close timeout')
                return False
        
            self.gripper_closed = True
            self.get_logger().info('Gripper closed successfully')
            return True
        
        except Exception as e:
            self.get_logger().error(f'Failed to close gripper: {e}')
            return False


    # MOVEIT MOTION PLANNING


    def _create_quaternion_from_rpy(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Create quaternion from roll-pitch-yaw angles"""
        q = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # [x,y,z,w]
        return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    def compute_inverse_kinematics(self, target_pose: Pose) -> Optional[List[float]]:
        """Compute IK solution for target pose, handling joint count mismatch"""
        if not self.ik_client.service_is_ready():
            self.get_logger().error('IK service not available')
            return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.arm_group
        request.ik_request.ik_link_name = self.ee_link
        request.ik_request.avoid_collisions = True
        
        # Set target pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = target_pose
        request.ik_request.pose_stamped = pose_stamped
        
        # Use current joint state as seed if available
        if self.current_joint_state is not None:
            request.ik_request.robot_state = RobotState()
            request.ik_request.robot_state.joint_state = self.current_joint_state

        # Call IK service
        try:
            future = self.ik_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.result() is None:
                self.get_logger().error('IK service call failed')
                return None
                
            response = future.result()
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().warn(f'IK failed with error code: {response.error_code.val}')
                return None
            
            # Extract only the joint positions we need (7 for xArm7)
            ik_solution = list(response.solution.joint_state.position)
            ik_joint_names = list(response.solution.joint_state.name)
            
            # Filter to get only arm joints (exclude gripper joints)
            arm_positions = []
            for joint_name in self.xarm7_joint_names:
                if joint_name in ik_joint_names:
                    idx = ik_joint_names.index(joint_name)
                    arm_positions.append(ik_solution[idx])
                else:
                    self.get_logger().error(f'Missing joint {joint_name} in IK solution')
                    return None
            
            return arm_positions
            
        except Exception as e:
            self.get_logger().error(f'IK service call exception: {e}')
            return None

    def plan_to_joint_target(self, joint_positions: List[float]):
        """Plan trajectory to joint space target"""
        if not self.plan_client.service_is_ready():
            self.get_logger().error('Planning service not available')
            return None
            
        if self.current_joint_state is None:
            self.get_logger().error('No current joint state available')
            return None
            
        if len(joint_positions) != 7:
            self.get_logger().error(f'Expected 7 joint positions, got {len(joint_positions)}')
            return None

        # Create motion plan request
        request = GetMotionPlan.Request()
        motion_plan_req = MotionPlanRequest()
        
        # Set workspace parameters
        motion_plan_req.workspace_parameters = WorkspaceParameters()
        motion_plan_req.workspace_parameters.header.frame_id = self.base_frame
        motion_plan_req.workspace_parameters.min_corner = Vector3(
            x=self.workspace['x_min'], y=self.workspace['y_min'], z=self.workspace['z_min'])
        motion_plan_req.workspace_parameters.max_corner = Vector3(
            x=self.workspace['x_max'], y=self.workspace['y_max'], z=self.workspace['z_max'])
        
        # Set planning parameters
        motion_plan_req.group_name = self.arm_group
        motion_plan_req.num_planning_attempts = self.planning_attempts
        motion_plan_req.allowed_planning_time = self.planning_time
        motion_plan_req.max_velocity_scaling_factor = self.max_vel_scale
        motion_plan_req.max_acceleration_scaling_factor = self.max_acc_scale
        
        # Set start state (current state)
        motion_plan_req.start_state = RobotState()
        motion_plan_req.start_state.joint_state = self.current_joint_state
        
        # Set goal constraints (joint space)
        goal_constraint = Constraints()
        for i, target_position in enumerate(joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = self.xarm7_joint_names[i]
            joint_constraint.position = float(target_position)
            joint_constraint.tolerance_above = 0.02
            joint_constraint.tolerance_below = 0.02
            joint_constraint.weight = 1.0
            goal_constraint.joint_constraints.append(joint_constraint)
        
        motion_plan_req.goal_constraints = [goal_constraint]
        request.motion_plan_request = motion_plan_req

        # Call planning service
        try:
            future = self.plan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.planning_time + 3.0)
            
            if future.result() is None:
                self.get_logger().error('Motion planning service call failed')
                return None
                
            response = future.result()
            if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(f'Motion planning failed with error: {response.motion_plan_response.error_code.val}')
                return None
                
            return response.motion_plan_response.trajectory
            
        except Exception as e:
            self.get_logger().error(f'Planning service exception: {e}')
            return None

    def plan_to_pose_target(self, x: float, y: float, z: float, 
                          roll: float = 0.0, pitch: float = math.pi, yaw: float = 0.0):
        """Plan trajectory to Cartesian pose target"""
        # Create target pose
        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        target_pose.orientation = self._create_quaternion_from_rpy(roll, pitch, yaw)
        
        self.get_logger().info(f'Planning to pose: ({x:.3f}, {y:.3f}, {z:.3f}) RPY({roll:.2f}, {pitch:.2f}, {yaw:.2f})')
        
        # Step 1: Solve inverse kinematics
        ik_solution = self.compute_inverse_kinematics(target_pose)
        if ik_solution is None:
            self.get_logger().error('Cannot solve IK for target pose')
            return None
        
        # Step 2: Plan to IK solution
        return self.plan_to_joint_target(ik_solution)

    def execute_trajectory(self, trajectory, timeout: float = 30.0) -> bool:
        """Execute planned trajectory using action client"""
        if trajectory is None:
            self.get_logger().error('No trajectory to execute')
            return False
            
        if self.arm_action_client is None:
            self.get_logger().error('Arm action client not available')
            return False

        # Extract joint trajectory
        if hasattr(trajectory, 'joint_trajectory'):
            joint_traj = trajectory.joint_trajectory
        else:
            joint_traj = trajectory

        if not joint_traj.joint_names:
            self.get_logger().error('Trajectory has no joint names')
            return False

        # Create and send goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        
        self.get_logger().info(f'Executing trajectory with {len(joint_traj.points)} waypoints...')
        
        try:
            # Send goal
            send_goal_future = self.arm_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)
            
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Trajectory goal rejected')
                return False
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
            
            if result_future.result() is None:
                self.get_logger().error('Trajectory execution timeout')
                return False
            
            self.get_logger().info('Trajectory executed successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Trajectory execution failed: {e}')
            return False


    # HIGH-LEVEL MOTION COMMANDS


    def move_to_pose(self, x: float, y: float, z: float, 
                    roll: float = 0.0, pitch: float = math.pi, yaw: float = 0.0) -> bool:
        """Move end-effector to specified pose"""
        trajectory = self.plan_to_pose_target(x, y, z, roll, pitch, yaw)
        if trajectory is None:
            return False
        return self.execute_trajectory(trajectory)

    def move_to_safe_home(self) -> bool:
        """Move to a safe home position"""
        # Safe home joint positions for xArm7 (pointing up to avoid table collision)
        home_joints = [math.pi, -0.8, 0.0, 0.5, 0.0, 1.3, 0.0]  # Safer configuration
        return self.move_to_joint_positions(home_joints)
        
    def move_to_safe_start(self) -> bool:
        """Move to a safe home position"""
        # Safe home joint positions for xArm7 (pointing up to avoid table collision)
        home_joints = [0.0, -0.8, 0.0, 0.5, 0.0, 1.3, 0.0]  # Safer configuration
        return self.move_to_joint_positions(home_joints)

    def move_to_joint_positions(self, joint_positions: List[float]) -> bool:
        """Move to specified joint positions"""
        if len(joint_positions) != 7:
            self.get_logger().error(f'Expected 7 joint positions, got {len(joint_positions)}')
            return False
            
        trajectory = self.plan_to_joint_target(joint_positions)
        if trajectory is None:
            return False
        return self.execute_trajectory(trajectory)

    def _add_table_to_scene(self):
        """Add static table collision object to planning scene"""
        self.get_logger().info('Adding table to planning scene...')
        
        # Create collision object for table
        table = CollisionObject()
        table.header.frame_id = self.base_frame
        table.header.stamp = self.get_clock().now().to_msg()
        table.id = 'table'
        
        # Table is a box
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.5, 2.0, 0.02]  # 1.5m x 2.0m x 2cm thick
        
        # Table pose (centered under robot, at specified height)
        table_pose = Pose()
        table_pose.position.x = 0.0
        table_pose.position.y = 0.0
        table_pose.position.z = -self.table_height - 0.01  # Just below table surface
        table_pose.orientation.w = 1.0
        
        table.primitives.append(box)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD
        
        # Publish to planning scene
        self._publish_collision_object(table)
        self.collision_objects['table'] = table
        
        self.get_logger().info('Table added to planning scene')

    def _publish_collision_object(self, collision_object: CollisionObject):
        """Publish collision object to planning scene"""
        planning_scene = PlanningScene()
        planning_scene.world.collision_objects.append(collision_object)
        planning_scene.is_diff = True
        
        # Publish multiple times to ensure it's received
        for _ in range(3):
            self.planning_scene_pub.publish(planning_scene)
            time.sleep(0.1)

    def add_cube_to_scene(self, cube: DetectedCube):
        """Add detected cube as collision object to planning scene"""
        cube_id = f'cube_{cube.cube_id}'
        
        # Check if already exists
        if cube_id in self.collision_objects:
            self.get_logger().debug(f'{cube_id} already in scene')
            return
        
        self.get_logger().info(f'Adding {cube_id} to planning scene at ({cube.x:.3f}, {cube.y:.3f}, {cube.z:.3f})')
        
        # Create collision object
        cube_obj = CollisionObject()
        cube_obj.header.frame_id = self.base_frame
        cube_obj.header.stamp = self.get_clock().now().to_msg()
        cube_obj.id = cube_id
        
        # Cube is a box
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.05]  # 5cm cube
        
        # Cube pose (from detection)
        cube_pose = Pose()
        cube_pose.position.x = cube.x
        cube_pose.position.y = cube.y
        cube_pose.position.z = cube.z
        cube_pose.orientation.w = 1.0
        
        cube_obj.primitives.append(box)
        cube_obj.primitive_poses.append(cube_pose)
        cube_obj.operation = CollisionObject.ADD
        
        # Publish to scene
        self._publish_collision_object(cube_obj)
        self.collision_objects[cube_id] = cube_obj
        
        self.get_logger().info(f'{cube_id} added to planning scene')

    def remove_cube_from_scene(self, cube_id: int):
        """Remove cube collision object from planning scene"""
        obj_id = f'cube_{cube_id}'
        
        if obj_id not in self.collision_objects:
            self.get_logger().debug(f'{obj_id} not in scene, nothing to remove')
            return
        
        self.get_logger().info(f'Removing {obj_id} from planning scene')
        
        # Create removal message
        cube_obj = CollisionObject()
        cube_obj.header.frame_id = self.base_frame
        cube_obj.header.stamp = self.get_clock().now().to_msg()
        cube_obj.id = obj_id
        cube_obj.operation = CollisionObject.REMOVE
        
        # Publish removal
        self._publish_collision_object(cube_obj)
        
        # Remove from tracking
        del self.collision_objects[obj_id]
        
        self.get_logger().info(f'{obj_id} removed from planning scene')

    def attach_cube_to_gripper(self, cube_id: int):
        """Attach cube to gripper (when picked up)"""
        obj_id = f'cube_{cube_id}'
        
        self.get_logger().info(f'Attaching {obj_id} to gripper')
        
        # First remove from scene as separate object
        self.remove_cube_from_scene(cube_id)
        
        # Create attached collision object
        attached_obj = AttachedCollisionObject()
        attached_obj.link_name = self.ee_link  # Attach to end-effector
        attached_obj.object.header.frame_id = self.ee_link
        attached_obj.object.header.stamp = self.get_clock().now().to_msg()
        attached_obj.object.id = obj_id
        
        # Cube shape
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.05]
        
        # Pose relative to end-effector (in gripper)
        cube_pose = Pose()
        cube_pose.position.x = 0.0
        cube_pose.position.y = 0.0
        cube_pose.position.z = 0.08  # Slightly below gripper center
        cube_pose.orientation.w = 1.0
        
        attached_obj.object.primitives.append(box)
        attached_obj.object.primitive_poses.append(cube_pose)
        attached_obj.object.operation = CollisionObject.ADD
        
        # Touch links (allow contact with gripper fingers)
        attached_obj.touch_links = [self.ee_link, 'left_finger', 'right_finger', 'link_tcp']
        
        # Publish attached object
        planning_scene = PlanningScene()
        planning_scene.robot_state.attached_collision_objects.append(attached_obj)
        planning_scene.is_diff = True
        
        for _ in range(3):
            self.planning_scene_pub.publish(planning_scene)
            time.sleep(0.1)
        
        self.attached_objects[obj_id] = attached_obj
        self.get_logger().info(f'{obj_id} attached to gripper')

    def detach_cube_from_gripper(self, cube_id: int, place_pose: Pose):
        """Detach cube from gripper and add as static object at place location"""
        obj_id = f'cube_{cube_id}'
        
        if obj_id not in self.attached_objects:
            self.get_logger().warn(f'{obj_id} not attached, cannot detach')
            return
        
        self.get_logger().info(f'Detaching {obj_id} from gripper')
        
        # Remove attached object
        attached_obj = AttachedCollisionObject()
        attached_obj.object.id = obj_id
        attached_obj.object.operation = CollisionObject.REMOVE
        
        planning_scene = PlanningScene()
        planning_scene.robot_state.attached_collision_objects.append(attached_obj)
        planning_scene.is_diff = True
        
        for _ in range(3):
            self.planning_scene_pub.publish(planning_scene)
            time.sleep(0.1)
        
        del self.attached_objects[obj_id]
        
        # Add as static collision object at place location
        time.sleep(0.2)
        self._add_placed_cube_to_scene(cube_id, place_pose)
        
        self.get_logger().info(f'{obj_id} detached and added at destination')

    def _add_placed_cube_to_scene(self, cube_id: int, pose: Pose):
        """Add cube at destination as static collision object"""
        obj_id = f'cube_{cube_id}_placed'
        
        self.get_logger().info(f'Adding {obj_id} to scene at destination')
        
        # Create collision object
        cube_obj = CollisionObject()
        cube_obj.header.frame_id = self.base_frame
        cube_obj.header.stamp = self.get_clock().now().to_msg()
        cube_obj.id = obj_id
        
        # Cube shape
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.05]
        
        cube_obj.primitives.append(box)
        cube_obj.primitive_poses.append(pose)
        cube_obj.operation = CollisionObject.ADD
        
        # Publish to scene
        self._publish_collision_object(cube_obj)
        self.collision_objects[obj_id] = cube_obj

    def add_all_detected_cubes_to_scene(self):
        """Add all currently detected cubes to planning scene"""
        self.get_logger().info(f'Adding {len(self.detected_cubes)} detected cubes to planning scene...')
        
        for cube in self.detected_cubes:
            if not cube.picked:
                self.add_cube_to_scene(cube)
        
        self.get_logger().info('All detected cubes added to planning scene')

    def clear_all_cubes_from_scene(self):
        """Remove all cube collision objects from scene"""
        self.get_logger().info('Clearing all cubes from planning scene...')
        
        cube_ids = [obj_id for obj_id in self.collision_objects.keys() if 'cube' in obj_id]
        
        for obj_id in cube_ids:
            cube_obj = CollisionObject()
            cube_obj.header.frame_id = self.base_frame
            cube_obj.header.stamp = self.get_clock().now().to_msg()
            cube_obj.id = obj_id
            cube_obj.operation = CollisionObject.REMOVE
            
            self._publish_collision_object(cube_obj)
            del self.collision_objects[obj_id]
        
        self.get_logger().info(f'Removed {len(cube_ids)} cubes from planning scene')

    # ENHANCED PICK AND PLACE OPERATIONS


    def pick_specific_cube(self, cube: DetectedCube) -> bool:
        """Pick up a specific detected cube using manual gripper"""
        self.get_logger().info(f'Picking Cube{cube.cube_id} at ({cube.x:.3f}, {cube.y:.3f}, {cube.z:.3f})')

        # Step 1: Open gripper
        if not self.open_gripper():
            self.get_logger().error(f'Failed to open gripper before picking Cube{cube.cube_id}')
            return False
        
        # Step 2: Move to pre-pick position (above object)
        pre_pick_z = cube.z + 0.3  # 30cm above object
        if not self.move_to_pose(cube.x, cube.y, pre_pick_z):
            self.get_logger().error(f'Failed to reach pre-pick position for Cube{cube.cube_id}')
            return False
        
        self.remove_cube_from_scene(cube.cube_id)
        time.sleep(0.2)

        # Step 3: Move down close to object
        approach_z = cube.z + 0.105  # 2cm above object
        if not self.move_to_pose(cube.x, cube.y, approach_z):
            self.get_logger().error(f'Failed to reach approach position for Cube{cube.cube_id}')
            return False

        
        # Step 4: Close gripper to grasp object
        if not self.close_gripper():
            self.get_logger().error(f'Failed to close gripper on Cube{cube.cube_id}')
            return False

        self.attach_cube_to_gripper(cube.cube_id)
        time.sleep(0.3)
        
        # Step 5: Lift object up
        if not self.move_to_pose(cube.x, cube.y, pre_pick_z):
            self.get_logger().error(f'Failed to lift Cube{cube.cube_id}')
            return False
        
        # Mark cube as picked
        cube.picked = True
        self.get_logger().info(f'Cube{cube.cube_id} picked successfully')
        return True

    def place_object_at_spot(self, spot: DestinationSpot, cube_id: int) -> bool:
        """Place currently held object at specified destination spot"""
        
        self.get_logger().info(f'Placing Cube{cube_id} at {spot.trait} spot ({spot.x:.3f}, {spot.y:.3f}, {spot.z:.3f})')
        
        # Step 1: Move to home position first
        if not self.move_to_safe_home():
            self.get_logger().error('Failed to move to home position')
            return False
        
        # Step 2: Move to pre-place position (above drop location)
        pre_place_z = spot.z + 0.3  # 30cm above surface
        if not self.move_to_pose(spot.x, spot.y, pre_place_z):
            self.get_logger().error('Failed to reach pre-place position')
            return False
        
        # Step 3: Move down to place position
        place_z = spot.z + 0.15  # 5cm above surface
        if not self.move_to_pose(spot.x, spot.y, place_z):
            self.get_logger().error('Failed to reach place position')
            return False

        place_pose = Pose()
        place_pose.position.x = spot.x
        place_pose.position.y = spot.y
        place_pose.position.z = spot.z
        place_pose.orientation.w = 1.0
        
        # Step 4: Open gripper to release object
        time.sleep(1.0)  # Brief pause before release
        if not self.open_gripper():
            self.get_logger().error('Failed to open gripper to release object')
            return False
        
        # Step 5: Mark spot as occupied
        spot.occupied = True
        spot.cube_id = cube_id
        
        # Step 6: Move back up
        if not self.move_to_pose(spot.x, spot.y, pre_place_z):
            self.get_logger().error('Failed to retract from place position')
            return False
        
        self.get_logger().info(f'Cube{cube_id} placed successfully at {spot.trait} spot')
        return True

    def perform_complete_sorting_sequence(self) -> bool:
        """Perform complete sorting sequence for all cubes with enhanced detection"""
        self.get_logger().info('Starting complete cube sorting sequence...')
        
        if not self.move_to_safe_start():
            self.get_logger().error('Failed to move to start position')
            return False
            
        # Step 2: Enhanced cube detection with multiple scans
        if not self._enhanced_cube_detection():
            self.get_logger().error('Failed to detect sufficient cubes')
            return False
            
        # Step 1: Move to observation pose
        if not self.move_to_safe_home():
            self.get_logger().error('Failed to move to home position')
            return False
        
        # Step 3: Fill red spots first
        self.get_logger().info('Phase 1: Filling RED destination spots...')
        red_spots_filled = self._fill_spots_by_trait('red')
        
        # Step 4: Fill green spots
        self.get_logger().info('Phase 2: Filling GREEN destination spots...')
        green_spots_filled = self._fill_spots_by_trait('green')
        
        # Step 5: Return to home
        if not self.move_to_safe_home():
            self.get_logger().warn('Failed to return home - but sorting completed')
        
        total_filled = red_spots_filled + green_spots_filled
        self.get_logger().info(f'Sorting sequence completed! Filled {total_filled} spots total')
        self._log_final_status()
        
        return total_filled > 0

    def _enhanced_cube_detection(self) -> bool:
        """Enhanced cube detection with multiple scanning attempts"""
        self.get_logger().info('Starting enhanced cube detection with multiple scans...')
        
        # Enable detection
        self.is_actively_detecting = True
        
        max_cubes_detected = 0
        best_detection_round = 0
        detection_rounds = 1  # Number of detection attempts
        scan_duration = 8.0   # Seconds per scan
        
        for round_num in range(1, detection_rounds + 1):
            self.get_logger().info(f'Detection round {round_num}/{detection_rounds}...')
            
            # Clear previous detections for this round
            self.detected_cubes.clear()
            self.next_cube_id = 1
            
            # Scan for specified duration
            start_time = time.time()
            cube_count_this_round = 0
            
            while time.time() - start_time < scan_duration:
                rclpy.spin_once(self, timeout_sec=0.1)
                
                # Count current detections
                current_count = len([cube for cube in self.detected_cubes if not cube.picked])
                cube_count_this_round = max(cube_count_this_round, current_count)
            
            self.get_logger().info(f'Round {round_num}: Detected {cube_count_this_round} cubes')
            
            # Keep track of best detection round
            if cube_count_this_round > max_cubes_detected:
                max_cubes_detected = cube_count_this_round
                best_detection_round = round_num
                # Store the best detection set (you could save this if needed)
            
            # Brief pause between rounds
            if round_num < detection_rounds:
                time.sleep(2.0)
        
        self.is_actively_detecting = False
        
        # Final detection summary
        unpicked_cubes = [cube for cube in self.detected_cubes if not cube.picked]
        final_count = len(unpicked_cubes)
        
        self.get_logger().info(f'Enhanced detection complete:')
        self.get_logger().info(f'  Best round: {best_detection_round} with {max_cubes_detected} cubes')
        self.get_logger().info(f'  Final count: {final_count} available cubes')
        
        if final_count == 0:
            self.get_logger().warn('No cubes detected after enhanced scanning!')
            return False
        
        # Log all detected cubes
        self._log_detection_results()
        self.add_all_detected_cubes_to_scene()
        return True

    def _fill_spots_by_trait(self, trait: str) -> int:
        """Fill all available spots of specified trait"""
        spots_filled = 0
        available_spots = self.get_available_spots_by_trait(trait)
        
        self.get_logger().info(f'Filling {len(available_spots)} {trait.upper()} spots...')
        
        for spot in available_spots:
            # Get next appropriate cube
            target_cube = self.get_next_cube_for_trait(trait)
            
            if target_cube is None:
                self.get_logger().warn(f'No more {trait} cubes available for spot at ({spot.x:.3f}, {spot.y:.3f})')
                break
            
            # Perform pick and place
            if self._pick_and_place_cube_to_spot(target_cube, spot):
                spots_filled += 1
                self.get_logger().info(f'Successfully placed Cube{target_cube.cube_id} in {trait} spot {spots_filled}')
            else:
                self.get_logger().error(f'Failed to place Cube{target_cube.cube_id} in {trait} spot')
                # Mark cube as picked anyway to avoid retry loops
                target_cube.picked = True
        
        return spots_filled

    def _pick_and_place_cube_to_spot(self, cube: DetectedCube, spot: DestinationSpot) -> bool:
        """Pick a specific cube and place it at a specific spot"""
        # Step 1: Pick up the cube
        if not self.pick_specific_cube(cube):
            return False
        
        # Step 2: Place at destination
        if not self.place_object_at_spot(spot, cube.cube_id):
            return False
        
        return True

    def _log_final_status(self):
        """Log final status of all spots and cubes"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('FINAL SORTING STATUS')
        self.get_logger().info('='*60)
        
        # Log destination spots status
        for i, spot in enumerate(self.destination_spots):
            status = f"OCCUPIED by Cube{spot.cube_id}" if spot.occupied else "EMPTY"
            self.get_logger().info(f'Spot {i+1} ({spot.trait.upper()}): ({spot.x:.3f}, {spot.y:.3f}) - {status}')
        
        # Log cube status
        red_cubes = [cube for cube in self.detected_cubes if cube.color_type == 'red']
        green_cubes = [cube for cube in self.detected_cubes if cube.color_type == 'green']
        
        self.get_logger().info(f'\nRED CUBES (pos Y): {len(red_cubes)} detected, {sum(1 for c in red_cubes if c.picked)} picked')
        self.get_logger().info(f'GREEN CUBES (neg Y): {len(green_cubes)} detected, {sum(1 for c in green_cubes if c.picked)} picked')
        self.get_logger().info('='*60 + '\n')


    # LEGACY METHODS (for backward compatibility)


    def pick_object_at_pixel(self, pixel_x: int, pixel_y: int) -> bool:
        """Legacy method - pick up object at specified pixel coordinates"""
        # Find closest detected cube to these pixel coordinates
        if not self.detected_cubes:
            self.get_logger().error('No cubes detected')
            return False
        
        # Find closest cube
        min_distance = float('inf')
        closest_cube = None
        
        for cube in self.detected_cubes:
            if cube.picked:
                continue
            distance = math.sqrt((cube.pixel_x - pixel_x)**2 + (cube.pixel_y - pixel_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_cube = cube
        
        if closest_cube is None:
            self.get_logger().error('No available cube found near specified pixels')
            return False
        
        return self.pick_specific_cube(closest_cube)

    def place_object_at_position(self, x: float, y: float, z: float = 0.12) -> bool:
        """Legacy method - place object at specified world coordinates"""
        if not self.gripper_closed:
            self.get_logger().error('No object held to place')
            return False
        
        # Create temporary spot
        temp_spot = DestinationSpot(x, y, z, 'temp')
        return self.place_object_at_spot(temp_spot, 0)  # Use dummy cube_id

    def perform_pick_and_place_cycle(self) -> bool:
        """Legacy method - perform a single pick and place cycle"""
        # Use the new complete sorting sequence instead
        return self.perform_complete_sorting_sequence()

    # ============================================================================
    # VISUALIZATION (Enhanced for multiple cubes and spots)
    # ============================================================================

    def _publish_markers(self):
        """Publish enhanced visualization markers for RViz"""
        marker_array = MarkerArray()
        
        # Workspace bounds marker
        workspace_marker = Marker()
        workspace_marker.header.frame_id = self.base_frame
        workspace_marker.header.stamp = self.get_clock().now().to_msg()
        workspace_marker.ns = 'workspace'
        workspace_marker.id = 0
        workspace_marker.type = Marker.CUBE
        workspace_marker.action = Marker.ADD
        
        # Position at center of workspace
        center_x = (self.workspace['x_max'] + self.workspace['x_min']) / 2
        center_y = (self.workspace['y_max'] + self.workspace['y_min']) / 2
        center_z = (self.workspace['z_max'] + self.workspace['z_min']) / 2
        
        workspace_marker.pose.position.x = center_x
        workspace_marker.pose.position.y = center_y
        workspace_marker.pose.position.z = center_z
        workspace_marker.pose.orientation.w = 1.0
        
        # Size of workspace
        workspace_marker.scale.x = self.workspace['x_max'] - self.workspace['x_min']
        workspace_marker.scale.y = self.workspace['y_max'] - self.workspace['y_min']
        workspace_marker.scale.z = self.workspace['z_max'] - self.workspace['z_min']
        
        # Semi-transparent blue
        workspace_marker.color.r = 0.0
        workspace_marker.color.g = 0.0
        workspace_marker.color.b = 1.0
        workspace_marker.color.a = 0.2
        
        marker_array.markers.append(workspace_marker)
        
        # Destination spot markers
        for i, spot in enumerate(self.destination_spots):
            spot_marker = Marker()
            spot_marker.header.frame_id = self.base_frame
            spot_marker.header.stamp = self.get_clock().now().to_msg()
            spot_marker.ns = 'destination_spots'
            spot_marker.id = i + 10
            spot_marker.type = Marker.CYLINDER
            spot_marker.action = Marker.ADD
            
            spot_marker.pose.position.x = spot.x
            spot_marker.pose.position.y = spot.y
            spot_marker.pose.position.z = spot.z - 0.05  # Slightly below to show as platform
            spot_marker.pose.orientation.w = 1.0
            
            spot_marker.scale.x = 0.08  # Diameter
            spot_marker.scale.y = 0.08
            spot_marker.scale.z = 0.02  # Height
            
            # Color based on trait and occupancy
            if spot.trait == 'red':
                spot_marker.color.r = 1.0
                spot_marker.color.g = 0.0
                spot_marker.color.b = 0.0
            else:  # green
                spot_marker.color.r = 0.0
                spot_marker.color.g = 1.0
                spot_marker.color.b = 0.0
            
            # Transparency based on occupancy
            spot_marker.color.a = 0.8 if spot.occupied else 0.4
            
            marker_array.markers.append(spot_marker)
        
        # Detected cube markers
        for i, cube in enumerate(self.detected_cubes):
            if cube.picked:
                continue  # Don't show picked cubes
                
            cube_marker = Marker()
            cube_marker.header.frame_id = self.base_frame
            cube_marker.header.stamp = self.get_clock().now().to_msg()
            cube_marker.ns = 'detected_cubes'
            cube_marker.id = i + 100
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            
            cube_marker.pose.position.x = cube.x
            cube_marker.pose.position.y = cube.y
            cube_marker.pose.position.z = cube.z
            cube_marker.pose.orientation.w = 1.0
            
            cube_marker.scale.x = 0.05
            cube_marker.scale.y = 0.05
            cube_marker.scale.z = 0.05
            
            # Color based on classification
            if cube.color_type == 'red':  # Positive Y
                cube_marker.color.r = 1.0
                cube_marker.color.g = 0.5
                cube_marker.color.b = 0.5
            else:  # Negative Y (green)
                cube_marker.color.r = 0.5
                cube_marker.color.g = 1.0
                cube_marker.color.b = 0.5
            
            cube_marker.color.a = 1.0
            
            marker_array.markers.append(cube_marker)
        
        # Text markers for cube IDs
        for i, cube in enumerate(self.detected_cubes):
            if cube.picked:
                continue
                
            text_marker = Marker()
            text_marker.header.frame_id = self.base_frame
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'cube_labels'
            text_marker.id = i + 200
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = cube.x
            text_marker.pose.position.y = cube.y
            text_marker.pose.position.z = cube.z + 0.1  # Above cube
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.03  # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = f'Cube{cube.cube_id}'
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    # ============================================================================
    # MANUAL CONTROL AND TESTING
    # ============================================================================

    def manual_gripper_test(self) -> bool:
        """Test manual gripper open/close operations"""
        self.get_logger().info('Testing manual gripper operations...')
        
        try:
            # Test open
            self.get_logger().info('Opening gripper...')
            if not self.open_gripper():
                self.get_logger().error('Failed to open gripper')
                return False
            
            time.sleep(2.0)
            
            # Test close
            self.get_logger().info('Closing gripper...')
            if not self.close_gripper():
                self.get_logger().error('Failed to close gripper')
                return False
            
            time.sleep(2.0)
            
            # Open again
            self.get_logger().info('Opening gripper again...')
            if not self.open_gripper():
                self.get_logger().error('Failed to open gripper again')
                return False
            
            self.get_logger().info('Gripper test completed successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Gripper test failed: {e}')
            return False

    def run_manual_tests(self):
        """Run manual gripper and motion tests"""
        self.get_logger().info('Running manual tests for gripper system...')
        
        try:
            # Test 1: Move to home position
            self.get_logger().info('Test 1: Moving to home position')
            if not self.move_to_safe_home():
                self.get_logger().error('Failed to move home')
                return
            
            time.sleep(2.0)
            
            # Test 2: Test gripper
            self.get_logger().info('Test 2: Testing gripper operations')
            if not self.manual_gripper_test():
                self.get_logger().error('Gripper test failed')
                return
            
            # Test 3: Test detection
            self.get_logger().info('Test 3: Testing cube detection')
            if not self._enhanced_cube_detection():
                self.get_logger().warn('Detection test had issues')
            
            # Test 4: Move to various positions
            test_positions = [
                (0.4, 0.2, 0.3),
                (0.4, -0.2, 0.3),
                (0.3, 0.0, 0.2),
            ]
            
            for i, (x, y, z) in enumerate(test_positions):
                self.get_logger().info(f'Test 4.{i+1}: Moving to ({x}, {y}, {z})')
                if not self.move_to_pose(x, y, z):
                    self.get_logger().error(f'Failed to move to position {i+1}')
                    return
                time.sleep(2.0)
            
            # Return home
            self.get_logger().info('Returning home')
            self.move_to_safe_home()
            
            self.get_logger().info('Manual tests completed successfully')
            
        except Exception as e:
            self.get_logger().error(f'Manual tests failed: {e}')

    # ============================================================================
    # CONFIGURATION METHODS
    # ============================================================================

    def set_destination_spots(self, spots_config: List[Dict]):
        """Set destination spots from configuration
        
        Args:
            spots_config: List of dicts with keys: x, y, z, trait
            Example: [{'x': 0.6, 'y': 0.1, 'z': 0.12, 'trait': 'red'}, ...]
        """
        self.destination_spots.clear()
        
        for config in spots_config:
            spot = DestinationSpot(
                config['x'], 
                config['y'], 
                config.get('z', 0.12),  # Default Z
                config['trait']
            )
            self.destination_spots.append(spot)
        
        self.get_logger().info(f'Updated {len(self.destination_spots)} destination spots from configuration')

    def reset_all_spots(self):
        """Reset all destination spots to empty state"""
        for spot in self.destination_spots:
            spot.occupied = False
            spot.cube_id = None
        self.get_logger().info('All destination spots reset to empty')

    def reset_all_cubes(self):
        """Reset all cubes to unpicked state"""
        for cube in self.detected_cubes:
            cube.picked = False
        self.get_logger().info('All cubes reset to unpicked state')

    # ============================================================================
    # MAIN CONTROL INTERFACE
    # ============================================================================

    def run_demo(self):
        """Run demonstration sequence for cube sorting"""
        self.get_logger().info('Starting xArm7 Cube Sorting Demo with Manual Gripper')
        
        try:
            # Log initial configuration
            self._log_initial_configuration()
            
            # Run the complete sorting sequence
            if self.perform_complete_sorting_sequence():
                self.get_logger().info('Cube sorting demo completed successfully!')
            else:
                self.get_logger().error('Cube sorting demo failed')
                
        except KeyboardInterrupt:
            self.get_logger().info('Demo interrupted by user')
        except Exception as e:
            self.get_logger().error(f'Demo failed with exception: {e}')
        
        # Return to safe position
        self.get_logger().info('Returning to safe home position...')
        self.move_to_safe_home()

    def _log_initial_configuration(self):
        """Log the initial configuration of the system"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('CUBE SORTING SYSTEM CONFIGURATION - MANUAL GRIPPER')
        self.get_logger().info('='*60)
        
        # Log destination spots
        self.get_logger().info('Destination Spots:')
        for i, spot in enumerate(self.destination_spots):
            self.get_logger().info(f'  Spot {i+1}: ({spot.x:.3f}, {spot.y:.3f}, {spot.z:.3f}) - {spot.trait.upper()} trait')
        
        # Log sorting strategy
        self.get_logger().info('\nSorting Strategy:')
        self.get_logger().info('  1. Enhanced cube detection with multiple scanning rounds')
        self.get_logger().info('  2. Fill RED spots first using positive Y cubes')
        self.get_logger().info('  3. Fill GREEN spots second using negative Y cubes')
        self.get_logger().info('  4. Use manual gripper for pick and place operations')
        
        # Log gripper settings
        self.get_logger().info('\nGripper Configuration:')
        self.get_logger().info(f'  Open Position:  {self.gripper_open_position}')
        self.get_logger().info(f'  Close Position: {self.gripper_closed_position}')
        self.get_logger().info('='*60 + '\n')

    def run_step_by_step_demo(self):
        """Run step-by-step demo with user confirmation"""
        self.get_logger().info('Starting step-by-step cube sorting demo...')
        
        try:
            # Step 1: Initialize and scan
            self.get_logger().info('STEP 1: Moving to home and scanning for cubes...')
            if not self.move_to_safe_home():
                return False
            
            if not self._enhanced_cube_detection():
                return False
            
            input('Press Enter to continue to Phase 1 (RED spots)...')
            
            # Step 2: Fill red spots
            self.get_logger().info('STEP 2: Filling RED spots...')
            red_filled = self._fill_spots_by_trait('red')
            self.get_logger().info(f'Filled {red_filled} red spots')
            
            input('Press Enter to continue to Phase 2 (GREEN spots)...')
            
            # Step 3: Fill green spots
            self.get_logger().info('STEP 3: Filling GREEN spots...')
            green_filled = self._fill_spots_by_trait('green')
            self.get_logger().info(f'Filled {green_filled} green spots')
            
            # Step 4: Final status
            self.get_logger().info('STEP 4: Returning home and showing final status...')
            self.move_to_safe_home()
            self._log_final_status()
            
            self.get_logger().info('Step-by-step demo completed!')
            
        except KeyboardInterrupt:
            self.get_logger().info('Step-by-step demo interrupted')
        except Exception as e:
            self.get_logger().error(f'Step-by-step demo failed: {e}')

    # ============================================================================
    # UTILITY AND DEBUG METHODS
    # ============================================================================

    def print_current_detections(self):
        """Print current cube detections for debugging"""
        self.get_logger().info('\n--- CURRENT CUBE DETECTIONS ---')
        if not self.detected_cubes:
            self.get_logger().info('No cubes currently detected')
        else:
            for cube in self.detected_cubes:
                status = "PICKED" if cube.picked else "AVAILABLE"
                self.get_logger().info(f'Cube{cube.cube_id} ({cube.color_type}): '
                                     f'World({cube.x:.3f}, {cube.y:.3f}, {cube.z:.3f}) '
                                     f'Pixel({cube.pixel_x}, {cube.pixel_y}) - {status}')
        self.get_logger().info('--- END DETECTIONS ---\n')

    def print_destination_status(self):
        """Print current destination spot status"""
        self.get_logger().info('\n--- DESTINATION SPOT STATUS ---')
        for i, spot in enumerate(self.destination_spots):
            status = f"OCCUPIED by Cube{spot.cube_id}" if spot.occupied else "EMPTY"
            self.get_logger().info(f'Spot {i+1} ({spot.trait.upper()}): '
                                 f'({spot.x:.3f}, {spot.y:.3f}, {spot.z:.3f}) - {status}')
        self.get_logger().info('--- END SPOT STATUS ---\n')

    def simulate_cube_detections_for_testing(self):
        """Create simulated cube detections for testing without camera"""
        self.get_logger().info('Creating simulated cube detections for testing...')
        
        # Clear existing detections
        self.detected_cubes.clear()
        self.next_cube_id = 1
        
        # Create simulated cubes in realistic positions
        simulated_positions = [
            # Negative Y side (green cubes)
            (0.5, -0.3, 0.08),   
            (0.4, -0.25, 0.08),  
            (0.35, -0.2, 0.08),  
            (0.3, -0.15, 0.08),  
            
            # Positive Y side (red cubes)
            (0.5, 0.3, 0.08),    
            (0.4, 0.25, 0.08),   
            (0.35, 0.2, 0.08),   
            (0.3, 0.15, 0.08),   
        ]
        
        # Convert world coordinates back to pixel coordinates for simulation
        for i, (x, y, z) in enumerate(simulated_positions):
            # Rough inverse of pixel_to_world_coordinates
            pixel_x = int(320 + (y - 0.5) * 500)  # Approximate conversion
            pixel_y = int(240 - (-x + 0.03) * 500)  # Approximate conversion
            
            cube = DetectedCube(x, y, z, pixel_x, pixel_y, self.next_cube_id)
            self.detected_cubes.append(cube)
            self.next_cube_id += 1
        
        self.get_logger().info(f'Created {len(self.detected_cubes)} simulated cube detections')
        self.print_current_detections()


class SortingGUI:
    def __init__(self, node: XArm7PickPlaceROS2):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Destination Spot Setup - Manual Gripper")

        # Define colors
        self.colors = ["red", "green"]
        self.current_traits = ["red", "green", "red", "green"]  # initial

        # Create 2x2 grid
        self.buttons = []
        for i in range(4):
            btn = tk.Button(self.root, text=self.current_traits[i].upper(),
                            bg=self.current_traits[i],
                            fg="white", font=("Arial", 20),
                            width=8, height=3,
                            command=lambda idx=i: self.toggle_color(idx))
            btn.grid(row=i // 2, column=i % 2, padx=10, pady=10)
            self.buttons.append(btn)

        # Add Go button
        go_btn = tk.Button(self.root, text="START SORTING", bg="blue", fg="white",
                           font=("Arial", 16), width=20, height=2,
                           command=self.on_go)
        go_btn.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Add test buttons
        test_btn = tk.Button(self.root, text="TEST GRIPPER", bg="orange", fg="white",
                           font=("Arial", 12), width=18, height=1,
                           command=self.on_test_gripper)
        test_btn.grid(row=3, column=0, pady=5)
        
        detect_btn = tk.Button(self.root, text="TEST DETECTION", bg="purple", fg="white",
                           font=("Arial", 12), width=18, height=1,
                           command=self.on_test_detection)
        detect_btn.grid(row=3, column=1, pady=5)

    def toggle_color(self, idx):
        # Toggle between red and green
        current = self.current_traits[idx]
        new_color = "green" if current == "red" else "red"
        self.current_traits[idx] = new_color
        self.buttons[idx].config(bg=new_color, text=new_color.upper())

    def on_go(self):
        # Update traits in robot node
        success = self.node.update_destination_traits(self.current_traits)
        if success:
            print(f"Traits updated: {self.current_traits}")
            # Run robot sequence in a separate thread so GUI doesn't freeze
            Thread(target=self.node.perform_complete_sorting_sequence).start()
        else:
            print("Failed to update traits!")
    
    def on_test_gripper(self):
        # Test gripper functionality
        print("Testing gripper...")
        Thread(target=self.node.manual_gripper_test).start()
    
    def on_test_detection(self):
        # Test detection functionality
        print("Testing detection...")
        Thread(target=self.node._enhanced_cube_detection).start()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = XArm7PickPlaceROS2()
        
        # Wait for initialization
        time.sleep(2.0)
        
        # Choose operation mode
        import sys
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            
            if mode == 'gui':
                # GUI Mode
                gui = SortingGUI(node)
                gui.run()
            elif mode == 'test':
                # Test Mode
                node.run_manual_tests()
            elif mode == 'step':
                # Step-by-step mode
                node.run_step_by_step_demo()
            elif mode == 'simulate':
                # Simulation mode (no camera needed)
                node.simulate_cube_detections_for_testing()
                node.perform_complete_sorting_sequence()
            else:
                # Default demo mode
                node.run_demo()
        else:
            # Default: GUI Mode
            gui = SortingGUI(node)
            gui.run()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Fatal error: {e}')
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
            
