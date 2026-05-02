#!/usr/bin/env python3
"""
sfm_pipeline.py  –  Structure-from-Motion pipeline for PX4 + ROS 2

Subscribes to:
    /mono_cam/image_raw          (sensor_msgs/msg/Image)       – camera frames
    /mono_cam/camera_info        (sensor_msgs/msg/CameraInfo)  – intrinsics
    /fmu/out/vehicle_odometry    (px4_msgs/msg/VehicleOdometry)– drone pose

Pipeline:
    1. ROS 2 node collects synchronised (image, pose) pairs
    2. Passes them into `run_sfm()` on a worker thread (non-blocking)
    3. The resulting 3-D point cloud and camera poses are published continuously over ROS 2 topics.
       This headless approach natively supports rendering in rviz2 on WSL without OpenGL/GUI crashes.

Coordinate conventions:
    - PX4 body frame : X-forward, Y-right, Z-down  (NED)
    - Camera frame   : X-right,   Y-down,  Z-forward (OpenCV)
    - The body→camera rotation is a fixed -90° roll about the body X-axis
      composed with a -90° yaw about the resulting Z-axis.

Usage:
    # Terminal 1  – make sure the camera bridge + DDS agent are running
    source /opt/ros/humble/setup.bash
    python3 sfm_pipeline.py

    # View results in Rviz2:
    ros2 run rviz2 rviz2
    # Add PointCloud2 topic: /sfm/pointcloud
    # Add PoseArray topic: /sfm/camera_poses
    # Set Fixed Frame to `map`

Dependencies:
    pip install "numpy<2" opencv-python
    # ROS 2 deps: rclpy, sensor_msgs, geometry_msgs, px4_msgs, cv_bridge
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

# ── ROS 2 ────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

# PX4 odometry
from px4_msgs.msg import VehicleOdometry


# ─────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SfMFrame:
    """One captured frame with its associated drone pose."""
    image: np.ndarray                         # BGR uint8, HxWx3
    timestamp_us: int                         # ROS header stamp in µs
    position: np.ndarray                      # [x, y, z]  NED metres
    orientation_q: np.ndarray                 # [w, x, y, z] quaternion
    camera_matrix: Optional[np.ndarray] = None  # 3×3 intrinsics (K)
    dist_coeffs: Optional[np.ndarray] = None    # distortion vector


@dataclass
class SfMResult:
    """Output of the SfM function, published to ROS."""
    points_3d: np.ndarray          # (N, 3) float64 — world-frame XYZ
    colors: np.ndarray             # (N, 3) float64 — RGB in [0, 1]
    camera_positions: np.ndarray   # (M, 3) float64 — camera centres
    camera_orientations: np.ndarray # (M, 4) float64 — camera orientations [w, x, y, z]


# ─────────────────────────────────────────────────────────────────────
# Quaternion → rotation matrix  (PX4 convention: w, x, y, z)
# ─────────────────────────────────────────────────────────────────────
def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w     ],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w     ],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y ],
    ], dtype=np.float64)


_R_BODY_TO_CAM = np.array([
    [0,  1,  0],
    [0,  0,  1],
    [1,  0,  0],
], dtype=np.float64)


def pose_to_proj(K: np.ndarray, position: np.ndarray, q: np.ndarray) -> np.ndarray:
    R_body2ned = quat_to_rot(q)
    R_cam = _R_BODY_TO_CAM @ R_body2ned.T
    t = -R_cam @ position.reshape(3, 1)
    return K @ np.hstack((R_cam, t))


def _filter_triangulated(pts3d: np.ndarray,
                         colors: np.ndarray,
                         cam_centre: np.ndarray,
                         max_dist: float = 200.0) -> tuple:
    dists = np.linalg.norm(pts3d - cam_centre, axis=1)
    mask = (dists < max_dist) & np.all(np.isfinite(pts3d), axis=1)
    return pts3d[mask], colors[mask]


_MIN_MATCHES = 15
_MAX_POINT_DIST = 200.0


def reprojection_error(P: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """Calculate the reprojection error for 3D points."""
    pts_proj = (P @ np.hstack((pts3d, np.ones((len(pts3d), 1)))).T).T
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
    return np.linalg.norm(pts_proj - pts2d, axis=1)


def run_sfm(frames: List[SfMFrame]) -> Optional[SfMResult]:
    if len(frames) < 2:
        return None

    # Use SIFT instead of ORB for far superior scale and rotation invariance
    detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for i in range(len(frames)):
        for j in range(i + 2, len(frames)):
            f1 = frames[i]
            f2 = frames[j]

            # Enforce an O(N^2) baseline check prior to expensive processing
            if np.linalg.norm(f1.position - f2.position) < 0.2:
                continue

            K = f1.camera_matrix
            if K is None:
                continue

            img1_grey = cv2.cvtColor(f1.image, cv2.COLOR_BGR2GRAY)
            img2_grey = cv2.cvtColor(f2.image, cv2.COLOR_BGR2GRAY)

            kp1, des1 = detector.detectAndCompute(img1_grey, None)
            kp2, des2 = detector.detectAndCompute(img2_grey, None)

            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                continue

            matches = bf.knnMatch(des1, des2, k=2)

            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < _MIN_MATCHES:
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

            # RANSAC Epipolar Filtering
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if mask is None:
                continue
                
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            
            if len(pts1) < _MIN_MATCHES:
                continue

            D = f1.dist_coeffs
            if D is not None and np.any(D != 0):
                pts1_un = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
                pts2_un = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
            else:
                pts1_un = pts1
                pts2_un = pts2

            P1 = pose_to_proj(K, f1.position, f1.orientation_q)
            P2 = pose_to_proj(K, f2.position, f2.orientation_q)

            pts4d = cv2.triangulatePoints(P1, P2, pts1_un.T, pts2_un.T)
            w = pts4d[3]
            good_w = np.abs(w) > 1e-8
            pts4d = pts4d[:, good_w]
            pts1_good = pts1[good_w]
            pts1_un_good = pts1_un[good_w]
            
            # Filter by actual depth (Z axis) from camera 1
            z_cam = (P1 @ pts4d)[2, :] 
            valid_z = (z_cam > 0.1) & (z_cam < 100.0)

            pts4d = pts4d[:, valid_z]
            pts1_good = pts1_good[valid_z]
            pts1_un_good = pts1_un_good[valid_z]
            
            if pts4d.shape[1] == 0:
                continue

            pts3d = (pts4d[:3] / pts4d[3]).T
            
            # Reprojection Error Filtering
            err = reprojection_error(P1, pts3d, pts1_un_good)
            mask_err = err < 2.0
            
            pts3d = pts3d[mask_err]
            pts1_good = pts1_good[mask_err]

            if len(pts3d) == 0:
                continue

            colors = []
            img1 = f1.image
            for p in pts1_good.astype(int):
                x, y = p
                if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
                    bgr = img1[y, x]
                    colors.append(bgr[::-1] / 255.0)
                else:
                    colors.append([1.0, 1.0, 1.0])
            colors = np.array(colors, dtype=np.float64)

            cam_centre = f1.position
            pts3d, colors = _filter_triangulated(
                pts3d, colors, cam_centre, max_dist=_MAX_POINT_DIST)

            if len(pts3d) > 0:
                all_points.append(pts3d)
                all_colors.append(colors)

    if not all_points:
        return None

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    cam_positions = np.array([f.position for f in frames])
    cam_orientations = np.array([f.orientation_q for f in frames])

    return SfMResult(points, colors, cam_positions, cam_orientations)


class SfMCollectorNode(Node):
    COLLECT_INTERVAL_SEC = 1.0
    MIN_FRAMES_FOR_SFM = 5
    MAX_FRAMES = 200
    MIN_BASELINE_M = 0.15

    def __init__(self):
        super().__init__('sfm_collector')
        self._bridge = CvBridge()

        # state
        self._frames: List[SfMFrame] = []
        self._latest_image: Optional[np.ndarray] = None
        self._latest_image_stamp: Optional[int] = None
        self._camera_K: Optional[np.ndarray] = None
        self._camera_D: Optional[np.ndarray] = None
        self._latest_position = np.zeros(3)
        self._latest_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._sfm_running = False

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(
            Image,
            '/mono_cam/image_raw',
            self._image_cb,
            10,
        )

        self.create_subscription(
            CameraInfo,
            '/mono_cam/camera_info',
            self._caminfo_cb,
            10,
        )

        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._odom_cb,
            px4_qos,
        )

        # Rviz2 compatible ROS 2 Publishers (headless!)
        self._pc_pub = self.create_publisher(PointCloud2, '/sfm/pointcloud', 10)
        self._poses_pub = self.create_publisher(PoseArray, '/sfm/camera_poses', 10)

        self.create_timer(self.COLLECT_INTERVAL_SEC, self._capture_tick)

        self.get_logger().info('🛩️  SfM headless mapping ready — waiting for images & odometry …')
        self.get_logger().info('    (Open rviz2, set fixed frame to "map", add PointCloud2 on /sfm/pointcloud and PoseArray on /sfm/camera_poses)')

    def _image_cb(self, msg: Image):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._latest_image_stamp = (
                msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1_000
            )
        except AttributeError as e:
            if "_ARRAY_API" in str(e):
                self.get_logger().fatal("cv_bridge failed due to NumPy 2.x incompatibility!")
                self.get_logger().fatal("Please downgrade numpy by running:  pip install 'numpy<2'")
                sys.exit(1)
            else:
                self.get_logger().error(f"cv_bridge error: {e}")

    def _caminfo_cb(self, msg: CameraInfo):
        if self._camera_K is None:
            self._camera_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._camera_D = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(f'📷  Camera intrinsics received:\n{self._camera_K}')

    def _odom_cb(self, msg: VehicleOdometry):
        self._latest_position = np.array(
            [msg.position[0], msg.position[1], msg.position[2]],
            dtype=np.float64,
        )
        self._latest_orientation = np.array(
            [msg.q[0], msg.q[1], msg.q[2], msg.q[3]],
            dtype=np.float64,
        )

    def _capture_tick(self):
        if self._latest_image is None or self._camera_K is None:
            return

        pos = self._latest_position.copy()

        if self._frames:
            baseline = np.linalg.norm(pos - self._frames[-1].position)
            if baseline < self.MIN_BASELINE_M:
                return

        frame = SfMFrame(
            image=self._latest_image.copy(),
            timestamp_us=self._latest_image_stamp or 0,
            position=pos,
            orientation_q=self._latest_orientation.copy(),
            camera_matrix=self._camera_K.copy(),
            dist_coeffs=self._camera_D.copy() if self._camera_D is not None else None,
        )
        self._frames.append(frame)
        if len(self._frames) > self.MAX_FRAMES:
            self._frames.pop(0)

        n = len(self._frames)
        self.get_logger().info(f'📸  Frame {n} captured  |  pos {np.round(frame.position, 2)}')

        if n >= self.MIN_FRAMES_FOR_SFM and not self._sfm_running:
            self._sfm_running = True
            self.get_logger().info(f'⚙️   Running SfM on {n} frames …')
            snapshot = list(self._frames)
            threading.Thread(
                target=self._run_sfm_thread,
                args=(snapshot,),
                daemon=True,
            ).start()

    def _run_sfm_thread(self, frames: List[SfMFrame]):
        try:
            result = run_sfm(frames)
            if result is not None:
                self._publish_results(result)
                self.get_logger().info(
                    f'✅  SfM generated {result.points_3d.shape[0]} points — published to rviz2'
                )
            else:
                self.get_logger().warn('⚠️  SfM returned no result')
        except Exception as exc:
            self.get_logger().error(f'❌  SfM failed: {exc}')
        finally:
            self._sfm_running = False

    def _publish_results(self, result: SfMResult):
        header = Header()
        header.frame_id = 'map'
        header.stamp = self.get_clock().now().to_msg()

        # 1. Publish PointCloud2
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        r = np.clip(result.colors[:, 0] * 255.0, 0, 255).astype(np.uint32)
        g = np.clip(result.colors[:, 1] * 255.0, 0, 255).astype(np.uint32)
        b = np.clip(result.colors[:, 2] * 255.0, 0, 255).astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b
        
        pts_list = []
        for i in range(len(result.points_3d)):
            pts_list.append((
                float(result.points_3d[i, 0]), 
                float(result.points_3d[i, 1]), 
                float(result.points_3d[i, 2]), 
                int(rgb[i])
            ))

        pc_msg = pc2.create_cloud(header, fields, pts_list)
        self._pc_pub.publish(pc_msg)

        # 2. Publish Camera Poses
        pose_array = PoseArray()
        pose_array.header = header
        for i in range(len(result.camera_positions)):
            pos = result.camera_positions[i]
            q = result.camera_orientations[i]
            
            p = Pose()
            p.position.x = float(pos[0])
            p.position.y = float(pos[1])
            p.position.z = float(pos[2])
            # std_msgs Quaternion is x, y, z, w
            p.orientation.x = float(q[1])
            p.orientation.y = float(q[2])
            p.orientation.z = float(q[3])
            p.orientation.w = float(q[0])
            pose_array.poses.append(p)
            
        self._poses_pub.publish(pose_array)


def main():
    rclpy.init()
    node = SfMCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down …')
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
