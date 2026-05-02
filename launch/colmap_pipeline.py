#!/usr/bin/env python3
from __future__ import annotations
"""
colmap_pipeline.py

Subscribes to:
    /mono_cam/image_raw          (sensor_msgs/msg/Image)
    /mono_cam/camera_info        (sensor_msgs/msg/CameraInfo)
    /fmu/out/vehicle_odometry    (px4_msgs/msg/VehicleOdometry)

Pipeline:
    1. Records images to colmap_dataset/images/ at a >0.2m baseline.
    2. Stores exact camera poses from PX4 Odometry.
    3. On shutdown (Ctrl+C), generates COLMAP cameras.txt and images.txt priors.
    4. Automatically launches COLMAP to extract, match, and triangulate using priors.

Dependencies:
    pip install "numpy<2" opencv-python
    sudo apt install colmap
"""

import os
import sys
import shutil
import signal
import subprocess
import argparse
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from px4_msgs.msg import VehicleOdometry
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    rclpy = None
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = None
    Image = CameraInfo = VehicleOdometry = None
    CvBridge = None
    ROS_AVAILABLE = False

    class _DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warn(self, *_args, **_kwargs):
            pass

        def fatal(self, *_args, **_kwargs):
            pass

    class Node:
        def __init__(self, *_args, **_kwargs):
            self._logger = _DummyLogger()

        def get_logger(self):
            return self._logger

        def create_subscription(self, *_args, **_kwargs):
            return None

        def create_timer(self, *_args, **_kwargs):
            return None

        def destroy_node(self):
            return None

@dataclass
class ColmapFrame:
    image_name: str
    position: np.ndarray        # [x, y, z] NED
    orientation_q: np.ndarray   # [w, x, y, z] PX4 Body to NED


# Body-frame (X-fwd, Y-right, Z-down) → OpenCV camera (X-right, Y-down, Z-fwd)
_R_BODY_TO_CAM = np.array([
    [0,  1,  0],
    [0,  0,  1],
    [1,  0,  0],
], dtype=np.float64)

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w     ],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w     ],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y ],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    # Normalize to ensure unit length 
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def remap_prior_image_ids_to_db(model_dir: str, db_path: str) -> bool:
    """Rewrite prior images.txt IDs to match COLMAP database IMAGE_ID assignment."""
    images_txt = os.path.join(model_dir, "images.txt")

    if not os.path.exists(images_txt) or not os.path.exists(db_path):
        return False

    with open(images_txt, "r") as f:
        lines = f.readlines()

    pose_lines = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 10:
            pose_lines.append((idx, parts))

    if not pose_lines:
        return False

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, name FROM images")
        db_rows = cursor.fetchall()
    finally:
        conn.close()

    db_id_by_name = {name: image_id for image_id, name in db_rows}

    remapped_entries = []
    for _, parts in pose_lines:
        name = parts[9]
        if name not in db_id_by_name:
            return False
        remapped_entries.append((db_id_by_name[name], parts))

    remapped_entries.sort(key=lambda x: x[0])

    header_lines = [ln for ln in lines if ln.strip().startswith("#")]
    new_lines = list(header_lines)
    for db_image_id, parts in remapped_entries:
        parts[0] = str(db_image_id)
        new_lines.append(" ".join(parts) + "\n")
        new_lines.append("\n")

    with open(images_txt, "w") as f:
        f.writelines(new_lines)

    return True


def has_colmap_model(model_path: str) -> bool:
    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
    txt_files = ["cameras.txt", "images.txt", "points3D.txt"]
    has_bin = all(os.path.exists(os.path.join(model_path, f)) for f in bin_files)
    has_txt = all(os.path.exists(os.path.join(model_path, f)) for f in txt_files)
    return has_bin or has_txt


def points3d_count_from_bin(model_path: str) -> Optional[int]:
    points_bin = os.path.join(model_path, "points3D.bin")
    if not os.path.exists(points_bin):
        return None

    # COLMAP binary points3D starts with uint64 count.
    with open(points_bin, "rb") as f:
        header = f.read(8)
    if len(header) < 8:
        return None
    return int.from_bytes(header, byteorder="little", signed=False)


def database_match_stats(db_path: str) -> tuple[int, int]:
    """Return (matches_count, two_view_geometries_count) from the COLMAP database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        matches_count = int(cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
        two_view_count = int(cursor.fetchone()[0])
    finally:
        conn.close()
    return matches_count, two_view_count


def _pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    max_image_id = 2147483647
    image_id2 = pair_id % max_image_id
    image_id1 = (pair_id - image_id2) // max_image_id
    return int(image_id1), int(image_id2)


def _draw_feature_sample(image: np.ndarray, keypoints_xy: np.ndarray, sample_count: int = 300) -> np.ndarray:
    if keypoints_xy.shape[0] == 0:
        return image

    vis = image.copy()
    step = max(1, keypoints_xy.shape[0] // max(1, sample_count))
    sampled = keypoints_xy[::step][:sample_count]
    for x, y in sampled:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return vis


def _draw_match_sample(
    image1: np.ndarray,
    image2: np.ndarray,
    kp1_xy: np.ndarray,
    kp2_xy: np.ndarray,
    match_idx: np.ndarray,
    sample_count: int = 120,
) -> np.ndarray:
    h = max(image1.shape[0], image2.shape[0])
    w = image1.shape[1] + image2.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: image1.shape[0], : image1.shape[1]] = image1
    canvas[: image2.shape[0], image1.shape[1] : image1.shape[1] + image2.shape[1]] = image2

    if match_idx.shape[0] == 0:
        return canvas

    step = max(1, match_idx.shape[0] // max(1, sample_count))
    sampled = match_idx[::step][:sample_count]
    x_offset = image1.shape[1]

    for idx1, idx2 in sampled:
        i1 = int(idx1)
        i2 = int(idx2)
        if i1 < 0 or i2 < 0 or i1 >= kp1_xy.shape[0] or i2 >= kp2_xy.shape[0]:
            continue
        x1, y1 = kp1_xy[i1]
        x2, y2 = kp2_xy[i2]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)) + x_offset, int(round(y2)))
        cv2.line(canvas, p1, p2, (255, 200, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return canvas


def export_feature_and_match_samples(
    dataset_dir: str,
    db_path: str,
    image_dir: str,
    max_feature_images: int = 6,
    max_match_pairs: int = 6,
) -> Optional[str]:
    """Export small visual samples of features and verified matches from COLMAP DB."""
    samples_root = os.path.join(dataset_dir, "samples")
    features_dir = os.path.join(samples_root, "features")
    matches_dir = os.path.join(samples_root, "matches")

    try:
        if os.path.isdir(samples_root):
            shutil.rmtree(samples_root)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(matches_dir, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT image_id, name FROM images")
        image_rows = cursor.fetchall()
        name_by_image_id = {int(image_id): name for image_id, name in image_rows}

        cursor.execute(
            "SELECT image_id, rows, cols, data FROM keypoints WHERE data IS NOT NULL AND rows > 0 ORDER BY rows DESC LIMIT ?",
            (max_feature_images,),
        )
        feature_rows = cursor.fetchall()

        exported_features = 0
        for image_id, rows, cols, blob in feature_rows:
            if blob is None:
                continue
            rows = int(rows)
            cols = int(cols)
            if rows <= 0 or cols <= 0:
                continue

            kps = np.frombuffer(blob, dtype=np.float32)
            if kps.size < rows * cols:
                continue
            kps = kps.reshape(rows, cols)
            kps_xy = kps[:, :2]

            image_name = name_by_image_id.get(int(image_id))
            if not image_name:
                continue
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            vis = _draw_feature_sample(image, kps_xy)
            out_name = f"{os.path.splitext(image_name)[0]}_features.jpg"
            cv2.imwrite(os.path.join(features_dir, out_name), vis)
            exported_features += 1

        cursor.execute(
            "SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE data IS NOT NULL AND rows > 0 ORDER BY rows DESC LIMIT ?",
            (max_match_pairs,),
        )
        match_rows = cursor.fetchall()

        exported_matches = 0
        for pair_id, rows, cols, blob in match_rows:
            if blob is None:
                continue
            rows = int(rows)
            cols = int(cols)
            if rows <= 0 or cols <= 0:
                continue

            image_id1, image_id2 = _pair_id_to_image_ids(int(pair_id))
            name1 = name_by_image_id.get(image_id1)
            name2 = name_by_image_id.get(image_id2)
            if not name1 or not name2:
                continue

            cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id1,))
            row1 = cursor.fetchone()
            cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id2,))
            row2 = cursor.fetchone()
            if not row1 or not row2 or row1[2] is None or row2[2] is None:
                continue

            kp1 = np.frombuffer(row1[2], dtype=np.float32).reshape(int(row1[0]), int(row1[1]))
            kp2 = np.frombuffer(row2[2], dtype=np.float32).reshape(int(row2[0]), int(row2[1]))
            kp1_xy = kp1[:, :2]
            kp2_xy = kp2[:, :2]

            matches = np.frombuffer(blob, dtype=np.uint32)
            if matches.size < rows * cols:
                continue
            matches = matches.reshape(rows, cols)
            if matches.shape[1] < 2:
                continue

            image1 = cv2.imread(os.path.join(image_dir, name1))
            image2 = cv2.imread(os.path.join(image_dir, name2))
            if image1 is None or image2 is None:
                continue

            vis = _draw_match_sample(image1, image2, kp1_xy, kp2_xy, matches[:, :2])
            stem1 = os.path.splitext(os.path.basename(name1))[0]
            stem2 = os.path.splitext(os.path.basename(name2))[0]
            out_name = f"{stem1}__{stem2}_matches.jpg"
            cv2.imwrite(os.path.join(matches_dir, out_name), vis)
            exported_matches += 1

        conn.close()

        summary_path = os.path.join(samples_root, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"features_exported={exported_features}\n")
            f.write(f"matches_exported={exported_matches}\n")
            f.write(f"features_dir={features_dir}\n")
            f.write(f"matches_dir={matches_dir}\n")

        return samples_root
    except Exception as e:
        print(f"⚠️ Failed to export feature/match samples: {e}")
        return None


def read_prior_camera_model(model_dir: str) -> Optional[str]:
    """Read camera model token from COLMAP cameras.txt (e.g., PINHOLE, SIMPLE_RADIAL)."""
    cameras_txt = os.path.join(model_dir, "cameras.txt")
    if not os.path.exists(cameras_txt):
        return None

    try:
        with open(cameras_txt, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    return parts[1]
    except OSError:
        return None

    return None


def _colmap_option_supported(colmap_bin: str, command: str, option_name: str) -> bool:
    """Check whether a COLMAP subcommand advertises a specific option."""
    try:
        help_proc = subprocess.run(
            [colmap_bin, command, "-h"],
            check=False,
            capture_output=True,
            text=True,
        )
        help_text = (help_proc.stdout or "") + "\n" + (help_proc.stderr or "")
        return option_name in help_text
    except Exception:
        return False


def resolve_colmap_gpu_option_names(colmap_bin: str) -> tuple[str, str]:
    """Return (feature_extractor_gpu_option, matcher_gpu_option)."""
    if _colmap_option_supported(colmap_bin, "feature_extractor", "--FeatureExtraction.use_gpu"):
        feature_gpu_opt = "--FeatureExtraction.use_gpu"
    elif _colmap_option_supported(colmap_bin, "feature_extractor", "--SiftExtraction.use_gpu"):
        feature_gpu_opt = "--SiftExtraction.use_gpu"
    else:
        feature_gpu_opt = "--FeatureExtraction.use_gpu"

    if _colmap_option_supported(colmap_bin, "exhaustive_matcher", "--FeatureMatching.use_gpu"):
        matcher_gpu_opt = "--FeatureMatching.use_gpu"
    elif _colmap_option_supported(colmap_bin, "exhaustive_matcher", "--SiftMatching.use_gpu"):
        matcher_gpu_opt = "--SiftMatching.use_gpu"
    else:
        matcher_gpu_opt = "--FeatureMatching.use_gpu"

    return feature_gpu_opt, matcher_gpu_opt


def run_colmap_matcher(
    colmap_bin: str,
    db_path: str,
    use_gpu: str,
    matcher: str,
    image_path: str,
    matcher_gpu_option: str,
) -> None:
    if matcher == "custom":
        match_list_path = os.path.join(os.path.dirname(db_path), "custom_matches.txt")
        if not os.path.exists(match_list_path):
            raise FileNotFoundError(match_list_path)

        print(">> Running Matches Importer with custom pairs...")
        subprocess.run([
            colmap_bin, "matches_importer",
            "--database_path", db_path,
            "--match_list_path", match_list_path,
            "--match_type", "pairs",
        ], check=True)
        return

    if matcher == "exhaustive":
        print(">> Running Exhaustive Matcher...")
        subprocess.run([
            colmap_bin, "exhaustive_matcher",
            "--database_path", db_path,
            matcher_gpu_option, use_gpu,
        ], check=True)
        return

    raise ValueError(f"Unknown matcher: {matcher}")


def resolve_colmap_model_dir(model_root: str) -> Optional[str]:
    """Return the COLMAP model directory without creating duplicate copies."""
    zero_model = os.path.join(model_root, "0")

    if has_colmap_model(zero_model):
        return zero_model

    if has_colmap_model(model_root):
        return model_root

    return None


def prompt_before_triangulation() -> bool:
    try:
        answer = input("Triangulation fallback is about to run. Proceed? [y/N]: ").strip().lower()
    except EOFError:
        return False

    return answer in {"y", "yes"}


class ColmapCollectorNode(Node):
    COLLECT_INTERVAL_SEC = 1.0
    MIN_BASELINE_M = 0.2

    def __init__(self, dataset_dir: Optional[str] = None, clean_dataset: bool = True):
        super().__init__('colmap_collector')
        self._bridge = CvBridge()

        self._dataset_dir = os.path.abspath(dataset_dir or "colmap_dataset")
        self._images_dir = os.path.join(self._dataset_dir, "images")
        self._model_dir = os.path.join(self._dataset_dir, "sparse", "model")
        
        # Fresh dataset only when explicitly collecting new data.
        if clean_dataset and os.path.exists(self._dataset_dir):
            self.get_logger().info(f'Cleaning old dataset: {self._dataset_dir}')
            shutil.rmtree(self._dataset_dir)
            
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)

        self._frames: List[ColmapFrame] = []
        self._latest_image: Optional[np.ndarray] = None
        self._camera_K: Optional[np.ndarray] = None
        self._img_width = 0
        self._img_height = 0
        self._latest_position = np.zeros(3)
        self._latest_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        self._img_counter = 0

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Image, '/mono_cam/image_raw', self._image_cb, 10)
        self.create_subscription(CameraInfo, '/mono_cam/camera_info', self._caminfo_cb, 10)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, px4_qos)

        self.create_timer(self.COLLECT_INTERVAL_SEC, self._capture_tick)

        self.get_logger().info('📸 Colmap data collector ready! Fly the drone, then press Ctrl+C to trigger matching.')

    def _image_cb(self, msg: Image):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._img_width = msg.width
            self._img_height = msg.height
        except AttributeError as e:
            if "_ARRAY_API" in str(e):
                self.get_logger().fatal("cv_bridge failed due to NumPy 2.x incompatibility! Run: pip install 'numpy<2'")
                sys.exit(1)

    def _caminfo_cb(self, msg: CameraInfo):
        if self._camera_K is None:
            self._camera_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _odom_cb(self, msg: VehicleOdometry):
        self._latest_position = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=np.float64)
        self._latest_orientation = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=np.float64)

    def _capture_tick(self):
        if self._latest_image is None or self._camera_K is None:
            return

        pos = self._latest_position.copy()

        if self._frames:
            baseline = np.linalg.norm(pos - self._frames[-1].position)
            if baseline < self.MIN_BASELINE_M:
                return

        self._img_counter += 1
        img_name = f"img_{self._img_counter:04d}.jpg"
        img_path = os.path.join(self._images_dir, img_name)
        
        cv2.imwrite(img_path, self._latest_image)

        frame = ColmapFrame(
            image_name=img_name,
            position=pos,
            orientation_q=self._latest_orientation.copy()
        )
        self._frames.append(frame)
        
        self.get_logger().info(f'Saved {img_name} | Total valid: {len(self._frames)} frames')

    def generate_colmap_model(self):
        """Converts collected data into Colmap priors on shutdown."""
        if len(self._frames) == 0:
            self.get_logger().warn("No frames collected. Skipping Colmap.")
            return False

        cam_path = os.path.join(self._model_dir, "cameras.txt")
        img_path = os.path.join(self._model_dir, "images.txt")
        pts_path = os.path.join(self._model_dir, "points3D.txt")

        # Create points3D.txt (empty initially)
        with open(pts_path, "w") as f:
            pass

        # Create cameras.txt
        with open(cam_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            # PINHOLE requires: fx, fy, cx, cy
            K = self._camera_K
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            f.write(f"1 PINHOLE {self._img_width} {self._img_height} {fx} {fy} {cx} {cy}\n")

        # Create images.txt
        with open(img_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for i, frame in enumerate(self._frames):
                img_id = i + 1
                
                # PX4: q is w,x,y,z from Body -> NED
                R_body2ned = quat_to_rot(frame.orientation_q)
                
                # Colmap: R_cam is World(NED) -> Camera
                R_cam = _R_BODY_TO_CAM @ R_body2ned.T
                
                # T_cam = -R_cam @ P_world
                T_cam = -R_cam @ frame.position.reshape(3, 1)
                
                # Convert R_cam (3x3) to Quat (Qw, Qx, Qy, Qz)  
                qw, qx, qy, qz = rot_to_quat(R_cam)
                tx, ty, tz = T_cam.flatten()

                f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {frame.image_name}\n")
                f.write("\n")

        # Create custom_matches.txt for efficient Loop Closure
        match_path = os.path.join(self._dataset_dir, "custom_matches.txt")
        num_pairs = 0
        with open(match_path, "w") as f:
            for i in range(len(self._frames)):
                for j in range(i + 1, len(self._frames)):
                    f1 = self._frames[i]
                    f2 = self._frames[j]
                    
                    # 1. Distance check (only match if within 15 meters physical distance)
                    dist = np.linalg.norm(f1.position - f2.position)
                    if dist < 15.0:
                        temporal_neighbor = (j - i) <= 5

                        # 2. View angle check (dot product of camera forward vectors)
                        R1 = quat_to_rot(f1.orientation_q)
                        R2 = quat_to_rot(f2.orientation_q)
                        
                        # PX4 body forward is X [1, 0, 0]
                        v1 = R1[:, 0]
                        v2 = R2[:, 0]
                        
                        # Cosine similarity (dot product of unit vectors)
                        dot = np.dot(v1, v2)
                        
                        # Always connect nearby temporal neighbors; otherwise allow wider view-angle gating.
                        if temporal_neighbor or dot > -0.3:
                            f.write(f"{f1.image_name} {f2.image_name}\n")
                            num_pairs += 1

        self.get_logger().info(f"✅ Generated exact PX4 camera priors in {self._model_dir}")
        self.get_logger().info(f"✅ Generated {num_pairs} custom spatial match pairs for O(N) Loop Closure")
        return True


def run_dense_reconstruction(
    colmap_bin: str,
    dataset_dir: str,
    sparse_model_dir: str,
    dense_mode: str = "normal",
) -> Optional[str]:
    """Run dense reconstruction with graceful failure and memory-tuned modes.
    
    Modes:
      'normal': standard settings (1600x1600 max, cache=4)
      'low-memory': aggressive reduction (800x800 max, cache=2)
    'ultra-low-memory': extreme reduction (400x400 max, cache=1)
    'fusion-only': run stereo_fusion only on existing dense/stereo outputs
    """
    dense_dir = os.path.join(dataset_dir, "dense")
    fusion_only = dense_mode == "fusion-only"

    # Memory tuning by mode
    config = {
        "normal": {"patch_max_size": "1600", "fusion_max_size": "2000", "cache_size": "4"},
        "low-memory": {"patch_max_size": "800", "fusion_max_size": "800", "cache_size": "2"},
        "ultra-low-memory": {"patch_max_size": "400", "fusion_max_size": "400", "cache_size": "1"},
        "fusion-only": {"patch_max_size": "800", "fusion_max_size": "800", "cache_size": "2"},
    }
    cfg = config.get(dense_mode, config["normal"])
    print(f">> Dense mode: {dense_mode} | max_size={cfg['patch_max_size']} cache={cfg['cache_size']}")

    try:
        if fusion_only:
            stereo_dir = os.path.join(dense_dir, "stereo")
            depth_maps_dir = os.path.join(stereo_dir, "depth_maps")
            normal_maps_dir = os.path.join(stereo_dir, "normal_maps")
            if not os.path.isdir(stereo_dir):
                print(f"\n❌ fusion-only mode requires existing dense stereo workspace: {stereo_dir}")
                print("   Run once with --dense-only --dense-mode low-memory (or normal) to generate stereo maps.")
                return None

            has_depth = any(name.endswith(".bin") for name in os.listdir(depth_maps_dir)) if os.path.isdir(depth_maps_dir) else False
            has_normals = any(name.endswith(".bin") for name in os.listdir(normal_maps_dir)) if os.path.isdir(normal_maps_dir) else False
            if not has_depth or not has_normals:
                print("\n❌ fusion-only mode found no existing geometric stereo maps to fuse.")
                print(f"   depth_maps: {depth_maps_dir}")
                print(f"   normal_maps: {normal_maps_dir}")
                print("   Run patch_match_stereo first using a non fusion-only dense mode.")
                return None

            print(">> fusion-only mode: reusing existing stereo depth/normal maps (no undistort, no patch match).")
        else:
            if os.path.isdir(dense_dir):
                shutil.rmtree(dense_dir)
            os.makedirs(dense_dir, exist_ok=True)

            print(">> Running Image Undistorter...")
            subprocess.run([
                colmap_bin, "image_undistorter",
                "--image_path", os.path.join(dataset_dir, "images"),
                "--input_path", sparse_model_dir,
                "--output_path", dense_dir,
                "--output_type", "COLMAP",
            ], check=True)

            print(">> Running Patch Match Stereo...")
            subprocess.run([
                colmap_bin, "patch_match_stereo",
                "--workspace_path", dense_dir,
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true",
                "--PatchMatchStereo.max_image_size", cfg["patch_max_size"],
                "--PatchMatchStereo.num_iterations", "3",
            ], check=True)

        print(">> Running Stereo Fusion...")
        fused_path = os.path.join(dense_dir, "fused.ply")
        subprocess.run([
            colmap_bin, "stereo_fusion",
            "--workspace_path", dense_dir,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", fused_path,
            "--StereoFusion.min_num_pixels", "5",
            "--StereoFusion.max_image_size", cfg["fusion_max_size"],
            "--StereoFusion.cache_size", cfg["cache_size"],
        ], check=True)

        print(f">> Dense reconstruction saved to {fused_path}")
        return fused_path
    
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  Dense reconstruction failed in {dense_mode} mode: {e}")
        if dense_mode != "ultra-low-memory":
            print(f"   Try re-running with --dense-mode ultra-low-memory for more aggressive memory reduction.")
        print("✅ Sparse model and BA snapshots remain available for visualization.")
        return None
    except Exception as e:
        print(f"\n⚠️  Dense reconstruction error: {e}")
        return None

def resolve_colmap_binary() -> str:
    """Resolve COLMAP binary in a portable order for host and container runs."""
    env_colmap_bin = os.environ.get("COLMAP_BIN", "").strip()
    if env_colmap_bin:
        return env_colmap_bin

    local_cuda_build = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "colmap_cuda_src",
            "build_cuda",
            "src",
            "colmap",
            "exe",
            "colmap",
        )
    )
    candidates = [
        local_cuda_build,
        "/opt/colmap/bin/colmap",
        "/usr/local/bin/colmap",
        "/usr/bin/colmap",
    ]

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    path_colmap = shutil.which("colmap")
    if path_colmap:
        return path_colmap

    raise FileNotFoundError("COLMAP binary not found. Set COLMAP_BIN to the executable path.")

def run_colmap(dataset_dir: str, enable_dense: bool = False, dense_mode: str = "normal"):
    print("\n🚀 Starting Automated Colmap Pipeline...\n")
    dataset_dir = os.path.abspath(dataset_dir)
    db_path = os.path.join(dataset_dir, "db.db")
    img_dir = os.path.join(dataset_dir, "images")
    sparse_dir = os.path.join(dataset_dir, "sparse")
    model_dir = os.path.join(sparse_dir, "model")
    reconstruction_dir = os.path.join(sparse_dir, "reconstruction")
    snapshot_dir = os.path.join(sparse_dir, "snapshots")
    match_list_path = os.path.join(dataset_dir, "custom_matches.txt")
    colmap_bin = resolve_colmap_binary()
    prior_camera_model = read_prior_camera_model(model_dir) or "PINHOLE"
    feature_gpu_option, matcher_gpu_option = resolve_colmap_gpu_option_names(colmap_bin)
    use_gpu = "1"
    
    if enable_dense:
        print(f">> Dense reconstruction: ENABLED ({dense_mode} mode)")

    os.makedirs(sparse_dir, exist_ok=True)

    # Keep priors in sparse/model and write the reconstruction into one dedicated folder.
    # Remove legacy output folders so old runs do not accumulate alongside the new layout.
    for legacy_dir in [os.path.join(sparse_dir, "0"), os.path.join(sparse_dir, "triangulated"), reconstruction_dir, snapshot_dir]:
        if os.path.isdir(legacy_dir):
            shutil.rmtree(legacy_dir)
        elif os.path.exists(legacy_dir):
            os.remove(legacy_dir)
    os.makedirs(reconstruction_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    try:
        # Auto-fallback to CPU when COLMAP is built without CUDA support.
        try:
            help_proc = subprocess.run(
                [colmap_bin, "help"],
                check=True,
                capture_output=True,
                text=True,
            )
            if "without CUDA" in help_proc.stdout:
                use_gpu = "0"
        except Exception:
            pass

        print(f">> COLMAP GPU mode: {use_gpu}")
        print(f">> Camera model (from priors): {prior_camera_model}")
        print(f">> Feature extractor GPU option: {feature_gpu_option}")
        print(f">> Matcher GPU option: {matcher_gpu_option}")
        print(f">> Mapper snapshots enabled in: {snapshot_dir}")

        # 1. Wipe old DB and recreate
        if os.path.exists(db_path):
            os.remove(db_path)

        print(">> Creating fresh COLMAP database...")
        subprocess.run([
            colmap_bin, "database_creator",
            "--database_path", db_path,
        ], check=True)

        # 2. Feature Extraction
        print(">> Running Feature Extractor...")
        subprocess.run([
            colmap_bin, "feature_extractor",
            "--database_path", db_path,
            "--image_path", img_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", prior_camera_model,
            "--SiftExtraction.max_num_features", "8000",
            feature_gpu_option, use_gpu
        ], check=True)

        # 3. Try custom pair matching first; fall back to exhaustive matching if it did not verify any pairs.
        if not os.path.exists(match_list_path):
            print(f"\n❌ Missing match list file: {match_list_path}")
            return

        run_colmap_matcher(colmap_bin, db_path, use_gpu, "custom", img_dir, matcher_gpu_option)

        matches_count, two_view_count = database_match_stats(db_path)
        print(f">> Match stats after custom matcher: matches={matches_count}, two_view_geometries={two_view_count}")

        if two_view_count == 0:
            print(">> Custom matcher produced no verified two-view geometries; falling back to exhaustive matching...")
            run_colmap_matcher(colmap_bin, db_path, use_gpu, "exhaustive", img_dir, matcher_gpu_option)
            matches_count, two_view_count = database_match_stats(db_path)
            print(f">> Match stats after exhaustive matcher: matches={matches_count}, two_view_geometries={two_view_count}")

        # Export quick visual artifacts for debugging and reporting.
        samples_dir = export_feature_and_match_samples(dataset_dir, db_path, img_dir)
        if samples_dir:
            print(f">> Saved feature/match samples to: {samples_dir}")

        # 3b. Ensure prior image IDs match COLMAP DB IDs before loading input model.
        if not remap_prior_image_ids_to_db(model_dir, db_path):
            print("\n❌ Failed to remap prior IMAGE_IDs against database image IDs.")
            return

        # 4. Incremental mapper seeded from PX4 priors.
        print(">> Running Mapper...")
        subprocess.run([
            colmap_bin, "mapper",
            "--database_path", db_path,
            "--image_path", img_dir,
            "--input_path", model_dir,
            "--output_path", reconstruction_dir,
            "--Mapper.snapshot_path", snapshot_dir,
            "--Mapper.snapshot_frames_freq", "5",
            "--Mapper.tri_ignore_two_view_tracks", "0",
        ], check=True)

        model_out = resolve_colmap_model_dir(reconstruction_dir)
        if model_out is None:
            print("\n❌ Colmap failed to map any models.")
            return

        point_count = points3d_count_from_bin(model_out)
        if point_count == 0:
            if not prompt_before_triangulation():
                print("\n⏸️ Triangulation skipped by user. Final sparse model remains in:")
                print(f"  {model_out}")
                return

            print("\n⚠️ Incremental mapper produced 0 points. Running point_triangulator fallback...")
            if os.path.isdir(reconstruction_dir):
                shutil.rmtree(reconstruction_dir)
            os.makedirs(reconstruction_dir, exist_ok=True)
            subprocess.run([
                colmap_bin, "point_triangulator",
                "--database_path", db_path,
                "--image_path", img_dir,
                "--input_path", model_out,
                "--output_path", reconstruction_dir,
                "--clear_points", "1",
                "--Mapper.fix_existing_images", "1",
                "--Mapper.tri_ignore_two_view_tracks", "0",
            ], check=True)

            model_out = resolve_colmap_model_dir(reconstruction_dir) or model_out
            point_count = points3d_count_from_bin(model_out)

        if point_count == 0:
            print(f"\n⚠️ No 3D points were triangulated. Final model (single canonical set) is in:\n  {model_out}")
        else:
            print(f"\n✅ SUCCESS! Colmap sparse reconstruction saved to:\n  {model_out}\n   Points: {point_count}")
        print(f"To view it, run: {shutil.which('colmap') or '/usr/bin/colmap'} gui --database_path {db_path} --image_path {img_dir}")
        print(f"To view BA snapshots, import models from: {os.path.join(dataset_dir, 'sparse', 'snapshots')}")

        if enable_dense and point_count and point_count > 0:
            print("\n>> Attempting dense reconstruction...")
            run_dense_reconstruction(colmap_bin, dataset_dir, model_out, dense_mode)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Colmap failed while running: {colmap_bin}\nError: {e}")
    except FileNotFoundError:
        print("\n❌ COLMAP binary not found. Set COLMAP_BIN to the executable path.")


def run_existing_colmap(dataset_dir: str, enable_dense: bool = False, dense_mode: str = "normal"):
    dataset_dir = os.path.abspath(dataset_dir)
    prior_model_dir = os.path.join(dataset_dir, "sparse", "model")

    if not os.path.exists(dataset_dir):
        print(f"\n❌ Dataset directory does not exist: {dataset_dir}")
        return

    if not os.path.exists(prior_model_dir):
        print(f"\n❌ Missing prior model directory: {prior_model_dir}")
        return

    run_colmap(dataset_dir, enable_dense=enable_dense, dense_mode=dense_mode)


def run_dense_only(dataset_dir: str, dense_mode: str = "normal"):
    """Rerun dense reconstruction on existing sparse model (skip mapper)."""
    dataset_dir = os.path.abspath(dataset_dir)
    sparse_dir = os.path.join(dataset_dir, "sparse")
    reconstruction_dir = os.path.join(sparse_dir, "reconstruction")
    model_dir = os.path.join(sparse_dir, "model")
    
    # Check if sparse reconstruction exists
    model_out = resolve_colmap_model_dir(reconstruction_dir)
    if model_out is None:
        print(f"\n❌ No sparse model found in: {reconstruction_dir}")
        print("   Run with --reconstruct-existing first to generate sparse model.")
        return
    
    colmap_bin = resolve_colmap_binary()
    print(f"\n✅ Found existing sparse model at: {model_out}")
    print(f">> Running dense reconstruction only ({dense_mode} mode)...\n")
    run_dense_reconstruction(colmap_bin, dataset_dir, model_out, dense_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="colmap_dataset", help="Dataset root directory")
    parser.add_argument(
        "--reconstruct-existing",
        action="store_true",
        help="Run COLMAP on the existing dataset without starting ROS collection",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Rerun dense reconstruction on existing sparse model (skip mapper)",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Enable dense reconstruction (stereo fusion) after sparse mapping",
    )
    parser.add_argument(
        "--dense-mode",
        choices=["normal", "low-memory", "ultra-low-memory", "fusion-only"],
        default="normal",
        help="Dense mode: normal/low-memory/ultra-low-memory or fusion-only (reuse existing stereo maps)",
    )
    args = parser.parse_args()

    if args.dense_only:
        run_dense_only(args.dataset_dir, dense_mode=args.dense_mode)
        return

    if args.reconstruct_existing:
        run_existing_colmap(args.dataset_dir, enable_dense=args.dense, dense_mode=args.dense_mode)
        return

    if not ROS_AVAILABLE:
        print("\n❌ ROS packages are not available in this shell. Use --reconstruct-existing to run COLMAP on the saved dataset.")
        return

    rclpy.init()
    node = ColmapCollectorNode(dataset_dir=args.dataset_dir, clean_dataset=True)
    
    _shutdown_handled = False

    def handle_shutdown(sig, frame):
        nonlocal _shutdown_handled
        if _shutdown_handled: return
        _shutdown_handled = True
        
        print("\n[Ctrl+C Detected] Stopping data collection...")
        generated_model = node.generate_colmap_model()
        node.destroy_node()
        rclpy.try_shutdown()

        if not generated_model:
            sys.exit(0)

        # Kick off colmap
        run_colmap(args.dataset_dir, enable_dense=args.dense, dense_mode=args.dense_mode)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
