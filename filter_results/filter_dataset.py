import os
import shutil
import math
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import pickle
import cv2


INPUT_FOLDER = "/home/liaohaixun/styleswap_data/dateset/vggface2_hq512"
THRESHOLD_YAW = 25.0  # degrees
THRESHOLD_PITCH = 25.0  # degrees
THRESHOLD_EAR = 0.15  # eye aspect ratio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LARGE_ANGLE_FILE = os.path.join(BASE_DIR, "large_angle.txt")
CLOSED_EYES_FILE = os.path.join(BASE_DIR, "closed_eyes.txt")

large_angle_folder = os.path.join(os.path.dirname(__file__), "large_angle")
closed_eyes_folder = os.path.join(os.path.dirname(__file__), "closed_eyes")


# 创建目标文件夹（如果不存在）
os.makedirs(large_angle_folder, exist_ok=True)
os.makedirs(closed_eyes_folder, exist_ok=True)


def estimate_affine_matrix_3d23d(X, Y):
    """Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    """
    X_homo = np.hstack((X, np.ones([X.shape[0], 1])))  # n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T  # Affine matrix. 3 x 4
    return P


def P2sRt(P):
    """decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    """
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    """get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz


class LandMark:
    def __init__(self, device="cuda:0"):
        self.device = device
        model_path = os.path.join(BASE_DIR, "1k3d68.onnx")
        self.lmk_dim = 3
        self.lmk_num = 68

        sess_opt = ort.SessionOptions()
        self.d3d68 = ort.InferenceSession(
            model_path, sess_options=sess_opt, providers=["CUDAExecutionProvider"]
        )

        input_info = self.d3d68.get_inputs()[0]
        self.input_name, self.input_shape = input_info.name, input_info.shape
        self.input_size = tuple(self.input_shape[2:4][::-1])

        self.input_mean = 0.0
        self.input_std = 1.0

        with open(os.path.join(BASE_DIR, "meanshape_68.pkl"), "rb") as f:
            self.mean_lmk = pickle.load(f)

    def detect(self, image_path):
        aimg = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)

        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )

        pred = self.d3d68.run(None, {self.input_name: blob})[0][0]

        pred = pred.reshape((-1, 3))

        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1 :, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= self.input_size[0] // 2
        if pred.shape[1] == 3:
            pred[:, 2] *= self.input_size[0] // 2

        return pred

    def pose(self, lmk):

        P = estimate_affine_matrix_3d23d(self.mean_lmk, lmk)
        s, R, t = P2sRt(P)
        rx, ry, rz = matrix2angle(R)
        pose = np.array([rx, ry, rz], dtype=np.float32)
        return {
            "pitch": pose[0],
            "yaw": pose[1],
            "roll": pose[2],
        }


def plot_3d_coords(coords):
    """
    Visualize a (N,3) array of 3D coordinates, annotating each point with its index.

    Parameters:
    coords (np.ndarray): Array of shape (N, 3).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2]

    # Scatter plot
    ax.scatter(xs, ys, zs)

    # Annotate each point with its index
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i), fontsize=8)

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def eye_aspect_ratio_3d(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """
    计算单只眼睛的 3D EAR。

    参数:
        landmarks: np.ndarray, shape=(68,3)，3D 面部关键点
        eye_indices: list of 6 ints, 对应 [p1,p2,p3,p4,p5,p6] 索引

    返回:
        EAR 值 (float)
    """
    # 取出 6 个关键点
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    # 计算两组垂直距离
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    # 计算水平距离
    horiz = np.linalg.norm(p1 - p4)
    # EAR
    return (vert1 + vert2) / (2.0 * horiz)


def both_eyes_ear(landmarks: np.ndarray) -> tuple[float, float]:
    """
    计算左右眼的 EAR。

    返回:
        (left_EAR, right_EAR)
    """
    left_idx = [36, 37, 38, 39, 40, 41]
    right_idx = [42, 43, 44, 45, 46, 47]
    left_ear = eye_aspect_ratio_3d(landmarks, left_idx)
    right_ear = eye_aspect_ratio_3d(landmarks, right_idx)
    return left_ear, right_ear


if __name__ == "__main__":

    detector = LandMark()

    files = [
        f
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # large_list = []
    closed_list = []

    for fname in tqdm(files):
        img_path = os.path.join(INPUT_FOLDER, fname)

        lmk = detector.detect(img_path)
        pose = detector.pose(lmk)
        l_ear, r_ear = both_eyes_ear(lmk)

        # if abs(pose["yaw"]) > THRESHOLD_YAW or abs(pose["pitch"]) > THRESHOLD_PITCH:

        #     # print(pose)
        #     # shutil.copy(img_path, os.path.join(large_angle_folder, fname))
        #     large_list.append(fname)

        if l_ear < THRESHOLD_EAR or r_ear < THRESHOLD_EAR:
            # print(f"Left EAR: {l_ear:.3f}, Right EAR: {r_ear:.3f}")
            shutil.copy(img_path, os.path.join(closed_eyes_folder, fname))
            closed_list.append(fname)

    # Write results to text files
    # with open(LARGE_ANGLE_FILE, "w") as f:
    #     f.write(",".join(large_list))
    with open(CLOSED_EYES_FILE, "w") as f:
        f.write(",".join(closed_list))
