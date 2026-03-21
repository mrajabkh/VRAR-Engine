# Tiny 3D Renderer (with outlines) augmented with perspective projection
import time
from pathlib import Path

import cv2
import numpy as np


def triangle(t, v0, v1, v2, intensity):
    global coords, image, zbuffer

    xmin = int(max(0, min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1], max(v0[0], v1[0], v2[0]) + 1))
    ymin = int(max(0, min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0], max(v0[1], v1[1], v2[1]) + 1))

    P = coords[:, xmin:xmax, ymin:ymax].reshape(2, -1)
    B = np.dot(t, np.vstack((P, np.ones((1, P.shape[1]), dtype=int))))

    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y = P[0, I], P[1, I]
    Z = v0[2] * B[0, I] + v1[2] * B[1, I] + v2[2] * B[2, I]

    # Z-buffer (smaller Z = closer)
    I = np.argwhere(zbuffer[Y, X] > Z)[:, 0]
    X, Y, Z = X[I], Y[I], Z[I]
    zbuffer[Y, X] = Z
    image[Y, X] = intensity, intensity, intensity, 255

    # Outline
    Pedge = []
    Pedge.extend(line(v0, v1))
    Pedge.extend(line(v1, v2))
    Pedge.extend(line(v2, v0))

    if len(Pedge) == 0:
        return

    Pedge = np.array(Pedge).T
    Xall = Pedge[0].astype(int)
    Yall = Pedge[1].astype(int)

    mask = (
        (Xall >= 0) & (Xall < image.shape[1]) &
        (Yall >= 0) & (Yall < image.shape[0])
    )

    if not np.any(mask):
        return

    Pedge = np.vstack((Xall[mask], Yall[mask]))

    B = np.dot(t, np.vstack((Pedge, np.ones((1, Pedge.shape[1]), dtype=int))))
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y = Pedge[0, I], Pedge[1, I]
    Z = v0[2] * B[0, I] + v1[2] * B[1, I] + v2[2] * B[2, I]

    I = np.argwhere(zbuffer[Y, X] >= Z)[:, 0]
    X, Y = X[I], Y[I]
    image[Y, X] = 0, 0, 0, 255


def line(A, B):
    (x0, y0, _), (x1, y1, _) = np.array(A).astype(int), np.array(B).astype(int)
    P = []
    steep = False

    if abs(x0 - x1) < abs(y0 - y1):
        steep, x0, y0, x1, y1 = True, y0, x0, y1, x1

    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    dx, dy = x1 - x0, y1 - y0
    y, error2, derror2 = y0, 0, abs(dy) * 2

    for x in range(x0, x1 + 1):
        P.append((y, x) if steep else (x, y))
        error2 += derror2
        if error2 > dx:
            y += 1 if y1 > y0 else -1
            error2 -= dx * 2
    return P


def obj_load(filename):
    V, Vi = [], []

    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue

            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])

            elif values[0] == 'f':
                face = [int(tok.split('/')[0]) - 1 for tok in values[1:]]
                for i in range(1, len(face) - 1):
                    Vi.append([face[0], face[i], face[i + 1]])

    return np.array(V, dtype=float), np.array(Vi, dtype=int)


def lookat(eye, center, up):
    def normalize(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.eye(4, dtype=float)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=float)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def viewport(x, y, w, h, d):
    return np.array([
        [w / 2, 0, 0, x + w / 2],
        [0, h / 2, 0, y + h / 2],
        [0, 0, d / 2, d / 2],
        [0, 0, 0, 1]
    ], dtype=float)


################################
# Problem 1 (Rendering):
# Transformation matrices: translation, rotation (X/Y/Z), scaling
################################

class Model:
    def __init__(self, V, Vi, name=""):
        self.name = name
        self.V = V
        self.Vi = Vi
        self.pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=float)   # radians: rx, ry, rz
        self.scl = np.array([1.0, 1.0, 1.0], dtype=float)

    def model_matrix(self):
        T = mat_translate(self.pos[0], self.pos[1], self.pos[2])
        Rx = mat_rot_x(self.rot[0])
        Ry = mat_rot_y(self.rot[1])
        Rz = mat_rot_z(self.rot[2])
        S = mat_scale(self.scl[0], self.scl[1], self.scl[2])
        return T @ Rz @ Ry @ Rx @ S


def render_model(model, view, proj, vp_mat, light):
    V = model.V
    Vi = model.Vi

    Vh = np.c_[V, np.ones(len(V))]
    V_world_h = Vh @ model.model_matrix().T

    V_cam_h = V_world_h @ view.T
    V_cam = V_cam_h[:, :3]

    clip = V_cam_h @ proj.T
    ndc = clip[:, :3] / clip[:, 3:4]

    ndc_h = np.c_[ndc, np.ones(len(ndc))]
    Vs = (ndc_h @ vp_mat.T)[:, :3]

    V_tri = V_cam[Vi]
    Vs_tri = Vs[Vi]

    T = np.transpose(Vs_tri, axes=[0, 2, 1]).copy()
    T[:, 2, :] = 1
    T = np.linalg.inv(T)

    N = np.cross(V_tri[:, 2] - V_tri[:, 0], V_tri[:, 1] - V_tri[:, 0])
    N = N / np.linalg.norm(N, axis=1).reshape(len(N), 1)

    I = np.dot(N, light) * 255

    for i in np.argwhere(I >= 0)[:, 0]:
        triangle(T[i], Vs_tri[i][0], Vs_tri[i][1], Vs_tri[i][2], I[i])


def prep_model(V, scale=1.25):
    vmin, vmax = V.min(), V.max()
    V = (2 * (V - vmin) / (vmax - vmin) - 1) * scale
    V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
    V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) / 2
    return V


def mat_translate(tx, ty, tz):
    M = np.eye(4, dtype=float)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def mat_scale(sx, sy, sz):
    M = np.eye(4, dtype=float)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def mat_rot_x(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=float)
    M[1, 1] = c
    M[1, 2] = -s
    M[2, 1] = s
    M[2, 2] = c
    return M


def mat_rot_y(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=float)
    M[0, 0] = c
    M[0, 2] = s
    M[2, 0] = -s
    M[2, 2] = c
    return M


def mat_rot_z(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=float)
    M[0, 0] = c
    M[0, 1] = -s
    M[1, 0] = s
    M[1, 1] = c
    return M


################################
# Problem 2 (Tracking, Handling POSE Data):
# Quaternion format: [w, x, y, z]
# rotX = roll rotY = pitch rotZ = yaw
################################

def load_imu_csv(path):
    """
    Loads the IMU CSV dataset and converts gyroscope
    readings from deg/s to rad/s.

    Returns:
        t        : (N,) time in seconds
        gyro_rad : (N,3) angular velocity in rad/s
        accel    : (N,3) acceleration in m/s^2
        mag      : (N,3) magnetometer in Gauss
    """

    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1
    )

    t = data[:, 0]
    gyro_deg = data[:, 1:4]
    accel = data[:, 4:7]
    mag = data[:, 7:10]

    gyro_rad = np.deg2rad(gyro_deg)

    return t, gyro_rad, accel, mag


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return q
    return q / n


def euler_to_quat(rotX, rotY, rotZ):
    """
    Euler angles (radians) -> quaternion [w, x, y, z]
    Convention: ZYX (rotZ Z, rotY Y, rotX X)
    """
    cr = np.cos(rotX * 0.5)
    sr = np.sin(rotX * 0.5)
    cp = np.cos(rotY * 0.5)
    sp = np.sin(rotY * 0.5)
    cy = np.cos(rotZ * 0.5)
    sy = np.sin(rotZ * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr

    return quat_normalize(np.array([w, x, y, z], dtype=float))


def quat_to_euler(q):
    """
    Quaternion [w, x, y, z] -> Euler angles (rotX, rotY, rotZ) in radians
    Convention: ZYX
    """
    q = quat_normalize(q)
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rotX = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    rotY = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rotZ = np.arctan2(siny_cosp, cosy_cosp)

    return rotX, rotY, rotZ


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a
    bw, bx, by, bz = b

    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw

    return np.array([w, x, y, z], dtype=float)


################################
# Problem 3 (Tracking, POSE Calculation):
################################

def axis_angle_to_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    axis = axis / n
    half = 0.5 * angle
    s = np.sin(half)
    return quat_normalize(np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float))


def delta_quat_from_gyro(omega_rad_s, dt):
    omega = np.asarray(omega_rad_s, dtype=float)
    omega_mag = float(np.linalg.norm(omega))
    if omega_mag < 1e-12 or dt <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    axis = omega / omega_mag
    angle = omega_mag * dt
    return axis_angle_to_quat(axis, angle)


def integrate_gyro_dead_reckoning(t, gyro_rad):
    t = np.asarray(t, dtype=float)
    gyro_rad = np.asarray(gyro_rad, dtype=float)

    N = len(t)
    Q = np.zeros((N, 4), dtype=float)
    Q[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    for k in range(N - 1):
        dt = float(t[k + 1] - t[k])
        dq = delta_quat_from_gyro(gyro_rad[k], dt)
        Q[k + 1] = quat_normalize(quat_multiply(Q[k], dq))

    return Q


def quat_rotate_vector(q, v):
    q = quat_normalize(q)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    qr = quat_multiply(quat_multiply(q, vq), quat_conjugate(q))
    return qr[1:]


def integrate_gyro_accel_complementary(t, gyro_rad, accel, alpha, up_global, accel_sign):
    t = np.asarray(t, dtype=float)
    gyro_rad = np.asarray(gyro_rad, dtype=float)
    accel = np.asarray(accel, dtype=float)

    up_global = np.asarray(up_global, dtype=float)
    up_global = up_global / np.linalg.norm(up_global)

    N = len(t)
    Q = np.zeros((N, 4), dtype=float)
    Q[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    for k in range(N - 1):
        dt = float(t[k + 1] - t[k])

        dq = delta_quat_from_gyro(gyro_rad[k], dt)
        q_gyro = quat_normalize(quat_multiply(Q[k], dq))

        a_body = accel[k+1] #k+1 or k??
        a_norm = np.linalg.norm(a_body)

        if a_norm < 1e-12:
            Q[k + 1] = q_gyro
            continue

        a_body = a_body / a_norm

        a_global = accel_sign * quat_rotate_vector(q_gyro, a_body)
        a_global_norm = np.linalg.norm(a_global)

        if a_global_norm < 1e-12:
            Q[k + 1] = q_gyro
            continue

        a_global = a_global / a_global_norm

        tilt_axis = np.cross(a_global, up_global)
        axis_norm = np.linalg.norm(tilt_axis)

        dot_val = np.clip(np.dot(a_global, up_global), -1.0, 1.0)
        phi = np.arccos(dot_val)

        if axis_norm < 1e-12 or phi < 1e-12:
            Q[k + 1] = q_gyro
            continue

        tilt_axis = tilt_axis / axis_norm

        correction_angle = (1.0 - alpha) * phi

        q_corr = axis_angle_to_quat(tilt_axis, correction_angle)
        Q[k + 1] = quat_normalize(quat_multiply(q_corr, q_gyro))

    return Q

################################
# Problem 4 (Advanced Tracking, Mitigating Yaw Drift):
################################


################################
# MAIN
################################

if __name__ == "__main__":

    width, height = 1200, 1200

    light = np.array([0, 0, -1], dtype=float)
    up = np.array([0, 1, 0], dtype=float)

    image = np.zeros((height, width, 4), dtype=np.uint8)
    zbuffer = 1e18 * np.ones((height, width), dtype=float)
    coords = np.mgrid[0:width, 0:height].astype(int)

    here = Path(__file__).resolve().parent
    V_bunny, Vi_bunny = obj_load(str(here / "bunny.obj"))
    V_floor, Vi_floor = obj_load(str(here / "floor.obj"))

    V_bunny = prep_model(V_bunny, 1.0)
    V_floor = prep_model(V_floor, 1.0)

    bunny = Model(V_bunny, Vi_bunny, "bunny")
    floor = Model(V_floor, Vi_floor, "floor")

    floor.scl[:] = [1.5, 0.10, 1.5]
    floor.pos[:] = [0.0, -1.2, 0.0]

    bunny.scl[:] = [0.5, 0.5, 0.5]
    bunny.pos[:] = [0.0, -0.2, 0.0]

    floor_top_y = (V_floor[:, 1] * floor.scl[1] + floor.pos[1]).max()
    bunny_bottom_y = (V_bunny[:, 1] * bunny.scl[1] + bunny.pos[1]).min()
    bunny.pos[1] += (floor_top_y - bunny_bottom_y) + 0.02

    vp_mat = viewport(32, 32, width - 64, height - 64, 1000)
    proj = perspective(60, width / height, 0.1, 100.0)

    center = bunny.pos.copy()

    bunny_size = np.max(V_bunny, axis=0) - np.min(V_bunny, axis=0)
    bunny_r = float(np.linalg.norm(bunny_size * bunny.scl))

    rotZ = 1.0
    rotY = 0.55
    radius = max(1.5, bunny_r * 2.0)

    step_rotZ = 0.12
    step_rotY = 0.08
    step_zoom = 0.35

    cv2.namedWindow("Framebuffer", cv2.WINDOW_NORMAL)

    # IMU data + fused orientation sequence
    t, gyro, accel, mag = load_imu_csv(str(here / "IMUData.csv"))

    Q_gyro = integrate_gyro_dead_reckoning(t, gyro)

    Q_fused = integrate_gyro_accel_complementary(
        t,
        gyro,
        accel,
        alpha=0.98,
        up_global=np.array([0.0, 0.0, 1.0]),
        accel_sign=1.0
    )

    # Re-reference the motion so the first frame is the visual starting pose
    # q_start_inv = quat_conjugate(Q_fused[0])
    # Q_fused = np.array([quat_normalize(quat_multiply(q_start_inv, q)) for q in Q_fused])
    # ####

    # Timestamp-based playback
    playback_start = time.time()
    imu_t0 = float(t[0])
    imu_duration = float(t[-1] - t[0])

    # TESTING

    # print("Number of samples:", len(t))
    # print("Gyro shape:", gyro.shape)
    # print("Accel shape:", accel.shape)
    # print("Mag shape:", mag.shape)

    # print("\n================ GYRO DEAD RECKONING CHECK ================\n")
    # print("Q shape:", Q_gyro.shape)
    # print("First quaternion:", Q_gyro[0])
    # print("Last quaternion:", Q_gyro[-1])
    # q_norms = np.linalg.norm(Q_gyro, axis=1)
    # print("\nQuaternion norm min:", float(q_norms.min()))
    # print("Quaternion norm max:", float(q_norms.max()))
    # print("\nAny NaNs in Q:", np.isnan(Q_gyro).any())
    # print("Any Infs in Q:", np.isinf(Q_gyro).any())
    # print("\nFirst 5 Euler angles (deg):")
    # for i in range(5):
    #     r, p, y = quat_to_euler(Q_gyro[i])
    #     print(i, np.rad2deg([r, p, y]))
    # r, p, y = quat_to_euler(Q_gyro[-1])
    # print("\nFinal Euler angles (deg):", np.rad2deg([r, p, y]))
    # omega_mag = np.linalg.norm(gyro, axis=1)
    # print("\nGyro magnitude min/max (rad/s):", float(omega_mag.min()), float(omega_mag.max()))
    # print("\n============================================================\n")

    # print("\n================ COMPLEMENTARY FILTER CHECK ================\n")
    # print("Q_fused shape:", Q_fused.shape)
    # print("First fused quaternion:", Q_fused[0])
    # print("Last fused quaternion:", Q_fused[-1])
    # qf_norms = np.linalg.norm(Q_fused, axis=1)
    # print("Fused norm min:", float(qf_norms.min()))
    # print("Fused norm max:", float(qf_norms.max()))
    # rf, pf, yf = quat_to_euler(Q_fused[-1])
    # print("Final fused Euler angles (deg):", np.rad2deg([rf, pf, yf]))
    # print("\n===========================================================\n")

    while True:
        start = time.time()

        image[:] = 0
        zbuffer[:] = 1e18

        eye = center + np.array([
            radius * np.cos(rotY) * np.cos(rotZ),
            radius * np.sin(rotY),
            radius * np.cos(rotY) * np.sin(rotZ),
        ], dtype=float)

        view = lookat(eye, center, up)

        # Use IMU timestamps for playback timing
        elapsed = (time.time() - playback_start) % imu_duration
        playback_t = imu_t0 + elapsed
        frame_idx = np.searchsorted(t, playback_t, side="left")
        frame_idx = min(frame_idx, len(Q_fused) - 1)

        # Drive bunny from fused gyro + accelerometer orientation
        roll_bunny, pitch_bunny, yaw_bunny = quat_to_euler(Q_fused[frame_idx])
        # roll_bunny, pitch_bunny, yaw_bunny = quat_to_euler(Q_gyro[frame_idx])
        bunny.rot[:] = [roll_bunny, pitch_bunny, yaw_bunny]

        # render_model(floor, view, proj, vp_mat, light)
        render_model(bunny, view, proj, vp_mat, light)

        frame = image[::-1, :, :][:, :, [2, 1, 0, 3]]
        cv2.imshow("Framebuffer", frame)

        end = time.time()
        cv2.setWindowTitle("Framebuffer", "Framebuffer | Frame time: %.3fs" % (end - start))

        key = cv2.waitKey(30) & 0xFF

        if key == ord("a"):
            rotZ -= step_rotZ
        elif key == ord("d"):
            rotZ += step_rotZ
        elif key == ord("w"):
            rotY += step_rotY
        elif key == ord("s"):
            rotY -= step_rotY
        elif key == ord("q"):
            radius = max(1.0, radius - step_zoom)
        elif key == ord("e"):
            radius += step_zoom
        elif key == 27:
            break

        rotY = max(0.10, min(1.20, rotY))

        if cv2.getWindowProperty("Framebuffer", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()