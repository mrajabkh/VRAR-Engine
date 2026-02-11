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
    Z = v0[2]*B[0, I] + v1[2]*B[1, I] + v2[2]*B[2, I]

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
    Z = v0[2]*B[0, I] + v1[2]*B[1, I] + v2[2]*B[2, I]

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
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                Vi.append([int(x) for x in values[1:4]])
    return np.array(V), np.array(Vi) - 1


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
    M[2, 3] =  np.dot(f, eye)
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
        [w/2,   0,    0, x + w/2],
        [0,   h/2,    0, y + h/2],
        [0,     0,  d/2,     d/2],
        [0,     0,    0,       1]
    ], dtype=float)


def prep_model(V, scale=1.25):
    vmin, vmax = V.min(), V.max()
    V = (2 * (V - vmin) / (vmax - vmin) - 1) * scale
    V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
    V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) / 2
    return V


def render_model(V, Vi, view, proj, vp_mat, light):
    Vh = np.c_[V, np.ones(len(V))]
    V_cam_h = Vh @ view.T
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

    # Your model prep (keep it simple)
    V_bunny = prep_model(V_bunny, 1.0) / 2.0
    V_floor = prep_model(V_floor, 1.25)
    V_floor *= 1.5

    # Put floor lower first (so it is clearly "below")
    V_floor[:, 1] -= 1.2

    # Put bunny on top of floor
    gap = 0.02
    V_bunny[:, 1] += (V_floor[:, 1].max() - V_bunny[:, 1].min()) + gap

    vp_mat = viewport(32, 32, width - 64, height - 64, 1000)
    proj = perspective(60, width / height, 0.1, 100.0)

    # Orbit around bunny
    center = V_bunny.mean(axis=0).copy()

    # Choose a stable radius based on bunny size only
    bunny_size = np.max(V_bunny, axis=0) - np.min(V_bunny, axis=0)
    bunny_r = float(np.linalg.norm(bunny_size))

    # Good default view: above the floor, looking down at bunny
    yaw = 2.2
    pitch = 0.55
    radius = max(2.2, bunny_r * 3.1)

    step_yaw = 0.12
    step_pitch = 0.08
    step_zoom = 0.35

    cv2.namedWindow("Framebuffer", cv2.WINDOW_NORMAL)

    while True:
        start = time.time()

        image[:] = 0
        zbuffer[:] = 1e18

        # Camera position
        eye = center + np.array([
            radius * np.cos(pitch) * np.cos(yaw),
            radius * np.sin(pitch),
            radius * np.cos(pitch) * np.sin(yaw),
        ], dtype=float)

        view = lookat(eye, center, up)

        render_model(V_floor, Vi_floor, view, proj, vp_mat, light)
        render_model(V_bunny, Vi_bunny, view, proj, vp_mat, light)

        frame = image[::-1, :, :][:, :, [2, 1, 0, 3]]
        cv2.imshow("Framebuffer", frame)

        end = time.time()
        cv2.setWindowTitle("Framebuffer", "Framebuffer | Frame time: %.3fs" % (end - start))

        key = cv2.waitKey(30) & 0xFF

        if key == ord("a"):
            yaw -= step_yaw
        elif key == ord("d"):
            yaw += step_yaw
        elif key == ord("w"):
            pitch += step_pitch
        elif key == ord("s"):
            pitch -= step_pitch
        elif key == ord("q"):
            radius = max(1.0, radius - step_zoom)
        elif key == ord("e"):
            radius += step_zoom
        elif key == 27:
            break

        # Keep camera above the floor-ish (prevents going underneath)
        pitch = max(0.10, min(1.20, pitch))

        if cv2.getWindowProperty("Framebuffer", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
