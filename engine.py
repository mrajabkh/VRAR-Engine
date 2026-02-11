# Tiny 3D Renderer (with outlines) augmented with perspective projection
import numpy as np


def triangle(t, v0, v1, v2, intensity):
    global coords, image, zbuffer

    # Triangle bounding box (clipped to screen)
    xmin = int(max(0,               min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1],  max(v0[0], v1[0], v2[0]) + 1))
    ymin = int(max(0,               min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0],  max(v0[1], v1[1], v2[1]) + 1))

    # All pixel positions in the bounding box (vectorized)
    P = coords[:, xmin:xmax, ymin:ymax].reshape(2, -1)

    # Barycentric coordinates for each pixel
    B = np.dot(t, np.vstack((P, np.ones((1, P.shape[1]), dtype=int))))

    # Keep pixels inside the triangle (α,β,γ >= 0)
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y = P[0, I], P[1, I]
    Z = v0[2]*B[0, I] + v1[2]*B[1, I] + v2[2]*B[2, I]

    # Z-buffer test (keep closest)
    I = np.argwhere(zbuffer[Y, X] < Z)[:, 0]
    X, Y, Z = X[I], Y[I], Z[I]
    zbuffer[Y, X] = Z
    image[Y, X] = intensity, intensity, intensity, 255

    # Outline (black) using line rasterization + depth check
    Pedge = []
    Pedge.extend(line(v0, v1))
    Pedge.extend(line(v1, v2))
    Pedge.extend(line(v2, v0))
    Pedge = np.array(Pedge).T  # shape (2, n)

    B = np.dot(t, np.vstack((Pedge, np.ones((1, Pedge.shape[1]), dtype=int))))
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y = Pedge[0, I], Pedge[1, I]
    Z = v0[2]*B[0, I] + v1[2]*B[1, I] + v2[2]*B[2, I]

    I = np.argwhere(zbuffer[Y, X] <= Z)[:, 0]
    X, Y = X[I], Y[I]
    image[Y, X] = 0, 0, 0, 255


def line(A, B):
    # Bresenham-ish integer line rasterizer; returns list[(x,y)]
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
    # Minimal OBJ loader: supports only 'v' and triangular 'f'
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
                # NOTE: expects faces like "f 1 2 3" (no slashes)
                Vi.append([int(x) for x in values[1:4]])
    return np.array(V), np.array(Vi) - 1


def lookat(eye, center, up):
    """
    Standard right-handed LookAt view matrix.
    Transforms world -> camera space.
    """
    def normalize(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    f = normalize(center - eye)         # forward
    s = normalize(np.cross(f, up))      # right
    u = np.cross(s, f)                  # corrected up

    M = np.eye(4, dtype=float)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f                       # camera looks down -Z
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] =  np.dot(f, eye)
    return M


def perspective(fov_deg, aspect, near, far):
    """
    Right-handed perspective projection matrix (OpenGL-style clip space).
    After multiplying: ndc = clip.xyz / clip.w
    """
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=float)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def viewport(x, y, w, h, d):
    # Maps NDC [-1..1] to screen coordinates
    return np.array([
        [w/2,   0,    0, x + w/2],
        [0,   h/2,    0, y + h/2],
        [0,     0,  d/2,     d/2],
        [0,     0,    0,       1]
    ], dtype=float)


if __name__ == '__main__':
    import time
    import PIL.Image

    width, height = 1200, 1200

    light  = np.array([0, 0, -1], dtype=float)  # directional light
    eye    = np.array([-1.5, 1, 2], dtype=float)
    center = np.array([0, 0, 0], dtype=float)
    up     = np.array([0, 1, 0], dtype=float)

    image = np.zeros((height, width, 4), dtype=np.uint8)
    zbuffer = -1e18 * np.ones((height, width), dtype=float)  # very far away

    # coords[0] = x grid, coords[1] = y grid
    coords = np.mgrid[0:width, 0:height].astype(int)

    V, Vi = obj_load("bunny.obj")

    # Centering and scaling the model roughly into view
    vmin, vmax = V.min(), V.max()
    V = (2 * (V - vmin) / (vmax - vmin) - 1) * 1.25
    V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
    V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) / 2

    # Matrices
    vp_mat  = viewport(32, 32, width - 64, height - 64, 1000)
    view    = lookat(eye, center, up)
    proj    = perspective(fov_deg=60, aspect=width/height, near=0.1, far=100.0)

    # Object vertices to homogeneous
    Vh = np.c_[V, np.ones(len(V))]  # (N,4)

    # World -> camera
    V_cam_h = Vh @ view.T          # (N,4)
    V_cam   = V_cam_h[:, :3]       # keep for normals/lighting if desired

    # Camera -> clip
    clip = V_cam_h @ proj.T        # (N,4)

    # Perspective divide: clip -> NDC
    ndc = clip[:, :3] / clip[:, 3:4]

    # NDC -> screen
    ndc_h = np.c_[ndc, np.ones(len(ndc))]
    Vs_h  = ndc_h @ vp_mat.T
    Vs    = Vs_h[:, :3]            # (N,3) screen x,y,z

    # Expand indexed faces (triangles)
    V_tri  = V_cam[Vi]   # (F,3,3) for normals
    Vs_tri = Vs[Vi]      # (F,3,3) for rasterization

    # Pre-compute barycentric transforms per triangle (screen space)
    T = np.transpose(Vs_tri, axes=[0, 2, 1]).copy()  # (F,3,3) but columns are vertices
    T[:, 2, :] = 1                                   # replace z row with ones for barycentrics
    T = np.linalg.inv(T)

    # Face normals (camera space) and flat intensity
    N = np.cross(V_tri[:, 2] - V_tri[:, 0], V_tri[:, 1] - V_tri[:, 0])
    N = N / np.linalg.norm(N, axis=1).reshape(len(N), 1)

    # Light dot normal -> intensity
    I = np.dot(N, light) * 255

    start = time.time()
    for i in np.argwhere(I >= 0)[:, 0]:
        vs0, vs1, vs2 = Vs_tri[i]
        triangle(T[i], vs0, vs1, vs2, I[i])
    end = time.time()

    print("Rendering time:", end - start)
    PIL.Image.fromarray(image[::-1, :, :]).save("bunny.png")
