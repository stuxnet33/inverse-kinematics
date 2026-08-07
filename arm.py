import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from fabrik import fabrik, positions_to_angles

# ── Arm configuration ─────────────────────────────────────────────────────────
LINK_LENGTHS = [2.0, 1.5, 1.0]

JOINT_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]

# ── Rotation matrices ─────────────────────────────────────────────────────────
def Rz(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1]
    ])

def Ry(a):
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [ 0,         1, 0        ],
        [-np.sin(a), 0, np.cos(a)]
    ])

# ── Forward kinematics ────────────────────────────────────────────────────────
def forward_kinematics(angles):
    """
    Computes joint positions using rotation matrices.
    Joint 0: base rotates around Z (yaw)
    Joint 1: shoulder rotates around Y (pitch)
    Joint 2: elbow rotates around Y (pitch)
    Returns list of np.array [x,y,z] for each joint + end effector.
    """
    positions = [np.zeros(3)]
    R = np.eye(3)
    p = np.zeros(3)

    R = R @ Rz(angles[0])
    p = p + R @ np.array([LINK_LENGTHS[0], 0.0, 0.0])
    positions.append(p.copy())

    R = R @ Ry(angles[1])
    p = p + R @ np.array([LINK_LENGTHS[1], 0.0, 0.0])
    positions.append(p.copy())

    R = R @ Ry(angles[2])
    p = p + R @ np.array([LINK_LENGTHS[2], 0.0, 0.0])
    positions.append(p.copy())

    return positions


def end_effector(angles):
    return forward_kinematics(angles)[-1].copy()


# ── Numerical Jacobian ────────────────────────────────────────────────────────
def jacobian(angles, delta=1e-5):
    """Central differences Jacobian — more accurate than forward differences."""
    n = len(angles)
    J = np.zeros((3, n))
    for i in range(n):
        ap = angles.copy(); ap[i] += delta
        am = angles.copy(); am[i] -= delta
        J[:, i] = (end_effector(ap) - end_effector(am)) / (2 * delta)
    return J


# ── DLS Inverse kinematics ────────────────────────────────────────────────────
def inverse_kinematics(target, initial_angles=None,
                        max_iterations=200, tolerance=1e-3):
    """
    Damped Least Squares (DLS) IK solver.
    Update rule: delta_q = J^T (J J^T + lambda^2 I)^-1 * error
    Adaptive lambda — large when far from target, small when close.
    """
    angles = np.zeros(len(LINK_LENGTHS)) if initial_angles is None \
             else np.array(initial_angles, dtype=float)

    target = np.array(target, dtype=float)
    history = [forward_kinematics(angles)]

    for i in range(max_iterations):
        pos  = end_effector(angles)
        error = target - pos
        dist  = np.linalg.norm(error)

        if dist < tolerance:
            print(f"Converged in {i} iterations  |  error: {dist:.6f}")
            return angles, history, True

        J   = jacobian(angles)
        lam = 0.5 * dist + 0.001
        A   = J @ J.T + lam**2 * np.eye(3)
        delta = J.T @ np.linalg.solve(A, error)

        step  = min(dist, 0.3)
        delta = delta * step / (np.linalg.norm(delta) + 1e-8)
        angles = angles + delta

        for j, (lo, hi) in enumerate(JOINT_LIMITS):
            angles[j] = np.clip(angles[j], lo, hi)

        if i % 15 == 0:
            history.append(forward_kinematics(angles))

    print(f"Did not converge after {max_iterations} iterations")
    return angles, history, False


# ── DLS with random restarts ──────────────────────────────────────────────────
def dls_with_restarts(target, current_angles, n_restarts=2):
    """
    Runs DLS with multiple random starting configurations.
    Returns the best result found.
    """
    # Check reachability first — skip DLS entirely if unreachable
    dist = np.linalg.norm(np.array(target))
    if dist > sum(LINK_LENGTHS) * 0.99:
        angles = np.zeros(3)
        positions = forward_kinematics(angles)
        return angles, [], False, 0, dist

    best_angles    = current_angles.copy()
    best_error     = float('inf')
    best_history   = []
    best_iters     = 0
    best_converged = False

    starts = [current_angles.copy()] + [
        np.random.uniform(-np.pi, np.pi, 3) for _ in range(n_restarts)
    ]

    for start in starts:
        angles, history, converged = inverse_kinematics(target, initial_angles=start)
        err = np.linalg.norm(end_effector(angles) - target)
        if err < best_error:
            best_error     = err
            best_angles    = angles
            best_history   = history
            best_iters     = len(history)
            best_converged = converged
        if converged:
            break

    return best_angles, best_history, best_converged, best_iters, best_error


# ── Visualisation helpers ─────────────────────────────────────────────────────
def setup_axes(ax, title):
    reach = sum(LINK_LENGTHS)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])


def draw_arm_3d(ax, positions, color='royalblue', alpha=1.0, lw=3):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    ax.plot(xs, ys, zs, 'o-', color=color,
            linewidth=lw, markersize=8, alpha=alpha)
    ax.scatter(*positions[-1], color='orange', s=120, zorder=5)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=== Robot Arm IK — FABRIK vs DLS ===")
    print("Drag sliders to move target")
    print("Press 1-5 for presets  |  R for home")
    print("Both solvers run simultaneously for live comparison\n")

    # Home position — arm stretched along X
    initial_positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([LINK_LENGTHS[0], 0.0, 0.0]),
        np.array([LINK_LENGTHS[0] + LINK_LENGTHS[1], 0.0, 0.0]),
        np.array([sum(LINK_LENGTHS), 0.0, 0.0]),
    ]

    current_angles    = np.zeros(3)
    current_positions = [p.copy() for p in initial_positions]

    # ── Figure layout ──
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Robot Arm Inverse Kinematics — FABRIK vs DLS", fontsize=13)

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    reach = sum(LINK_LENGTHS)
    ax_x = fig.add_axes([0.1,  0.10, 0.35, 0.025])
    ax_y = fig.add_axes([0.1,  0.06, 0.35, 0.025])
    ax_z = fig.add_axes([0.1,  0.02, 0.35, 0.025])

    slider_x = Slider(ax_x, 'X', -reach, reach, valinit=reach, color='royalblue')
    slider_y = Slider(ax_y, 'Y', -reach, reach, valinit=0.0,   color='royalblue')
    slider_z = Slider(ax_z, 'Z', -reach, reach, valinit=0.0,   color='royalblue')

    status_text = fig.text(
        0.55, 0.06,
        "Drag sliders to move target",
        ha='left', fontsize=8, color='navy',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    # ── Render function ──
    def render_comparison(target):
        global current_angles, current_positions

        target = np.array(target, dtype=float)

        # FABRIK
        fab_pos, _, fab_conv, fab_iters = fabrik(
            [p.copy() for p in initial_positions],
            LINK_LENGTHS, target
        )
        fab_err = np.linalg.norm(fab_pos[-1] - target)

        # DLS with restarts
        dls_ang, _, dls_conv, dls_iters, dls_err = dls_with_restarts(
            target, current_angles
        )
        dls_pos = forward_kinematics(dls_ang)
        current_angles = dls_ang
        current_positions = dls_pos

        # Draw
        ax1.clear()
        ax2.clear()
        setup_axes(ax1, f"FABRIK  ({fab_iters} iters)")
        setup_axes(ax2, f"DLS  ({'✓' if dls_conv else '✗'})")

        draw_arm_3d(ax1, initial_positions, color='lightblue', alpha=0.25)
        draw_arm_3d(ax2, initial_positions, color='lightblue', alpha=0.25)
        draw_arm_3d(ax1, fab_pos, color='royalblue')
        draw_arm_3d(ax2, dls_pos, color='seagreen')

        ax1.scatter(*target, color='red', s=200, marker='*', zorder=5)
        ax2.scatter(*target, color='red', s=200, marker='*', zorder=5)

        status_text.set_text(
            f"Target: {np.round(target, 2)}\n"
            f"FABRIK: {fab_iters} iters  |  err={fab_err:.4f}  |  "
            f"{'converged' if fab_conv else 'unreachable'}\n"
            f"DLS:    {dls_iters} iters  |  err={dls_err:.4f}  |  "
            f"{'converged' if dls_conv else 'did not converge'}"
        )

        fig.canvas.draw_idle()

    # ── Slider callback ──
    def on_slider(val):
        target = np.array([slider_x.val, slider_y.val, slider_z.val])
        render_comparison(target)
        fig.canvas.get_tk_widget().focus_set() 

    slider_x.on_changed(on_slider)
    slider_y.on_changed(on_slider)
    slider_z.on_changed(on_slider)

    # ── Keyboard callback ──
    presets = {
        '1': [3.0,  0.0,  0.5],
        '2': [2.0,  2.0,  1.0],
        '3': [1.0, -2.0,  1.5],
        '4': [-2.0, 1.0,  0.5],
        '5': [2.5,  1.5, -0.5],
        'r': [reach, 0.0,  0.0],
    }

    def on_key(event):
        if event.key in presets:
            t = presets[event.key]
            slider_x.set_val(t[0])
            slider_y.set_val(t[1])
            slider_z.set_val(t[2])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── Initial render at home ──
    render_comparison([reach, 0.0, 0.0])

    plt.subplots_adjust(bottom=0.18)
    plt.show()