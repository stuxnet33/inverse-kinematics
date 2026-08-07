import numpy as np


def normalize(v):
    """
    Returns the unit vector of v.
    If v is zero length, returns v unchanged to avoid division by zero.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def fabrik(positions, link_lengths, target, tolerance=1e-3, max_iterations=100):
    """
    FABRIK — Forward And Backward Reaching Inverse Kinematics.

    Parameters:
        positions: list of np.array [x,y,z] — current joint positions
                   positions[0] is the base (fixed), positions[-1] is end effector
        link_lengths: list of floats — length of each link between joints
        target: np.array [x,y,z] — desired end effector position
        tolerance: stop when end effector is within this distance of target
        max_iterations: maximum number of forward/backward passes

    Returns:
        positions: updated joint positions
        history: list of joint positions at each iteration for animation
        converged: bool
        iterations: how many iterations it took

    The algorithm:
    Each iteration = one forward pass + one backward pass.
    Forward pass pulls joints toward target from end to base.
    Backward pass anchors base and pushes joints toward target from base to end.
    Together they converge the end effector onto the target while keeping
    all link lengths exactly correct.
    """
    n = len(positions)
    positions = [p.copy() for p in positions]
    base = positions[0].copy()   # base is always fixed
    target = np.array(target, dtype=float)
    history = [[p.copy() for p in positions]]

    # Check if target is reachable
    total_reach = sum(link_lengths)
    dist_to_target = np.linalg.norm(target - base)

    if dist_to_target > total_reach:
        # Target unreachable — stretch arm toward target as far as possible
        print(f"Target unreachable (distance {dist_to_target:.2f} > reach {total_reach:.2f})")
        direction = normalize(target - base)
        for i in range(1, n):
            positions[i] = positions[i-1] + direction * link_lengths[i-1]
        return positions, history, False, 0

    for iteration in range(max_iterations):

        # ── FORWARD PASS ─────────────────────────────────────────────────
        # Pull end effector to target, drag each joint along behind it

        positions[-1] = target.copy()

        for i in range(n - 2, -1, -1):
            # Direction from current joint i toward the joint ahead (i+1)
            direction = normalize(positions[i] - positions[i + 1])
            # Place joint i at the correct link length behind joint i+1
            positions[i] = positions[i + 1] + direction * link_lengths[i]

        # ── BACKWARD PASS ────────────────────────────────────────────────
        # Anchor base, push joints forward toward target

        positions[0] = base.copy()

        for i in range(1, n):
            # Direction from previous joint toward current joint
            direction = normalize(positions[i] - positions[i - 1])
            # Place current joint at correct link length ahead of previous
            positions[i] = positions[i - 1] + direction * link_lengths[i - 1]

        # Store snapshot for animation
        history.append([p.copy() for p in positions])

        # Check convergence
        dist = np.linalg.norm(positions[-1] - target)
        if dist < tolerance:
            print(f"FABRIK converged in {iteration + 1} iterations  |  error: {dist:.6f}")
            return positions, history, True, iteration + 1

    dist = np.linalg.norm(positions[-1] - target)
    print(f"FABRIK did not converge after {max_iterations} iterations  |  error: {dist:.6f}")
    return positions, history, False, max_iterations


def positions_to_angles(positions, link_lengths):
    """
    Converts FABRIK joint positions back to joint angles.
    This is needed because FABRIK works in Cartesian space (positions)
    while the physical arm needs joint angles (degrees for servos).

    For each link, we compute the direction vector and extract
    the rotation angles from it.

    Returns angles in radians.
    """
    angles = []

    # Base angle — rotation around Z (yaw)
    # Direction of first link in XY plane
    link0 = positions[1] - positions[0]
    base_angle = np.arctan2(link0[1], link0[0])
    angles.append(base_angle)

    # Shoulder angle — rotation around Y (pitch of first link)
    # How much does the first link point up or down?
    xy_dist = np.sqrt(link0[0]**2 + link0[1]**2)
    shoulder_angle = np.arctan2(link0[2], xy_dist)
    angles.append(shoulder_angle)

    # Elbow angle — relative angle between first and second link
    link1 = positions[2] - positions[1]
    link2 = positions[3] - positions[2]
    # Dot product gives angle between links
    cos_angle = np.clip(np.dot(normalize(link1), normalize(link2)), -1, 1)
    elbow_angle = np.arccos(cos_angle)
    # Determine sign — is the elbow bending up or down?
    cross = np.cross(link1, link2)
    if cross[1] < 0:
        elbow_angle = -elbow_angle
    angles.append(elbow_angle)

    return np.array(angles)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # Test FABRIK on the same targets as the DLS solver
    LINK_LENGTHS = [2.0, 1.5, 1.0]

    # Initial positions — arm stretched along X axis
    initial_positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.5, 0.0, 0.0]),
        np.array([4.5, 0.0, 0.0]),
    ]

    targets = [
        np.array([3.0,  0.0,  0.5]),
        np.array([2.0,  2.0,  1.0]),
        np.array([1.0, -2.0,  1.5]),
        np.array([-2.0, 1.0,  0.5]),
        np.array([2.5,  1.5, -0.5]),
    ]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("FABRIK — Forward And Backward Reaching IK", fontsize=13)

    for idx, target in enumerate(targets):
        positions, history, converged, iters = fabrik(
            initial_positions, LINK_LENGTHS, target
        )

        angles = positions_to_angles(positions, LINK_LENGTHS)
        print(f"Target {idx+1}: {target} → "
              f"EE: {np.round(positions[-1], 3)} | "
              f"Angles: {np.round(np.degrees(angles), 1)}°")

        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        ax.set_title(f"T{idx+1} — {iters} iters")

        reach = sum(LINK_LENGTHS)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(-reach, reach)
        ax.set_box_aspect([1,1,1])

        # Draw convergence history
        for k, h in enumerate(history):
            alpha = (k + 1) / len(history)
            xs = [p[0] for p in h]
            ys = [p[1] for p in h]
            zs = [p[2] for p in h]
            ax.plot(xs, ys, zs, 'o-',
                    color=plt.cm.cool(alpha),
                    linewidth=2, markersize=4, alpha=alpha)

        # Final arm
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        ax.plot(xs, ys, zs, 'o-', color='royalblue',
                linewidth=3, markersize=8)
        ax.scatter(*target, color='red', s=150, marker='*', zorder=5)
        ax.scatter(*positions[-1], color='orange', s=100, zorder=5)

    plt.tight_layout()
    plt.show()