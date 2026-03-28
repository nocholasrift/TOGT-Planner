import numpy as np


def figure8_position(theta, a=10.0, b=5.0, z_min=1.0, z_max=8.0):
    z_mid = (z_min + z_max) / 2.0
    z_amp = (z_max - z_min) / 2.0
    x = a * np.sin(theta)
    y = b * np.sin(2.0 * theta)
    z = z_mid + z_amp * np.sin(theta)
    return np.array([x, y, z])


def figure8_kinematics(theta, a=10.0, b=5.0, z_min=1.0, z_max=8.0, v_ref=2.0):
    eps = 1e-5

    def pos(t):
        return figure8_position(t, a, b, z_min, z_max)

    p0 = pos(theta)
    p1 = pos(theta + eps)
    pm1 = pos(theta - eps)
    p2 = pos(theta + 2 * eps)

    tangent = (p1 - pm1) / (2 * eps)
    speed = np.linalg.norm(tangent)
    unit_t = tangent / speed
    vel = unit_t * v_ref

    tang_next = (p2 - p0) / (2 * eps)
    vel_next = (tang_next / np.linalg.norm(tang_next)) * v_ref
    tang_prev = (p0 - pos(theta - 2 * eps)) / (2 * eps)
    vel_prev = (tang_prev / np.linalg.norm(tang_prev)) * v_ref

    ds = np.linalg.norm(p1 - p0)
    dt = ds / v_ref
    acc = (vel_next - vel_prev) / (2 * dt) if dt > 1e-10 else np.zeros(3)
    jerk = np.zeros(3)

    yaw_rad = np.arctan2(vel[1], vel[0])
    yaw_deg = np.degrees(yaw_rad)
    quat = [np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]

    return p0, vel, acc, jerk, yaw_deg, quat


def generate_figure8_env(
    a=10.0, b=5.0, z_min=1.0, z_max=8.0,
    num_gates=8, v_ref=2.0,
):
    z_mid = (z_min + z_max) / 2.0

    p0, v0, a0, j0, yaw0, q0 = figure8_kinematics(0, a, b, z_min, z_max, v_ref)

    # ── initState ──
    print("initState:")
    print(f"  pos: [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}]")
    print(f"  vel: [0.0, 0.0, 0.0]")
    print(f"  acc: [0.0, 0.0, 0.0]")
    print(f"  jer: [0.0, 0.0, 0.0]")
    print(f"  rot: [1.0, 0.0, 0.0, 0.0]")
    print(f"  cthrustmass: 9.8066")
    print(f"  euler: [0.0, 0.0, 0.0]")
    print()

    # ── endState ──
    print("endState:")
    print(f"  pos: [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}]")
    print(f"  vel: [{v0[0]:.2f}, {v0[1]:.2f}, {v0[2]:.2f}]")
    print(f"  acc: [0.0, 0.0, 0.0]")
    print(f"  jer: [0.0, 0.0, 0.0]")
    print(f"  cthrustmass: 9.8066")
    print(f"  euler: [0.0, 0.0, {yaw0:.2f}]")
    print(f"  rot: [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]")
    print()

    # ── Gates ──
    gate_thetas = np.linspace(0, 2 * np.pi, num_gates + 1)[1:-1]
    gate_names = [f"Gate{i+1}" for i in range(len(gate_thetas))]
    print(f"orders: {gate_names}")
    print()

    gate_types = [
        "RectanglePrisma", "PentagonPrisma", "TrianglePrisma",
        "RectanglePrisma", "HexagonPrisma", "RectanglePrisma",
        "PentagonPrisma", "TrianglePrisma",
    ]
    gate_names_fancy = [
        "climb_gate", "upper_loop", "apex_gate", "descent_gate",
        "lower_cross", "dive_gate", "lower_loop", "return_gate",
    ]

    for i, theta in enumerate(gate_thetas):
        pos, vel, _, _, yaw, _ = figure8_kinematics(theta, a, b, z_min, z_max, v_ref)
        yaw_w = (yaw + 180) % 360 - 180
        gtype = gate_types[i % len(gate_types)]
        gname = gate_names_fancy[i % len(gate_names_fancy)]

        print(f"# GATE {i+1}: {gname}")
        print(f"Gate{i+1}:")
        print(f"  type: '{gtype}'")
        print(f"  name: '{gname}'")
        print(f"  position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(f"  rpy: [0.0, -90.0, {yaw_w:.2f}]")

        if gtype in ("RectanglePrisma", "TrianglePrisma"):
            print(f"  width: 2.4")
            print(f"  height: 2.4")
            print(f"  marginW: 0.0")
            print(f"  marginH: 0.0")
        elif gtype == "PentagonPrisma":
            print(f"  radius: 2.2")
            print(f"  margin: 0.0")
        elif gtype == "HexagonPrisma":
            print(f"  side: 1.2")
            print(f"  margin: 0.0")

        print(f"  length: 0.0")
        print(f"  midpoints: 0")
        print(f"  stationary: true")
        print()

    # ══════════════════════════════════════════════════════════════════
    #  OBSTACLES
    # ══════════════════════════════════════════════════════════════════

    # Figure-8 key positions:
    #   theta = pi/2  → max x = +a, y ≈ 0, z = z_max  (right tip)
    #   theta = 3pi/2 → min x = -a, y ≈ 0, z = z_min  (left tip)
    #
    # Horizontal squeeze plates at the two x-extremes force the drone
    # through a narrow z-band at the tips of the figure-8.
    #
    # Vertical corridor obstacles are evenly spaced along the rest.

    pos_right = figure8_position(np.pi / 2, a, b, z_min, z_max)   # (+a, ~0, z_max)
    pos_left = figure8_position(3 * np.pi / 2, a, b, z_min, z_max)  # (-a, ~0, z_min)

    # How much gap to leave for the drone (squeeze gap in z)
    squeeze_gap = 3.0  # meters of clearance
    plate_thickness = 0.4
    plate_width = 6.0  # wide enough to block bypass in y

    # Vertical corridor parameters
    corridor_gap = 3.0   # gap between pillars
    pillar_height = z_max + 2.0  # tall enough to block

    print("# ══════════════════════════════════════════════════════════════")
    print("# OBSTACLE DEFINITIONS")
    print("# ══════════════════════════════════════════════════════════════")
    print()
    print("templates:")
    print("  - &squeeze_plate")
    print("    type: \"box\"")
    print(f"    size: [{plate_width:.1f}, {plate_width:.1f}, {plate_thickness:.1f}]")
    print()
    print("  - &corridor_pillar")
    print("    type: \"box\"")
    print(f"    size: [0.5, 0.5, {pillar_height:.1f}]")
    print()
    print("  - &crossbar")
    print("    type: \"box\"")
    print("    size: [4.0, 0.4, 0.4]")
    print()
    print("obstacles:")

    # ── HORIZONTAL SQUEEZE 1: Right tip (max x, high z) ──
    # Ceiling plate just above the trajectory
    ceil_z_right = pos_right[2] + squeeze_gap / 2.0
    floor_z_right = pos_right[2] - squeeze_gap / 2.0

    print()
    print(f"  # ── HORIZONTAL SQUEEZE 1: Right tip (x ≈ +{a:.0f}) ──")
    print(f"  # Trajectory passes at z ≈ {pos_right[2]:.1f}, squeeze to {squeeze_gap:.1f}m gap")
    print()
    print(f"  # Ceiling plate")
    print(f"  - <<: *squeeze_plate")
    print(f"    position: [{pos_right[0]:.2f}, {pos_right[1]:.2f}, {ceil_z_right:.2f}]")
    print(f"    rotation: [0, 0, 0]")
    print()
    print(f"  # Floor plate")
    print(f"  - <<: *squeeze_plate")
    print(f"    position: [{pos_right[0]:.2f}, {pos_right[1]:.2f}, {floor_z_right:.2f}]")
    print(f"    rotation: [0, 0, 0]")

    # ── HORIZONTAL SQUEEZE 2: Left tip (min x, low z) ──
    ceil_z_left = pos_left[2] + squeeze_gap / 2.0
    floor_z_left = pos_left[2] - squeeze_gap / 2.0

    print()
    print(f"  # ── HORIZONTAL SQUEEZE 2: Left tip (x ≈ {-a:.0f}) ──")
    print(f"  # Trajectory passes at z ≈ {pos_left[2]:.1f}, squeeze to {squeeze_gap:.1f}m gap")
    print()
    print(f"  # Ceiling plate")
    print(f"  - <<: *squeeze_plate")
    print(f"    position: [{pos_left[0]:.2f}, {pos_left[1]:.2f}, {ceil_z_left:.2f}]")
    print(f"    rotation: [0, 0, 0]")
    print()
    print(f"  # Floor plate")
    print(f"  - <<: *squeeze_plate")
    print(f"    position: [{pos_left[0]:.2f}, {pos_left[1]:.2f}, {floor_z_left:.2f}]")
    print(f"    rotation: [0, 0, 0]")

    # ── 4 VERTICAL CORRIDOR OBSTACLES ──
    # Evenly spaced at theta = pi/4, 3pi/4, 5pi/4, 7pi/4
    # (the "diagonal" portions of the figure-8, between tips and crossing)
    corridor_thetas = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    corridor_labels = [
        "Upper-right corridor (climbing)",
        "Upper-left corridor (crossing high)",
        "Lower-left corridor (descending)",
        "Lower-right corridor (crossing low)",
    ]

    for ci, (ct, label) in enumerate(zip(corridor_thetas, corridor_labels)):
        cpos = figure8_position(ct, a, b, z_min, z_max)

        # Tangent direction to orient the corridor perpendicular to flight
        eps = 1e-5
        tang = figure8_position(ct + eps, a, b, z_min, z_max) - figure8_position(ct - eps, a, b, z_min, z_max)
        tang_xy = tang[:2]
        tang_xy /= np.linalg.norm(tang_xy)

        # Normal in XY (perpendicular to flight direction)
        normal_xy = np.array([-tang_xy[1], tang_xy[0]])

        # Place two pillars on either side of the trajectory
        half_gap = corridor_gap / 2.0
        p_left = cpos[:2] + normal_xy * half_gap
        p_right = cpos[:2] - normal_xy * half_gap

        # Crossbar connecting the tops
        crossbar_z = cpos[2] + corridor_gap / 2.0 + 0.5

        print()
        print(f"  # ── VERTICAL CORRIDOR {ci+1}: {label} ──")
        print(f"  # Trajectory at [{cpos[0]:.1f}, {cpos[1]:.1f}, {cpos[2]:.1f}]")
        print()
        print(f"  # Left pillar")
        print(f"  - <<: *corridor_pillar")
        print(f"    position: [{p_left[0]:.2f}, {p_left[1]:.2f}, 0.00]")
        print(f"    rotation: [0, 0, 0]")
        print()
        print(f"  # Right pillar")
        print(f"  - <<: *corridor_pillar")
        print(f"    position: [{p_right[0]:.2f}, {p_right[1]:.2f}, 0.00]")
        print(f"    rotation: [0, 0, 0]")
        print()
        print(f"  # Crossbar overhead")
        print(f"  - <<: *crossbar")
        print(f"    position: [{cpos[0]:.2f}, {cpos[1]:.2f}, {crossbar_z:.2f}]")
        yaw_rad = np.arctan2(tang_xy[1], tang_xy[0])
        print(f"    rotation: [0, 0, {np.degrees(yaw_rad):.2f}]")


if __name__ == "__main__":
    generate_figure8_env(
        a=10.0,
        b=5.0,
        z_min=1.0,
        z_max=8.0,
        num_gates=8,
        v_ref=2.0,
    )
