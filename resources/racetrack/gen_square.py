import numpy as np


def get_stable_kinematics(theta, a=6.0, b=6.0, p=4.0, v_ref=2.0):
    # 1. Position
    def pos_fn(angle):
        cp = np.cos(angle)
        sp = np.sin(angle)
        x = a * np.sign(cp) * (np.abs(cp) ** (2 / p))
        y = b * np.sign(sp) * (np.abs(sp) ** (2 / p))
        return np.array([x, y, 0.0])

    # 2. Velocity (Directional)
    eps = 1e-4
    p_now = pos_fn(theta)
    p_next = pos_fn(theta + eps)

    tangent = (p_next - p_now) / eps
    unit_tangent = tangent / np.linalg.norm(tangent)
    velocity_vec = unit_tangent * v_ref

    # 3. Acceleration (Numerical Centripetal)
    # a = dv/dt. We approximate dt as dist / v_ref
    p_prev = pos_fn(theta - eps)
    v_prev_dir = (p_now - p_prev) / eps
    v_prev = (v_prev_dir / np.linalg.norm(v_prev_dir)) * v_ref

    # dt calculation: arc_length / speed
    dt = np.linalg.norm(p_next - p_now) / v_ref
    accel_vec = (velocity_vec - v_prev) / dt

    # 4. Jerk (Rate of change of acceleration)
    # For a traj-gen starting point, we can damp this or use a wider step
    # to avoid numerical noise.
    p_next2 = pos_fn(theta + 2 * eps)
    v_next_dir = (p_next2 - p_next) / eps
    v_next = (v_next_dir / np.linalg.norm(v_next_dir)) * v_ref
    accel_next = (v_next - velocity_vec) / dt
    jerk_vec = (accel_next - accel_vec) / dt

    # 5. Orientation
    yaw_rad = np.arctan2(velocity_vec[1], velocity_vec[0])
    yaw_deg = np.degrees(yaw_rad)
    quat = [np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]

    return p_now, velocity_vec, accel_vec, jerk_vec, yaw_deg, quat


def generate_fixed_env(side=6.0, p=4.0, num_gates=12, height=1.2, v_ref=2.0):
    thetas = np.linspace(0, 2 * np.pi, num_gates, endpoint=False)

    # initState from theta = 0
    p0, v0, a0, j0, y0, q0 = get_stable_kinematics(0, side, side, p, v_ref)

    print(f"initState:")
    print(f"  pos: [{p0[0]:.2f}, {p0[1]:.2f}, {height:.2f}]")
    print(f"  vel: [{v0[0]:.2f}, {v0[1]:.2f}, 0.00]")
    print(f"  acc: [{a0[0]:.2f}, {a0[1]:.2f}, 0.00]")
    print(f"  jer: [{j0[0]:.2f}, {j0[1]:.2f}, 0.00]")
    print(f"  rot: [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]")
    print(f"  cthrustmass: 9.8066\n  euler: [0.0, 0.0, {y0:.2f}]\n")

    print(f"orders: {[f'Gate{i+1}' for i in range(1, num_gates)]}\n")

    for i in range(1, num_gates):
        pos, _, _, _, yaw, _ = get_stable_kinematics(thetas[i], side, side, p, v_ref)
        yaw_w = (yaw + 180) % 360 - 180
        print(f"Gate{i+1}:\n  type: 'RectanglePrisma'\n  name: 'vicon_gate'")
        print(f"  position: [{pos[0]:.2f}, {pos[1]:.2f}, {height:.2f}]")
        print(f"  rpy: [0.0, -90.0, {yaw_w:.2f}]")
        print(f"  width: 2.4\n  height: 2.4\n  stationary: true\n")


if __name__ == "__main__":
    generate_fixed_env(side=6.0, p=4.0, num_gates=12, v_ref=2.0)
