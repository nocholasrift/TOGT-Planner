import numpy as np


def euler_to_quat(yaw_deg):
    """Converts Yaw (degrees) to Quaternion [w, x, y, z] for a quadrotor."""
    yaw_rad = np.radians(yaw_deg)
    return [np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]


def generate_smooth_circle(radius=6.0, height=1.2, num_gates=8, v_ref=1.0):
    thetas = np.linspace(0, 2 * np.pi, num_gates, endpoint=False)
    omega = v_ref / radius  # Angular velocity

    # --- Calculate Kinematics for Start/End (at Theta=0) ---
    q0 = euler_to_quat(90.0)  # Tangent at (R, 0) is North (90 deg)

    # State vectors for a circle at t=0
    # Pos: [R, 0, Z]
    # Vel: [0, R*omega, 0]
    # Acc: [-R*omega^2, 0, 0]
    # Jer: [0, -R*omega^3, 0]

    header = f"""initState:
  pos: [{radius:.2f}, 0.00, {height:.2f}]
  vel: [0.00, {radius*omega:.2f}, 0.00]
  acc: [{-radius*(omega**2):.2f}, 0.00, 0.00]
  jer: [0.00, {-radius*(omega**3):.2f}, 0.00]
  rot: [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]
  cthrustmass: 9.8066
  euler: [0.0, 0.0, 90.00]

endState:
  pos: [{radius:.2f}, 0.00, {height:.2f}]
  vel: [0.00, {radius*omega:.2f}, 0.00]
  acc: [{-radius*(omega**2):.2f}, 0.00, 0.00]
  jer: [0.00, {-radius*(omega**3):.2f}, 0.00]
  rot: [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]
  cthrustmass: 9.8066
  euler: [0.0, 0.0, 90.00]
"""
    print(header)
    print(f"orders: {[f'Gate{i+1}' for i in range(num_gates)]}\n")

    # --- Generate Gates ---
    for i, theta in enumerate(thetas):
        px = radius * np.cos(theta)
        py = radius * np.sin(theta)

        # Wrap yaw to [-180, 180] for solver stability
        yaw = np.degrees(theta + np.pi / 2)
        yaw_wrapped = (yaw + 180) % 360 - 180

        print(f"Gate{i+1}:")
        print(f"  type: 'RectanglePrisma'")
        print(f"  name: 'vicon_gate'")
        print(f"  position: [{px:.2f}, {py:.2f}, {height:.2f}]")
        print(f"  rpy: [0.0, -90.0, {yaw_wrapped:.2f}]")
        print(
            f"  width: 2.4\n  height: 2.4\n  marginW: 0.0\n  marginH: 0.0\n  length: 0.0\n  midpoints: 0\n  stationary: true\n"
        )


if __name__ == "__main__":
    generate_smooth_circle(radius=6.0, height=1.2, num_gates=8, v_ref=2.0)
