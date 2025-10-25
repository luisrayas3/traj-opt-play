#!/usr/bin/env python3

import numpy as np
import tesseract_robotics as tr
import tesseract_robotics.tesseract_common as tr_common
import tesseract_robotics.tesseract_environment as tr_env
import tesseract_robotics.tesseract_command_language as tr_cmd
from tesseract_robotics.tesseract_command_language import (
    PlanInstructionType_START,
    PlanInstructionType_FREESPACE,
    MoveInstructionType_FREESPACE,
)


def create_seed_from_ik(
    env, manipulator, tcp_frame, start_pose, goal_pose, n_steps=15
):
    """
    Generate seed trajectory by solving IK along straight-line Cartesian path.

    Returns:
        seed_trajectory: (n_steps x n_dof) numpy array of joint positions
    """
    # Get kinematics
    kin_group = env.getKinematicGroup(manipulator)
    inv_kin = kin_group.getInvKinematics()
    joint_names = kin_group.getJointNames()
    n_dof = len(joint_names)

    # Get current state as seed
    current_state = env.getState()
    seed = np.array([current_state.getJointPosition(jn) for jn in joint_names])

    # Generate Cartesian waypoints (linear interpolation)
    seed_trajectory = np.zeros((n_steps, n_dof))

    for i in range(n_steps):
        t = i / (n_steps - 1)

        # Linear interpolation of pose (simplified - doesn't handle rotation properly)
        # For production, use SLERP for rotation
        waypoint_pose = tr_common.Isometry3d()
        waypoint_pose.matrix = (1 - t) * start_pose + t * goal_pose

        # Solve IK
        target_poses = {tcp_frame: waypoint_pose}
        solutions = inv_kin.calcInvKin(target_poses, seed)

        if len(solutions) > 0:
            seed_trajectory[i, :] = solutions[0]
            seed = solutions[0]  # Use for next waypoint
        else:
            print(f"Warning: IK failed at step {i}")
            seed_trajectory[i, :] = seed  # Keep previous

    return seed_trajectory


def create_trajopt_profiles(joint_names, look_vector_mode=True):
    """
    Create TrajOpt plan and composite profiles.

    Returns:
        plan_profile: TrajOptDefaultPlanProfile
        composite_profile: TrajOptDefaultCompositeProfile
    """
    n_dof = len(joint_names)

    # Plan profile - per-waypoint constraints
    plan_profile = tr_cmd.TrajOptDefaultPlanProfile()

    if look_vector_mode:
        # Look-vector: constrain position + 2 rotations, free 1 rotation
        plan_profile.cartesian_coeff = tr_cmd.Eigen_VectorXd(
            [
                10.0,
                10.0,
                10.0,  # Position (X, Y, Z)
                15.0,
                15.0,
                0.0,  # Rotation (Rx, Ry, Rz) - Z free
            ]
        )
    else:
        # Full 6-DOF constraint
        plan_profile.cartesian_coeff = tr_cmd.Eigen_VectorXd([10.0] * 6)

    plan_profile.term_type = tr_cmd.TrajOptTermType_TT_COST

    # Composite profile - trajectory-wide optimization
    composite_profile = tr_cmd.TrajOptDefaultCompositeProfile()

    # Time optimization
    composite_profile.smooth_velocities = True
    composite_profile.smooth_accelerations = True
    composite_profile.smooth_jerks = True
    composite_profile.velocity_coeff = tr_cmd.Eigen_VectorXd([2.0] * n_dof)
    composite_profile.acceleration_coeff = tr_cmd.Eigen_VectorXd([8.0] * n_dof)
    composite_profile.jerk_coeff = tr_cmd.Eigen_VectorXd([5.0] * n_dof)

    # Collision avoidance
    composite_profile.collision_cost_config.enabled = True
    composite_profile.collision_cost_config.type = (
        tr_cmd.CollisionEvaluatorType_DISCRETE_CONTINUOUS
    )
    composite_profile.collision_cost_config.safety_margin = 0.020  # 20mm
    composite_profile.collision_cost_config.safety_margin_buffer = 0.010  # 10mm
    composite_profile.collision_cost_config.coeff = 25.0

    return plan_profile, composite_profile


def build_program(
    start_pose, goal_pose, seed_trajectory, manipulator, tcp_frame, base_frame
):
    """
    Build CompositeInstruction program for TrajOpt.

    Args:
        start_pose: 4x4 numpy array
        goal_pose: 4x4 numpy array
        seed_trajectory: (n_steps x n_dof) joint positions
        manipulator: manipulator group name
        tcp_frame: TCP frame name
        base_frame: working frame name

    Returns:
        program: CompositeInstruction
    """
    program = tr_cmd.CompositeInstruction()

    # Set manipulator info
    manip_info = tr_cmd.ManipulatorInfo()
    manip_info.manipulator = manipulator
    manip_info.tcp_frame = tcp_frame
    manip_info.working_frame = base_frame
    program.setManipulatorInfo(manip_info)

    # Start instruction
    start_wp = tr_cmd.CartesianWaypoint()
    start_wp.setTransform(tr_common.Isometry3d(start_pose))

    start_instr = tr_cmd.MoveInstruction(
        start_wp, MoveInstructionType_FREESPACE
    )
    start_instr.setMoveType(tr_cmd.MoveInstructionType_START)
    program.append(start_instr)

    # Intermediate waypoints using seed
    n_steps = seed_trajectory.shape[0]
    for i in range(1, n_steps - 1):
        joint_wp = tr_cmd.JointWaypoint()
        joint_wp.setPositions(seed_trajectory[i, :])

        move_instr = tr_cmd.MoveInstruction(
            joint_wp, MoveInstructionType_FREESPACE
        )
        program.append(move_instr)

    # Goal instruction
    goal_wp = tr_cmd.CartesianWaypoint()
    goal_wp.setTransform(tr_common.Isometry3d(goal_pose))

    goal_instr = tr_cmd.MoveInstruction(goal_wp, MoveInstructionType_FREESPACE)
    program.append(goal_instr)

    return program


def execute_trajopt_planning(env, program, plan_profile, composite_profile):
    """
    Execute TrajOpt motion planner.

    Returns:
        trajectory: (n_steps x n_dof) optimized joint trajectory or None
    """
    # Create profile dictionary
    profiles = tr_cmd.ProfileDictionary()
    profiles.addProfile(
        tr_cmd.TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", plan_profile
    )
    profiles.addProfile(
        tr_cmd.TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", composite_profile
    )

    # Create planning request
    request = tr_cmd.PlannerRequest()
    request.env = env
    request.env_state = env.getState()
    request.instructions = program
    request.profiles = profiles

    # Create and execute planner
    planner = tr_cmd.TrajOptMotionPlanner()

    try:
        response = planner.solve(request)

        if response.successful:
            print("Planning successful!")

            # Extract trajectory
            results = response.results
            n_steps = len(results)

            # Get number of joints from first waypoint
            first_wp = results[0].getWaypoint()
            if hasattr(first_wp, "position"):
                n_dof = len(first_wp.position)
            else:
                n_dof = len(first_wp.getPositions())

            trajectory = np.zeros((n_steps, n_dof))

            for i in range(n_steps):
                wp = results[i].getWaypoint()
                if hasattr(wp, "position"):
                    trajectory[i, :] = wp.position
                else:
                    trajectory[i, :] = wp.getPositions()

            return trajectory
        else:
            print(f"Planning failed: {response.message}")
            return None

    except Exception as e:
        print(f"Exception during planning: {e}")
        import traceback

        traceback.print_exc()
        return None


def plan_trajectory(
    urdf_string,
    start_pose,
    goal_pose,
    manipulator="manipulator",
    tcp_frame="tool0",
    base_frame="base_link",
    n_steps=15,
    enable_look_vector=True,
):
    """
    Complete planning workflow.

    Args:
        urdf_string: Robot URDF as string
        start_pose: 4x4 numpy array for start pose
        goal_pose: 4x4 numpy array for goal pose
        manipulator: Manipulator group name
        tcp_frame: TCP frame name
        base_frame: World/base frame name
        n_steps: Number of trajectory waypoints
        enable_look_vector: Use look-vector constraint (2 of 3 rotation axes)

    Returns:
        trajectory: (n_steps x n_dof) optimized joint trajectory or None
    """
    # Step 1: Setup environment
    print("[1/5] Initializing environment...")
    env = setup_environment(urdf_string)

    # Step 2: Generate IK seed
    print("[2/5] Generating IK seed trajectory...")
    seed_trajectory = create_seed_from_ik(
        env, manipulator, tcp_frame, start_pose, goal_pose, n_steps
    )
    print(f"      Seed shape: {seed_trajectory.shape}")

    # Step 3: Create TrajOpt profiles
    print("[3/5] Configuring TrajOpt profiles...")
    joint_names = env.getKinematicGroup(manipulator).getJointNames()
    plan_profile, composite_profile = create_trajopt_profiles(
        joint_names, look_vector_mode=enable_look_vector
    )

    # Step 4: Build instruction program
    print("[4/5] Building instruction program...")
    program = build_program(
        start_pose,
        goal_pose,
        seed_trajectory,
        manipulator,
        tcp_frame,
        base_frame,
    )

    # Step 5: Execute TrajOpt
    print("[5/5] Running TrajOpt optimization...")
    trajectory = execute_trajopt_planning(
        env, program, plan_profile, composite_profile
    )

    if trajectory is not None:
        print(f"{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Number of waypoints: {trajectory.shape[0]}")
        print(f"Number of joints: {trajectory.shape[1]}")
    else:
        print(f"{'='*60}")
        print("PLANNING FAILED")
        print(f"{'='*60}")

    return trajectory


if __name__ == "__main__":
    # Define start and goal poses
    start_pose = np.eye(4)
    start_pose[:3, 3] = [0.5, 0.0, 0.5]

    goal_pose = np.eye(4)
    goal_pose[:3, 3] = [0.5, 0.3, 0.8]

    # Plan trajectory
    trajectory = plan_trajectory(
        urdf_string=urdf_string,
        start_pose=start_pose,
        goal_pose=goal_pose,
        manipulator="manipulator",
        tcp_frame="tool0",
        base_frame="base_link",
        n_steps=15,
        enable_look_vector=True,
    )

    if trajectory is not None:
        print("\nTrajectory ready for execution!")
        print(f"First waypoint: {trajectory[0, :]}")
        print(f"Last waypoint: {trajectory[-1, :]}")
