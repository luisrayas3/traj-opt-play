#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

import numpy as np
import tesseract_robotics as tr
import tesseract_robotics.tesseract_common as tr_common
import tesseract_robotics.tesseract_environment as tr_env
import tesseract_robotics.tesseract_srdf as tr_srdf
import tesseract_robotics.tesseract_collision as tr_collision
import tesseract_robotics.tesseract_kinematics as tr_kinematics
import tesseract_robotics.tesseract_motion_planners as tr_planners
import tesseract_robotics.tesseract_task_composer as tr_task_composer
import tesseract_robotics.tesseract_command_language as tr_cmd
import tesseract_robotics_viewer as tr_viewer


def load_urdf_from_xacro(xacro_filepath):
    """
    Load URDF from xacro file
    """
    try:
        # Set ROS_PACKAGE_PATH to help xacro find packages
        env = os.environ.copy()
        env["ROS_PACKAGE_PATH"] = "/app/universal_robot"

        return subprocess.run(
            ["xacro", str(xacro_filepath)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"Xacro error: {e.stderr}")
        raise


def create_tesseract_environment(
    urdf_filepath: str,
) -> tr_env.Environment:
    """
    Create and initialize Tesseract environment with UR5e
    """
    # Read the URDF file
    with open(urdf_filepath, "r") as f:
        urdf_string = f.read()

    # Replace package:// URIs with absolute file paths
    urdf_string = urdf_string.replace(
        "package://ur_description", "file:///app/universal_robot/ur_description"
    )

    env = tr_env.Environment()
    if not env.init(urdf_string, tr_common.GeneralResourceLocator()):
        raise RuntimeError(
            "Failed to initialize Tesseract environment with URDF"
        )

    return env


joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def run_example_planning(env, manipulator_info, initial_joints):
    goal_joints = np.array([1.57, -1.0, 0.5, -2.0, 0.0, 0.0])

    task_data = tr_task_composer.TaskComposerDataStorage()
    task_data.setData("environment", tr_env.AnyPoly_wrap_EnvironmentConst(env))
    task_data.setData(
        "profiles",
        tr_cmd.AnyPoly_wrap_ProfileDictionary(tr_cmd.ProfileDictionary()),
    )

    # Create composite instruction (program)
    program = tr_cmd.CompositeInstruction(
        "DEFAULT", manipulator_info, tr_cmd.CompositeInstructionOrder_ORDERED
    )

    start_waypoint_poly = tr_cmd.StateWaypointPoly_wrap_StateWaypoint(
        tr_cmd.StateWaypoint(joint_names, initial_joints)
    )
    start_instruction = tr_cmd.MoveInstruction(
        start_waypoint_poly, tr_cmd.MoveInstructionType_FREESPACE, "DEFAULT"
    )
    start_instruction_poly = tr_cmd.MoveInstructionPoly_wrap_MoveInstruction(
        start_instruction
    )
    program.appendMoveInstruction(start_instruction_poly)

    goal_waypoint_poly = tr_cmd.StateWaypointPoly_wrap_StateWaypoint(
        tr_cmd.StateWaypoint(joint_names, goal_joints)
    )
    goal_instruction = tr_cmd.MoveInstruction(
        goal_waypoint_poly, tr_cmd.MoveInstructionType_FREESPACE, "DEFAULT"
    )
    goal_instruction_poly = tr_cmd.MoveInstructionPoly_wrap_MoveInstruction(
        goal_instruction
    )
    program.appendMoveInstruction(goal_instruction_poly)

    # Create task composer factory with config file
    config_path = Path(__file__).parent / "task_composer_plugins.yaml"
    factory = tr_task_composer.TaskComposerPluginFactory(
        tr_common.FilesystemPath(str(config_path)),
        tr_common.GeneralResourceLocator(),
    )

    # Create TrajOpt pipeline task
    task = factory.createTaskComposerNode("TrajOptPipeline")
    if task is None:
        raise RuntimeError("Failed to create TrajOpt pipeline task")

    # Set the program in task data
    task_data.setData(
        "planning_input", tr_cmd.AnyPoly_wrap_CompositeInstruction(program)
    )

    # Create executor
    task_executor = factory.createTaskComposerExecutor("TaskflowExecutor")
    if task_executor is None:
        raise RuntimeError("Failed to create task executor")

    # Execute planning
    future = task_executor.run(task.get(), task_data)
    future.wait()

    if not future.context.isSuccessful():
        print(f"Planning failed with status: {future.context.isAborted()}")
        print(f"Task info: {task.getName() if hasattr(task, 'getName') else 'unknown'}")
        return None
    output_key = task.getOutputKeys().get("program")
    return tr_cmd.AnyPoly_as_CompositeInstruction(
        future.context.data_storage.getData(output_key)
    )


def main():
    # Enable debug logging
    os.environ["TRAJOPT_LOG_THRESH"] = "INFO"

    print("Creating Tesseract environment...")
    env = create_tesseract_environment("/app/src/ur5e.urdf")
    print("...Tesseract environment created!")

    manipulator_info = tr_common.ManipulatorInfo()
    manipulator_info.manipulator = "manipulator"
    manipulator_info.working_frame = "base_link"
    manipulator_info.tcp_frame = "tool0"

    initial_joints = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    viewer = tr_viewer.TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])
    viewer.update_joint_positions(joint_names, initial_joints)
    viewer.start_serve_background()

    print("Planning...")
    print("")
    result = run_example_planning(env, manipulator_info, initial_joints)
    if result is None:
        print("Planning failed!")
        return

    print("Planning success!")
    for instruction_poly in result:
        assert instruction_poly.isMoveInstruction()
        wp_poly = tr_cmd.InstructionPoly_as_MoveInstructionPoly(
            instruction_poly
        ).getWaypoint()
        assert wp_poly.isStateWaypoint()
        wp = tr_cmd.WaypointPoly_as_StateWaypointPoly(wp_poly)
        print(f"Waypoint: t={wp.getTime()}, j={wp.getPosition().flatten()}")

    viewer.update_trajectory(result)
    viewer.plot_trajectory(result, manipulator_info)


if __name__ == "__main__":
    main()
