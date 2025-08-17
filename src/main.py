#!/usr/bin/env python3

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import tesseract_robotics as tr
import tesseract_robotics.tesseract_common as tr_common
import tesseract_robotics.tesseract_environment as tr_environment
import tesseract_robotics.tesseract_scene_graph as tr_scene_graph
import tesseract_robotics.tesseract_srdf as tr_srdf
import tesseract_robotics.tesseract_collision as tr_collision
import tesseract_robotics.tesseract_kinematics as tr_kinematics
import tesseract_robotics.tesseract_motion_planners as tr_planners
import tesseract_robotics.tesseract_task_composer as tr_task_composer
import tesseract_robotics.tesseract_command_language as tr_command_lang


def load_urdf_from_xacro(xacro_file_path):
    """
    Load URDF from xacro file
    """
    try:
        result = subprocess.run(
            ["xacro", str(xacro_file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Xacro error: {e.stderr}")
        raise


def create_tesseract_environment():
    """
    Create and initialize Tesseract environment with UR5e
    """
    urdf_path = Path(__file__).parent / "ur5e.urdf"

    with open(urdf_path, "r") as f:
        urdf_string = f.read()

    tr_env = tr_environment.Environment()
    resource_locator = tr_common.GeneralResourceLocator()

    success = tr_env.init(urdf_string, resource_locator)
    if not success:
        raise RuntimeError(
            "Failed to initialize Tesseract environment with URDF"
        )

    return tr_env


def setup_basic_trajopt_problem(env):
    """
    Simple TrajOpt example call
    """
    initial_joints = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    goal_joints = np.array([1.57, -1.0, 0.5, -2.0, 0.0, 0.0])

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    task_data = tr_task_composer.TaskComposerDataStorage()

    manipulator_info = tr_common.ManipulatorInfo()
    manipulator_info.manipulator = "manipulator"
    manipulator_info.working_frame = "base_link"
    manipulator_info.tcp_frame = "tool0"

    start_waypoint = tr_command_lang.StateWaypoint(joint_names, start_joints)
    start_waypoint_poly = tr_command_lang.StateWaypointPoly_wrap_StateWaypoint(
        start_waypoint
    )
    start_instruction = tr_command_lang.MoveInstruction(
        start_waypoint_poly,
        tr_command_lang.MoveInstructionType_FREESPACE,
        "DEFAULT",
    )

    goal_waypoint = tr_command_lang.StateWaypoint(joint_names, goal_joints)
    goal_waypoint_poly = tr_command_lang.StateWaypointPoly_wrap_StateWaypoint(
        goal_waypoint
    )
    goal_instruction = tr_command_lang.MoveInstruction(
        goal_waypoint_poly,
        tr_command_lang.MoveInstructionType_FREESPACE,
        "DEFAULT",
    )

    # Create composite instruction (program)
    program = tr_command_lang.CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manipulator_info)
    start_instruction_poly = (
        tr_command_lang.MoveInstructionPoly_wrap_MoveInstruction(
            start_instruction
        )
    )
    goal_instruction_poly = (
        tr_command_lang.MoveInstructionPoly_wrap_MoveInstruction(
            goal_instruction
        )
    )
    program.appendMoveInstruction(start_instruction_poly)
    program.appendMoveInstruction(goal_instruction_poly)

    # Set the program in task data - wrap in AnyPoly
    program_poly = tr_command_lang.AnyPoly_wrap_CompositeInstruction(program)
    task_data.setData("input", program_poly)

    # Create task composer factory
    factory = tr_task_composer.TaskComposerPluginFactory()

    # Create TrajOpt pipeline task
    task = factory.createTaskComposerTask("TrajOptPipeline")
    if task is None:
        raise RuntimeError("Failed to create TrajOpt pipeline task")

    # Create executor
    task_executor = factory.createTaskComposerExecutor("TaskflowExecutor")
    if task_executor is None:
        raise RuntimeError("Failed to create task executor")

    # Create context and set environment
    context = tr_task_composer.TaskComposerContext()
    context.data_storage = task_data
    # TODO: Figure out how to properly set environment in context

    # Execute planning
    future = task_executor.run(task.get(), context)
    future.wait()

    return future.context


def main():
    """
    Main function to demonstrate Tesseract TrajOpt
    """
    print("Tesseract environment initializing...")
    env = create_tesseract_environment()
    print("...Tesseract environment created successfully!")

    context = setup_basic_trajopt_problem(env)

    if context.isSuccessful():
        print("Trajectory optimization successful!")
        try:
            # Try to get trajectory data
            output_keys = context.data_storage.getKeys()
            print(f"Available output keys: {list(output_keys)}")
            if output_keys:
                trajectory_data = context.data_storage.getData(
                    list(output_keys)[0]
                )
                print(f"Found trajectory data: {type(trajectory_data)}")
        except Exception as e:
            print(f"Could not extract trajectory details: {e}")
    else:
        error_msg = getattr(context, "message", "Unknown error")
        print(f"Trajectory optimization failed: {error_msg}")


if __name__ == "__main__":
    main()
