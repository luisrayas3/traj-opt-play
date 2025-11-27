#!/usr/bin/env python3

import time
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
import xacrodoc


def load_urdf_from_xacro(xacro_filepath):
    """
    Load URDF from xacro file
    """
    xacrodoc.packages.look_in(["/app/universal_robot"])
    return xacrodoc.XacroDoc.from_file(str(xacro_filepath)).to_urdf_string()


def create_tesseract_environment(
    urdf_filepath: str, srdf_filepath: str
) -> tr_env.Environment:
    """
    Create and initialize Tesseract environment
    """
    if urdf_filepath.endswith(".xacro"):
        urdf_string = load_urdf_from_xacro(urdf_filepath)
    else:
        with open(urdf_filepath, "r") as f:
            urdf_string = f.read()
        # Replace package:// URIs with absolute file paths
        urdf_string = urdf_string.replace(
            "package://ur_description",
            "file:///app/universal_robot/ur_description",
        )

    with open(srdf_filepath, "r") as f:
        srdf_string = f.read()

    locator = tr_common.GeneralResourceLocator()
    env = tr_env.Environment()
    if not env.init(urdf_string, srdf_string, locator):
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


def debug_task_state(context: tr_task_composer.TaskComposerContext):
    print(f"Aborted? {context.isAborted()}")
    aborting_node_info = context.task_infos.getInfo(
        context.task_infos.getAbortingNode()
    )
    print(f"  name={aborting_node_info.name}")
    print(f"  type={aborting_node_info.type}")
    print(f"  status_message={aborting_node_info.status_message}")
    for edge in aborting_node_info.inbound_edges:
        print(f"  {str(edge)}:")
        if edge_info := context.task_infos.getInfo(edge):
            print(f"    name={edge_info.name}")
            print(f"    type={edge_info.type}")
            print(f"    triggers_abort={edge_info.triggers_abort}")
            print(f"    return_value={edge_info.return_value}")
            print(f"    status_code={edge_info.status_code}")
            print(f"    status_message={edge_info.status_message}")
        else:
            print("    null")


def run_example_planning(env, manipulator_info, initial_joints):
    goal_joints = np.array([0.0, 0, 0.0, 0.0, 0.0, 0.5])

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
    config_path = Path(__file__).parent / "task_composer_plugins_trajopt.yaml"
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
    print("Starting actual plan...")
    future = task_executor.run(task.get(), task_data)
    future.wait()
    print("...Finished actual plan")

    if not future.context.isSuccessful():
        print(f"Task '{task.getName()}' failed:")
        debug_task_state(future.context)
        return None
    output_key = task.getOutputKeys().get("program")
    return tr_cmd.AnyPoly_as_CompositeInstruction(
        future.context.data_storage.getData(output_key)
    )


def main():
    print("Creating Tesseract environment...")
    env = create_tesseract_environment(
        "/app/src/system.urdf.xacro", "/app/src/ur5e.srdf"
    )
    print("...Tesseract environment created!")

    manipulator_info = tr_common.ManipulatorInfo()
    manipulator_info.manipulator = "manipulator"
    manipulator_info.working_frame = "base_link"
    manipulator_info.tcp_frame = "tool0"

    initial_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    viewer = tr_viewer.TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])
    viewer.update_joint_positions(joint_names, initial_joints)
    viewer.start_serve_background()

    print("Planning...")
    print("")
    result = run_example_planning(env, manipulator_info, initial_joints)
    if result is None:
        print("Planning failed!")
    else:
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

    try:
        while True:
            time.sleep(0.5)
    except Exception:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
