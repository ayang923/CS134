"""Show the URDF of the robot (not actual hardware required)

   This should start
     1) RVIZ, ready to view the robot
     2) The robot_state_publisher (listening to /joint_commands)
     3) The GUI to issues commands

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('sixdof'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('sixdof'), 'urdf/fivedof.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher_ACTUAL = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    node_robot_state_publisher_COMMAND = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}],
        remappings = [('/joint_states', '/joint_commands')])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    # Configure a node for the hebi interface.  Note the 200ms timeout
    # is useful as the GUI only runs at 10Hz.
    node_hebi_SLOW = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['6.4',  '6.7', '6.5', '6.2', '6.1']},
                      {'joints':   ['base', 'shoulder', 'elbow', 'wristpitch', 'wristroll']},
                      {'lifetime': 200.0}],
        on_exit    = Shutdown())

    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'testmode': 'off'},
                      {'family':   'robotlab'},
                      {'motors':   ['6.4',  '6.7', '6.5', '6.2', '6.1', '6.3']},
                      {'joints':   ['base', 'shoulder', 'elbow', 'wristpitch', 'wristroll', 'grip']}],
        on_exit    = Shutdown())

    # Configure a node for the GUI to command the robot.
    node_gui = Node(
        name       = 'gui', 
        package    = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output     = 'screen',
        remappings = [('/joint_states', '/joint_commands')],
        on_exit    = Shutdown())


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Use RVIZ to view the URDF commanded by the GUI.
        node_robot_state_publisher_COMMAND,
        node_rviz,
        node_gui,
    ])