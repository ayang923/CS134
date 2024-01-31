"""Launch the Goals3 demo

This instantiates the HEBI node and runs the Goals3 demo.

This should start
     1) RVIZ, ready to view the robot
     2) The robot_state_publisher (listening to /joint_states)
     3) The HEBI node to communicate with the motors
     4) Goals3, commanding the desired trajectory

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
    rvizcfg = os.path.join(pkgdir('basic134'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('basic134'), 'urdf/134threedof.urdf')
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

    # Configure a node for the hebi interface.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'testmode': 'off'},
                      {'family':   'robotlab'},
                      {'motors':   ['6.3',  '6.5',      '6.4']},
                      {'joints':   ['base', 'shoulder', 'elbow']}],
        on_exit    = Shutdown())

    # Configure a node for the simple demo.
    node_goals3 = Node(
        name       = 'goals', 
        package    = 'threedof',
        executable = 'goals3',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the state publisher, rviz, hebi and demo nodes.
        node_robot_state_publisher_ACTUAL,
        node_rviz,
        node_hebi,
        node_goals3,
    ])