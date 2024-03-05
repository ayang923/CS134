"""
Launch the Goals4 demo.

This instantiates the HEBI node and runs the Goals4 demo.
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

    # Configure a node for RVIZ
    node_rviz = Node(
       name       = 'rviz', 
       package    = 'rviz2',
       executable = 'rviz2',
       output     = 'screen',
       arguments  = ['-d', rvizcfg],
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

    # Configure the top-down USB camera node
    node_usbcam_top = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = [{'camera_name':         'logitech'},
                      {'video_device':        '/dev/video0'},
                      {'pixel_format':        'yuyv2rgb'},
                      {'image_width':         960},
                      {'image_height':        720},
                      {'framerate':           15.0},
                      {'brightness':          -1},
                      {'contrast':            -1},
                      {'saturation':          -1},
                      {'sharpness':           -1},
                      {'gain':                -1},
                      {'auto_white_balance':  False},
                      {'white_balance':       4000},
                      {'autoexposure':        False},
                      {'exposure':            10},
                      {'autofocus':           False},
                      {'focus':               -1}])
    
        # Configure the USB camera node
    node_usbcam_wrist = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'tip_cam',
        output     = 'screen',
        parameters = [{'camera_name':         'logitech'},
                    {'video_device':        '/dev/video2'},
                    {'pixel_format':        'yuyv2rgb'},
                    {'image_width':         640},
                    {'image_height':        480},
                    {'framerate':           15.0},
                    {'brightness':          -1},
                    {'contrast':            -1},
                    {'saturation':          -1},
                    {'sharpness':           -1},
                    {'gain':                -1},
                    {'auto_white_balance':  False},
                    {'white_balance':       4000},
                    {'autoexposure':        False},
                    {'exposure':            30},
                    {'autofocus':           False},
                    {'focus':               -1}])

    # Configure a node for the simple demo.
    node_traj = Node(
        name       = 'traj', 
        package    = 'sixdof',
        executable = 'traj',
        output     = 'screen')
    
    node_det = Node(
        name       = 'det', 
        package    = 'sixdof',
        executable = 'det',
        output     = 'screen')
    
    node_game = Node(
        name       = 'game', 
        package    = 'sixdof',
        executable = 'game',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([
        # Start the state publisher, rviz, hebi and demo nodes.
        node_robot_state_publisher_ACTUAL,
        node_hebi,
        node_traj,
    ])
