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


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

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
    node_detector = Node(
        name       = 'detectors', 
        package    = 'sixdof',
        executable = 'detectors',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the state publisher, rviz, hebi and demo nodes.
        node_usbcam_top,
        node_usbcam_wrist,
        node_detector
    ])
