"""Launch the Goals2 demo

This instantiates the HEBI node and runs the Goals2 trajectory.

"""

from launch                            import LaunchDescription
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the hebi interface.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family': 'robotlab'},
                      {'motors': ['6.3', '6.5', '6.4']},
                      {'joints': ['one', 'two', 'three']}])

    # Configure a node for the simple demo.
    node_goals2 = Node(
        name       = 'goals', 
        package    = 'demo134',
        executable = 'goals2',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the hebi and demo nodes.
        node_hebi,
        node_goals2,
    ])
