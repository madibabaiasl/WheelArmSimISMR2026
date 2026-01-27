# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from typing import List

import omni.ui as ui
from omni.isaac.ui.element_wrappers import (
    Button,
    CheckBox,
    CollapsableFrame,
    ColorPicker,
    DropDown,
    FloatField,
    IntField,
    StateButton,
    StringField,
    TextBlock,
    XYPlot,
)
from omni.isaac.ui.ui_utils import get_style

import sys

# Determine the path to the 'scripts' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))  # Adjust based on structure
scripts_dir = os.path.join(project_root, 'scripts')

# Add 'scripts' directory to sys.path if not already present
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import the data_collection module
try:
    from collect_data import start_save_data_in_thread
except ImportError as e:
    print(f"Error importing data_collection: {e}")
    start_save_data_in_thread = None


class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []

        # UI elements created using a UIElementWrapper from omni.isaac.ui.element_wrappers
        self.wrapped_ui_elements = []
        self.task_labels = []
        
        # References to the save data instance and thread
        self.start_data_instance = None
        self.ros_thread = None

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        pass

    def on_physics_step(self, step):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        pass

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        # None of the UI elements in this template actually have any internal state that needs to be cleaned up.
        # But it is best practice to call cleanup() on all wrapped UI elements to simplify development.
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        # Create a UI frame that prints the latest UI event.
        self._create_status_report_frame()

        # Create a UI frame demonstrating simple UI elements for user input
        self._create_simple_editable_fields_frame()

        # Create a UI frame with different button types
        self._create_start_frame()

        # Create a UI frame with different selection widgets
        # self._create_selection_widgets_frame()

        # Create a UI frame with different plotting tools
        # self._create_plotting_frame()

    def _create_status_report_frame(self):
        self._status_report_frame = CollapsableFrame("Status Report", collapsed=False)
        with self._status_report_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._status_report_field = TextBlock(
                    "Last UI Event",
                    num_lines=3,
                    tooltip="Prints the latest change to this UI",
                    include_copy_button=True,
                )

    def _create_simple_editable_fields_frame(self):
        self._simple_fields_frame = CollapsableFrame("Simple Editable Fields", collapsed=False)

        with self._simple_fields_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                def is_usd_or_python_path(file_path: str):
                    # Filter file paths shown in the file picker to only be USD or Python files
                    _, ext = os.path.splitext(file_path.lower())
                    return ext == ".usd" or ext == ".py"
                
                self.task_labels = StringField(
                    "File name",
                    default_value="Type Here or Use File Picker on the Right",
                    tooltip="Type a string or use the file picker to set a value",
                    read_only=False,
                )
                self.wrapped_ui_elements.append(self.task_labels)

                self.instructions = StringField(
                    "Instructions",
                    default_value="Please input your instructions",
                    tooltip="Type a string or use the file picker to set a value",
                    read_only=False,
                )
                self.wrapped_ui_elements.append(self.instructions)

    def _create_start_frame(self):
        buttons_frame = CollapsableFrame("Collection Panel", collapsed=False)

        with buttons_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                collect_button = Button(
                    "Collect Data",
                    "Start",
                    tooltip="Click This Button to activate a callback function",
                    on_click_fn=self._on_button_clicked_fn,
                )
                self.wrapped_ui_elements.append(collect_button)

                stop_button = Button(
                    "Stop Collecting Data",
                    "Stop",
                    tooltip="Click This Button to stop data collection",
                    on_click_fn=self._on_stop_button_clicked_fn,
                )
                self.wrapped_ui_elements.append(stop_button)

    ######################################################################################
    # Functions Below This Point Are Callback Functions Attached to UI Element Wrappers
    ######################################################################################

    def _on_int_field_value_changed_fn(self, new_value: int):
        status = f"Value was changed in int field to {new_value}"
        self._status_report_field.set_text(status)

    def _on_float_field_value_changed_fn(self, new_value: float):
        status = f"Value was changed in float field to {new_value}"
        self._status_report_field.set_text(status)

    def _on_string_field_value_changed_fn(self, new_value: str):
        status = f"Value was changed in string field to {new_value}"
        self._status_report_field.set_text(status)

    def _on_button_clicked_fn(self):
        labels = self.task_labels.get_value()
        instructions = self.instructions.get_value()
        if start_save_data_in_thread:
            if self.ros_thread is None or not self.ros_thread.is_alive():
                self.start_data_instance, self.ros_thread, self.shutdown_event = start_save_data_in_thread(labels, instructions)
                self._status_report_field.set_text("Data Collection Started")
            else:
                self._status_report_field.set_text("Data Collection Already Running")
        else:
            self._status_report_field.set_text("Data Collection Failed to Start")
    
    def _on_stop_button_clicked_fn(self):
        if self.start_data_instance and self.ros_thread and self.ros_thread.is_alive():
            # Stop data collection
            self.start_data_instance.stop_data_collection()
            # Stop ros thread
            self.shutdown_event.set() 
            # Wait for ROS thread to join
            self.ros_thread.join() 
            # Shut down ros
            self.start_data_instance.shutdown_thread()  
            # reset parameters
            self.start_data_instance = None
            self.ros_thread = None
            self.shutdown_event = None
            self._status_report_field.set_text("Data Collection Stopped")
        else:
            self._status_report_field.set_text("Data Collection Not Running")

    def _on_state_btn_a_click_fn(self):
        status = "State Button was Clicked in State A!"
        self._status_report_field.set_text(status)

    def _on_state_btn_b_click_fn(self):
        status = "State Button was Clicked in State B!"
        self._status_report_field.set_text(status)

    def _on_checkbox_click_fn(self, value: bool):
        status = f"CheckBox was set to {value}!"
        self._status_report_field.set_text(status)

