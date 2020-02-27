import glob
import json
import os
from PIL import Image

import ai2thor.controller

from mcs_action import MCS_Action
from mcs_controller import MCS_Controller
from mcs_goal import MCS_Goal
from mcs_pose import MCS_Pose
from mcs_return_status import MCS_Return_Status
from mcs_step_output import MCS_Step_Output
from mcs_util import MCS_Util

class MCS_Controller_AI2THOR(MCS_Controller):
    """
    MCS Controller class implementation for the AI2-THOR library.

    https://ai2thor.allenai.org/ithor/documentation/
    """

    ACTION_LIST = [item.value for item in MCS_Action]

    # AI2-THOR creates a square grid across the scene that is uses for "snap-to-grid" movement.
    # (This value may not really matter because we set continuous to True in the step input.)
    GRID_SIZE = 0.1

    # How far the player can move with a single step.
    MOVE_DISTANCE = 0.2

    # How far the player can reach.  I think this value needs to be bigger than the MOVE_DISTANCE or else the player
    # may not be able to move into a position to reach some objects (it may be mathematically impossible).
    REACH_DISTANCE = 0.4

    MAX_ROTATION = 360
    MIN_ROTATION = -360
    MAX_HORIZON = 90
    MIN_HORIZON = -90

    ROTATION_KEY = 'rotation'
    HORIZON_KEY = 'horizon'

    def __init__(self, unity_app_file_path, debug=False):
        super().__init__()

        self.__controller = ai2thor.controller.Controller(
            quality='Medium',
            fullscreen=False,
            # The headless flag does not work for me
            headless=False,
            local_executable_path=unity_app_file_path,
            width=600,
            height=400,
            # Set the name of our Scene in our Unity app
            scene='MCS',
            logs=True,
            # This constructor always initializes a scene, so add a scene config to ensure it doesn't error
            sceneConfig={
                "objects": []
            }
        )

        self.on_init(debug)

    def on_init(self, debug=False):
        self.__debug = debug
        if self.__debug:
            print("===============================================================================")

        self.__current_scene = None
        self.__output_folder = None # Save output image files to debug
        self.__step_number = 0

    # Override
    def end_scene(self, classification, confidence):
        super().end_scene(classification, confidence)
        # TODO MCS-54 Save classification, confidence, and list of actions (steps) taken in this scene for scoring (maybe save to file?)
        pass

    # Override
    def start_scene(self, config_data):
        super().start_scene(config_data)

        self.__current_scene = config_data
        self.__step_number = 0

        if config_data['name'] is not None:
            os.makedirs('./' + config_data['name'], exist_ok=True)
            self.__output_folder = './' + config_data['name'] + '/'
            file_list = glob.glob(self.__output_folder + '*')
            for file_path in file_list:
                os.remove(file_path)

        return self.wrap_output(self.__controller.step(self.wrap_step(action='Initialize', sceneConfig=config_data)))

    """
    Check if value is a number.
    """
    def is_number(self, value):
        try:
            float(value)
        except ValueError:
            return False
        return True

    # TODO: may need to reevaluate validation strategy/error handling in the future
    """
    Need a validation/conversion step for what ai2thor will accept as input
    to keep parameters more simple for the user (in this case, wrapping
    rotation degrees into an object)
    """
    def validate_and_convert_params(self, **kwargs):
        rotation = kwargs.get(self.ROTATION_KEY, 0)
        horizon = kwargs.get(self.HORIZON_KEY, 0)

        if self.is_number(rotation) == False:
           print('Value of rotation needs to be a number. Will be set to 0.')
           rotation = 0

        if self.is_number(horizon) == False:
           print('Value of horizon needs to be a number. Will be set to 0.')
           horizon = 0

        if horizon > self.MAX_HORIZON or horizon < self.MIN_HORIZON:
            print('Value of horizon needs to be between ' + str(self.MIN_HORIZON) + \
                ' and ' + str(self.MAX_HORIZON) + '. Current value: ' + str(horizon)+ \
                '. Will be reset to 0.')
            horizon = 0

        rotation_vector = {}
        rotation_vector['y'] = rotation

        return dict(
            objectId=kwargs.get("objectId", None),
            rotation=rotation_vector,
            horizon=horizon
        )

    # Override
    def step(self, action, **kwargs):
        super().step(action, **kwargs)

        if not action in self.ACTION_LIST:
            print("MCS Warning: The given action '" + action + "' is not valid. Exchanging it with the 'Pass' action.")
            action = "Pass"

        self.__step_number += 1

        if self.__debug:
            print("===============================================================================")
            print("STEP = " + str(self.__step_number))

        params = self.validate_and_convert_params(**kwargs)

        return self.wrap_output(self.__controller.step(self.wrap_step(action=action, **params)))

    def retrieve_goal(self, current_scene):
        # TODO MCS-53 Return goal object from scene configuration data object
        return MCS_Goal()

    def retrieve_head_tilt(self, scene_event):
        return scene_event.metadata['agent']['cameraHorizon']

    def retrieve_object_list(self, scene_event):
        # TODO MCS-52 Return the list of objects in the scene by non-descriptive UUID and their corresponding object metadata like the vector from the player to the object
        return []

    def retrieve_pose(self, scene_event):
        # TODO MCS-18 Return pose from Unity in step output object
        return MCS_Pose.STAND.name

    def retrieve_return_status(self, scene_event):
        # TODO MCS-47 Return proper step status from Unity in step output object
        return MCS_Return_Status.SUCCESSFUL.name if scene_event.metadata['lastActionSuccess'] == True \
                else MCS_Return_Status.UNDEFINED.name

    def save_images(self, scene_event):
        # TODO MCS-51 May have multiple images
        scene_image = Image.fromarray(scene_event.frame)
        # Divide the depth mask by 30 so it doesn't appear all white (some odd side effect of the depth grayscaling).
        depth_mask = Image.fromarray(scene_event.depth_frame / 30)
        depth_mask = depth_mask.convert('L')
        class_mask = Image.fromarray(scene_event.class_segmentation_frame)
        object_mask = Image.fromarray(scene_event.instance_segmentation_frame)

        if self.__output_folder is not None:
            scene_image.save(fp=self.__output_folder + 'frame' + str(self.__step_number) + '.png')
            depth_mask.save(fp=self.__output_folder + 'depth' + str(self.__step_number) + '.png')
            class_mask.save(fp=self.__output_folder + 'class' + str(self.__step_number) + '.png')
            object_mask.save(fp=self.__output_folder + 'object' + str(self.__step_number) + '.png')

        return scene_image, depth_mask, object_mask

    def wrap_output(self, scene_event):
        if self.__output_folder is not None:
            with open(self.__output_folder + 'metadata' + str(self.__step_number) + '.json', 'w') as json_file:
                json.dump({
                    "metadata": scene_event.metadata
                }, json_file, sort_keys=True, indent=4)

        image, depth_mask, object_mask = self.save_images(scene_event)

        step_output = MCS_Step_Output(
            action_list=self.ACTION_LIST,
            depth_mask_list=[depth_mask],
            goal=self.retrieve_goal(self.__current_scene),
            head_tilt=self.retrieve_head_tilt(scene_event),
            image_list=[image],
            object_list=self.retrieve_object_list(scene_event),
            object_mask_list=[object_mask],
            pose=self.retrieve_pose(scene_event),
            return_status=self.retrieve_return_status(scene_event),
            step_number=self.__step_number
        )

        if self.__debug:
            print("MCS STEP OUTPUT = " + str(step_output))

        return step_output

    def wrap_step(self, **kwargs):
        # Create the step data dict for the AI2-THOR step function.
        step_data = dict(
            continuous=True,
            gridSize=self.GRID_SIZE,
            logs=True,
            moveMagnitude=self.MOVE_DISTANCE,
            renderClassImage=True,
            renderDepthImage=True,
            renderObjectImage=True,
            # Yes, in AI2-THOR, the player's reach appears to be governed by the "visibilityDistance", confusingly...
            visibilityDistance=self.REACH_DISTANCE,
            **kwargs
        )

        if self.__debug:
            print("AI2THOR STEP INPUT = " + json.dumps(step_data, sort_keys=True, indent=4))

        return step_data
