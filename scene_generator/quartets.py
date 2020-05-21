import copy
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Optional

import intphys_goals


def find_target(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Find a 'target' object in the scene. (IntPhys goals don't really
    have them, but they do have objects that may behave plausibly or
    implausibly.)
    """
    target_id = scene['goal']['metadata']['objects'][0]
    return next((obj for obj in scene['objects'] if obj['id'] == target_id))


class Quartet(ABC):
    def __init__(self, template: Dict[str, Any], find_path: bool):
        self._template = template
        self._find_path = find_path
        self._scenes: List[Optional[Dict[str, Any]]] = [None]*4

    @abstractmethod
    def get_scene(self, q: int) -> Dict[str, Any]:
        pass


class ObjectPermanenceQuartet(Quartet):
    def __init__(self, template: Dict[str, Any], find_path: bool):
        super(ObjectPermanenceQuartet, self).__init__(template, find_path)
        self._goal = intphys_goals.ObjectPermanenceGoal()
        self._scenes[0] = copy.deepcopy(self._template)
        self._goal.update_body(self._scenes[0], self._find_path)

    def _appear_behind_occluder(self, body: Dict[str, Any]) -> None:
        target = find_target(body)
        if self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_and_occluders_moving_across:
            implausible_event_index = target['intphys_option']['occluder_indices'][0]
            implausible_event_step = implausible_event_index + target['forces'][0]['stepBegin']
            implausible_event_x = target['intphys_option']['position_by_step'][implausible_event_index]
            target['shows'][0]['position']['x'] = implausible_event_x
        elif self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_falling_down:
            # 8 is enough steps for anything to fall to the ground
            implausible_event_step = 8 + target['shows'][0]['stepBegin']
            target['shows'][0]['position']['y'] = target['intphys_option']['position_y']
        else:
            raise ValueError('unknown object creation function, cannot update scene')
        target['shows'][0]['stepBegin'] = implausible_event_step

    def _disappear_behind_occluder(self, body: Dict[str, Any]) -> None:
        target = find_target(body)
        if self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_and_occluders_moving_across:
            implausible_event_step = target['intphys_option']['occluder_indices'][0] + \
                target['forces'][0]['stepBegin']
        elif self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_falling_down:
            # 8 is enough steps for anything to fall to the ground
            implausible_event_step = 8 + target['shows'][0]['stepBegin']
        else:
            raise ValueError('unknown object creation function, cannot update scene')
        target['hides'] = [{
            'stepBegin': implausible_event_step
        }]

    def get_scene(self, q: int) -> Dict[str, Any]:
        if q < 1 or q > 4:
            raise ValueError(f'q must be between 1 and 4 (inclusive), not {q}')
        scene = self._scenes[q - 1]
        if scene is None:
            scene = copy.deepcopy(self._scenes[0])
            if q == 2:
                # target moves behind occluder and disappears (implausible)
                scene['answer']['choice'] = 'implausible'
                self._disappear_behind_occluder(scene)
            elif q == 3:
                # target first appears from behind occluder (implausible)
                scene['answer']['choice'] = 'implausible'
                self._appear_behind_occluder(scene)
            elif q == 4:
                # target not in the scene (plausible)
                target_id = scene['goal']['metadata']['objects'][0]
                for i in range(len(scene['objects'])):
                    obj = scene['objects'][i]
                    if obj['id'] == target_id:
                        del scene['objects'][i]
                        break
            self._scenes[q - 1] = scene
        return scene


class SpatioTemporalContinuityQuartet(Quartet):
    def __init__(self, template: Dict[str, Any], find_path: bool):
        super(SpatioTemporalContinuityQuartet, self).__init__(template, find_path)
        self._goal = intphys_goals.SpatioTemporalContinuityGoal()
        self._scenes[0] = copy.deepcopy(self._template)
        self._goal.update_body(self._scenes[0], self._find_path)

    def _teleport_forward(self, scene: Dict[str, Any]) -> None:
        if self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_and_occluders_moving_across:
            # TODO: in MCS-125
            pass
        elif self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_falling_down:
            # TODO: in MCS-132
            pass
        else:
            raise ValueError('unknown object creation function, cannot update scene')

    def _teleport_backward(self, scene: Dict[str, Any]) -> None:
        if self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_and_occluders_moving_across:
            # TODO: in MCS-125
            pass
        elif self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_falling_down:
            # TODO: in MCS-132

    def _move_later(self, scene: Dict[str, Any]) -> None:
        if self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_and_occluders_moving_across:
            # TODO: in MCS-125
            pass
        elif self._goal._object_creator == intphys_goals.IntPhysGoal._get_objects_falling_down:
            # TODO: in MCS-132

    def get_scene(self, q: int) -> Dict[str, Any]:
        if q < 1 or q > 4:
            raise ValueError(f'q must be between 1 and 4 (inclusive), not {q}')
        if self._scenes[q - 1] is None:
            scene = copy.deepcopy(self._scenes[0])
            if q == 2:
                self._teleport_forward(scene)
            elif q == 3:
                self._teleport_backward(scene)
            elif q == 4:
                self._move_later(scene)


class ShapeConstancyQuartet(Quartet):
    """This quartet is about one object turning into another object of a
    different shape. The 'a' object may turn into the 'b' object or
    vice versa.
    """

    def __init__(self, template: Dict[str, Any], find_path: bool):
        super(ShapeConstancyQuartet, self).__init__(template, find_path)
        self._goal = intphys_goals.ShapeConstancyGoal()
        self._scenes[0] = copy.deepcopy(self._template)
        self._goal.update_body(self._scenes[0], self._find_path)
        # we need the b object for 3/4 of the scenes, so generate it now
        self._b = self._create_b()

    def _create_b(self) -> Dict[str, Any]:
        a = self._scenes[0]['objects'][0]
        while True:
            b_def = random.choice(objects.OBJECTS_INTPHYS)
            if b_def['type'] != a['type']:
                break
        b_def = util.finalize_object_definition(b_def)
        b = util.instantiate_object(b_def, a['original_location'], a['material_list'])
        return b

    def _turn_a_into_b(self, body: Dict[str, Any], q: int) -> None:
        if self._goal._object_creator == IntPhysGoal._get_objects_and_occluders_moving_across:
            # TODO: In MCS-124
            pass
        elif self._goal._object_creator == IntPhysGoal._get_objects_falling_down:
            # TODO: In MCS-131
            pass
        else:
            raise ValueError('unknown object creation function, cannot update scene')

    def _turn_b_into_a(self, body: Dict[str, Any], q: int) -> None:
        if self._goal._object_creator == IntPhysGoal._get_objects_and_occluders_moving_across:
            # TODO: In MCS-124
            pass
        elif self._goal._object_creator == IntPhysGoal._get_objects_falling_down:
            # TODO: In MCS-131
            pass
        else:
            raise ValueError('unknown object creation function, cannot update scene')

    def _b_replaces_a(self, body: Dict[str, Any], q: int) -> None:
        body['objects'][0] = self._b

    def get_scene(self, q: int) -> Dict[str, Any]:
        if q < 1 or q > 4:
            raise ValueError(f'q must be between 1 and 4 (exclusive), not {q}')
        scene = self._scenes[q - 1]
        if scene is None:
            scene = copy.deepcopy(self._scenes[0])
            if q == 2:
                # Object A moves behind an occluder, then object B emerges
                # from behind the occluder (implausible)
                self._turn_a_into_b(scene)
            elif q == 3:
                # Object B moves behind an occluder (replacing object A's
                # movement), then object A emerges from behind the
                # occluder (implausible)
                self._turn_b_into_a(scene)
            elif q == 4:
                # Object B moves normally (replacing object A's movement),
                # object A is never added to the scene (plausible)
                self._b_replaces_a(scene)
            self._scenes[q - 1] = scene
        return scene


QUARTET_TYPES = [ObjectPermanenceQuartet]


def get_quartet_class(name: str) -> Type[Quartet]:
    class_name = name + 'Quartet'
    klass = globals()[class_name]
    return klass


def get_quartet_types() -> List[str]:
    return [klass.__name__.replace('Quartet', '') for klass in QUARTET_TYPES]
