from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, ClassVar, Set, Dict, List, Optional, Tuple
from pyCFTrackers.cftracker.base import BaseCF
from pyCFTrackers.lib.bbox_helper import cxy_wh_2_rect
from pyCFTrackers.cftracker.strcf import STRCF
from pyCFTrackers.cftracker.config import strdcf_hc_config
from helper import is_close

TRACKER_PERSISTENCE = 90 # number of frames to persist tracker after YOLO bound box is missing
TRACKER_ACCEPTANCE = 5 # number of consecutive YOLO detections required to initialize tracker for a target


XYWH = Tuple[int, int, int, int]

@dataclass
class YoloBox:
    id: int
    xywh: XYWH
    frameidx: int
    detect_count: int

@dataclass
class Tracker:
    instance: BaseCF
    xywh: XYWH

    def update(self, frame):
        self.xywh = self.instance.update(frame)

@dataclass
class DepthBox:
    depth: int
    xywh: XYWH

class Target:
    next_id: ClassVar[int] = 0
    curr_frame: ClassVar = None
    curr_frameidx: ClassVar[int] = 0
    yolo_mapping: ClassVar[Dict[int, Target]] = {}
    all_targets: ClassVar[Set[Target]] = set()
    tracker_generator: ClassVar[Callable[[], BaseCF]] = lambda: STRCF(config=strdcf_hc_config.STRDCFHCConfig()) 

    def __init__(self, 
                 yolo_box: Optional[YoloBox] = None, 
                 depth_box: Optional[DepthBox] = None):
        self.id = Target.next_id
        Target.next_id += 1

        self._yolo_box = yolo_box
        self._depth_box = depth_box
        self._tracker = None

        if depth_box is not None:
            self.init_tracker(depth_box.xywh)

        if yolo_box is not None:
            Target.yolo_mapping[yolo_box.id] = self

        Target.all_targets.add(self)

    def __del__(self):
        if self._yolo_box and self._yolo_box.id in Target.yolo_mapping:
            del Target.yolo_mapping[self._yolo_box.id]
        Target.all_targets.remove(self)

    def init_tracker(self, xywh):
        self._tracker = Tracker(Target.tracker_generator(), xywh)
        self._tracker.instance.init(Target.curr_frame, cxy_wh_2_rect(xywh[:2], xywh[2:]))

    @classmethod
    def next_frame(cls, frame):
        cls.curr_frameidx += 1
        cls.curr_frame = frame

        for target in cls.all_targets:
            if target._tracker is not None:
                target._tracker.update(cls.curr_frame)

    @classmethod
    def batch_update_yolo(cls, yolo_data: Dict[int, XYWH]):
        for yolo_id in yolo_data.keys() & cls.yolo_mapping.keys():
            # update existing yolo targets
            cls.yolo_mapping[yolo_id].update_yolo(yolo_data[yolo_id])

        for yolo_id in yolo_data.keys() - cls.yolo_mapping.keys():
            # new yolo targets, create them
            Target(yolo_box=YoloBox(yolo_id, yolo_data[yolo_id], cls.curr_frameidx, 0))

        for yolo_id in cls.yolo_mapping.keys() - yolo_data.keys():
            # existing yolo targets that were not detected this round
            target = cls.yolo_mapping[yolo_id]
            assert target._yolo_box is not None

            if target._tracker is None and target._yolo_box.detect_count < TRACKER_ACCEPTANCE:
                del cls.yolo_mapping[yolo_id]
                cls.all_targets.remove(target)


    @classmethod
    def get_by_yolo_id(cls, yolo_id: int):
        return cls.yolo_mapping.get(yolo_id)
    
    @classmethod
    def set_tracker_generator(cls, generator: Callable[[], BaseCF]):
        cls.tracker_generator = generator

    @classmethod
    def guess_by_depth_box(cls, frame, depth_box: DepthBox):
        for target in cls.all_targets:
            _, coords = target.aggregate_coords()
            if coords is None: continue
            if is_close(coords, depth_box.xywh):
                return target
        new_target = Target(frame, depth_box = depth_box)
        return new_target

    def update_yolo(self, xywh: XYWH):
        assert self._yolo_box is not None
        yolo = self._yolo_box
        yolo.xywh = xywh
        yolo.frameidx = Target.curr_frameidx
        yolo.detect_count += 1

        if yolo.detect_count >= TRACKER_ACCEPTANCE:
            self.init_tracker(xywh)

    def get_yolo_xywh(self) -> Optional[Tuple]:
        if self._yolo_box is not None and self._yolo_box.frameidx == Target.curr_frameidx:
            return self._yolo_box.xywh
        return None

    def get_tracker_xywh(self) -> Optional[Tuple]:
        if self._tracker is not None:
            return self._tracker.xywh
        return None

    def get_depth_detect_xywh(self) -> Optional[Tuple]:
        if self._depth_box is not None:
            return self._depth_box.xywh
        return None

    def aggregate_coords(self) -> Tuple[List[str], Optional[XYWH]]:
        yolo_coords = self.get_yolo_xywh()
        tracker_coords = self.get_tracker_xywh()
        depth_detect_coords = self.get_depth_detect_xywh()

        sources = []
        if yolo_coords: sources.append("YOLO")
        if tracker_coords: sources.append("Tracker")
        if depth_detect_coords: sources.append("Depth detection")

        if sources == []:
            return (sources, None)
        # placeholder aggregation method, can replace once formula finalised
        if yolo_coords:
            return (sources, yolo_coords)
        elif depth_detect_coords:
            return (sources, depth_detect_coords)
        else:
            return (sources, tracker_coords)




