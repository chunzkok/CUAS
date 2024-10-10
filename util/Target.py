from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, ClassVar, Set, Dict, List, Optional, Tuple
from pyCFTrackers.cftracker.base import BaseCF
from pyCFTrackers.lib.bbox_helper import cxy_wh_2_rect, corner2center
from pyCFTrackers.cftracker.strcf import STRCF
from pyCFTrackers.cftracker.config import strdcf_hc_config

TARGET_PERSISTENCE = 24 # number of frames to persist tracker after no detections are made
TRACKER_ACCEPTANCE = 16 # number of consecutive detections required to start tracking the target
BBOX_PROXIMITY_THRESHOLD = 20 # maximum euclidean distance (in pixels) from center of existing bound boxes to classify as the same target
DEPTH_DETECT_ACCEPTANCE = 5 # number of consecutive depth detections required to recognize as an actual target
MAX_DEPTH_DIFF_RATIO = 0.1 # used when comparing against existing depth detection targets. do not merge if their difference is larger than this ratio.
YOLO_WEIGHT = 3
DEPTH_WEIGHT = 1

XYWH = Tuple[int, int, int, int]

@dataclass
class YoloBox:
    id: int
    xywh: XYWH
    frameidx: int

@dataclass
class Tracker:
    instance: BaseCF
    xywh: XYWH

    def update(self, frame):
        x, y, w, h = self.instance.update(frame)
        # this (x, y) corresponds to a corner coordinate instead of the center
        # so we need to translate it 
        self.xywh = (x + w/2, y + h/2, w, h)

@dataclass
class DepthBox:
    depth: int
    xywh: XYWH
    detect_count: int
    frameidx: int

def is_in_proximity(x1y1: Tuple[int, int], x2y2: Tuple[int, int], threshold: int) -> bool:
    dx = x1y1[0] - x2y2[0]
    dy = x1y1[1] - x2y2[1]
    return  dx*dx + dy*dy < threshold * threshold

def is_within_bbox(x1y1: Tuple[int, int], xywh: XYWH) -> bool:
    x, y, w, h = xywh
    return (x - w//2 <= x1y1[0] <= x + w//2
        and y - h//2 <= x1y1[1] <= y + h//2)

def merge_boxes(xywh1: XYWH, weight1: float, xywh2: XYWH, weight2: float) -> XYWH:
    ratio = weight1 / (weight1 + weight2)
    return tuple(map(lambda vals: int(ratio*vals[0] + (1-ratio)*vals[1]), zip(xywh1, xywh2)))


class Target:
    next_id: ClassVar[int] = 0
    curr_frame: ClassVar = None
    curr_frameidx: ClassVar[int] = 0
    yolo_mapping: ClassVar[Dict[int, Target]] = {}
    all_targets: ClassVar[Set[Target]] = set()
    updated_targets: ClassVar[Set[Target]] = set()
    probation_depth_boxes: ClassVar[List[DepthBox]] = []
    updated_probation_depth_boxes: ClassVar[List[DepthBox]] = []
    tracker_generator: ClassVar[Callable[[], BaseCF]] = lambda: STRCF(config=strdcf_hc_config.STRDCFHCConfig()) 

    def __init__(self, 
                 yolo_box: Optional[YoloBox] = None, 
                 depth_box: Optional[DepthBox] = None,
                 detect_count: int = 1):
        self.id = Target.next_id
        Target.next_id += 1
        print(f"Creating target {self.id} with yolo_box [{yolo_box}], depth_box[{depth_box}], detect_count[{detect_count}]")

        self._yolo_box = yolo_box
        self._depth_box = depth_box
        self.frameidx = Target.curr_frameidx
        self.detect_count = detect_count
        self._tracker = None

        if yolo_box is not None:
            Target.yolo_mapping[yolo_box.id] = self

        Target.all_targets.add(self)
        Target.updated_targets.add(self)

    def init_tracker(self, xywh):
        self._tracker = Tracker(Target.tracker_generator(), xywh)
        self._tracker.instance.init(Target.curr_frame, cxy_wh_2_rect(xywh[:2], xywh[2:]))

    @classmethod
    def next_frame(cls, frame):
        cls.curr_frameidx += 1
        cls.curr_frame = frame
        print(f"frame: {cls.curr_frameidx}")

        def remove_target(target: Target) -> None:
            cls.all_targets.remove(target)
            if target._yolo_box is not None and target._yolo_box.id in cls.yolo_mapping:
                del cls.yolo_mapping[target._yolo_box.id]

        # check to purge targets that are outdated
        for target in cls.all_targets - cls.updated_targets:
            if cls.curr_frameidx - target.frameidx >= TARGET_PERSISTENCE:
                remove_target(target)

        # check to upgrade probation boxes if applicable
        cls.probation_depth_boxes = []
        for box in cls.updated_probation_depth_boxes:
            if box.detect_count >= DEPTH_DETECT_ACCEPTANCE:
                Target(depth_box = box, detect_count=box.detect_count)
            else:
                cls.probation_depth_boxes.append(box)

        # update or initialize (if applicable) tracker.
        # also prune away targets at the border
        for target in cls.all_targets.copy():
            _, xywh = target.aggregate_coords()
            if xywh is not None and cls.is_coord_at_border(xywh[0], xywh[1]):
                remove_target(target)
                continue

            if target._tracker is not None:
                target._tracker.update(cls.curr_frame)
            else:
                if target.detect_count >= TRACKER_ACCEPTANCE:
                    if xywh is None:
                        assert target._depth_box is not None, "Obtained no coordinates when aggregate_cords called with show_latest=False"
                        xywh = target._depth_box.xywh
                    print(f"Init tracker: {xywh}")
                    target.init_tracker(xywh)
                elif target not in cls.updated_targets:
                    # target has not passed the tracker acceptance stage!!! begone >:-( 
                    remove_target(target)
        
        cls.updated_targets.clear()
        cls.updated_probation_depth_boxes = []

    @classmethod
    def batch_update_yolo(cls, yolo_data: Dict[int, XYWH]) -> None:
        for yolo_id in yolo_data.keys() & cls.yolo_mapping.keys():
            # update existing yolo targets
            cls.yolo_mapping[yolo_id].update_yolo(yolo_data[yolo_id])

        for yolo_id in yolo_data.keys() - cls.yolo_mapping.keys():
            # new yolo targets, first check if there are existing boxes that are close enough
            closest_target = cls.get_closest_target(yolo_data[yolo_id])
            if closest_target is not None:
                closest_target.update_yolo(yolo_data[yolo_id], yolo_id)
            else:
                # no boxes to merge with, create new one
                Target(yolo_box=YoloBox(yolo_id, yolo_data[yolo_id], cls.curr_frameidx))

    @classmethod
    def batch_update_depth(cls, depth_data: List[List[int]]) -> None:
        for data in depth_data:
            assert len(data) == 5, f"Depth detection data has {len(data)} fields instead of 5!"
            x, y, w, h, d = data
            x += w//2
            y += h//2
            if cls.is_coord_at_border(x, y):
                continue
            closest_target = cls.get_closest_target((x, y), d)
            if closest_target is not None:
                closest_target.update_depth(*data)
            else:
                # no existing target, put on probation
                probation_box = cls.process_depth_detection(*data)
                cls.updated_probation_depth_boxes.append(probation_box)


    @classmethod
    def set_tracker_generator(cls, generator: Callable[[], BaseCF]):
        cls.tracker_generator = generator

    @classmethod
    def get_closest_target(cls, coords: Tuple[int, ...], depth: float = -1) -> Optional[Target]:
        min_sq_dist, result = -1, None
        sq_threshold = BBOX_PROXIMITY_THRESHOLD * BBOX_PROXIMITY_THRESHOLD
        for target in cls.all_targets:
            if depth > 0 and target._depth_box is not None:
                depth_diff = abs(target._depth_box.depth - depth)
                if depth_diff > MAX_DEPTH_DIFF_RATIO * target._depth_box.depth:
                    continue

            _, target_coords = target.aggregate_coords()
            if target_coords is None: 
                continue
            
            dx = target_coords[0] - coords[0]
            dy = target_coords[1] - coords[1]

            sq_dist = dx*dx + dy*dy
            if not is_within_bbox(coords, target_coords) and sq_dist > sq_threshold:
                continue

            if result is None or sq_dist < min_sq_dist:
                min_sq_dist, result = sq_dist, target
        return result

    @classmethod
    def process_depth_detection(cls, x0: int, y0: int, w0: int, h0: int, depth: int) -> DepthBox:
        # shift x0, y0 to be center coordinate
        x0 += w0//2 
        y0 += h0//2
        closest_box, min_sq_dist = None, -1
        sq_threshold = BBOX_PROXIMITY_THRESHOLD * BBOX_PROXIMITY_THRESHOLD

        for box in cls.probation_depth_boxes:
            x1, y1, _, _ = box.xywh
            dx, dy = x0 - x1,  y0 - y1
            sq_dist = dx * dx + dy * dy

            if sq_dist > sq_threshold: 
                continue

            depth_diff = abs(depth - box.depth)
            if depth_diff > MAX_DEPTH_DIFF_RATIO * box.depth:
                continue

            if closest_box is None or sq_dist < min_sq_dist:
                closest_box, min_sq_dist = box, sq_dist
        
        if closest_box is not None:
            closest_box.detect_count += 1
            closest_box.xywh = (x0, y0, w0, h0)
            closest_box.depth = depth
            closest_box.frameidx = Target.curr_frameidx
        else:
            closest_box = DepthBox(depth, (x0, y0, w0, h0), 1, cls.curr_frameidx)

        return closest_box

    @classmethod
    def is_coord_at_border(cls, x:int, y: int, margin: float = 0.10) -> bool:
        H, W, _ = cls.curr_frame.shape
        return (x < W * margin
                or x > W * (1 - margin)
                or y < H * margin
                or y > H * (1 - margin))

    def update_yolo(self, xywh: XYWH, id: int = -1) -> None:
        if self._yolo_box is None:
            assert id > -1, "Attempted to create YoloBox for existing target without valid YOLO id!"
            self._yolo_box = YoloBox(id, xywh, Target.curr_frameidx)
        else:
            yolo = self._yolo_box
            yolo.xywh = xywh
            if id > -1:
                yolo.id = id
            yolo.frameidx = Target.curr_frameidx

        self._post_update(xywh)

    def update_depth(self, x: int, y: int, w: int, h: int, depth: int) -> None:
        x += w//2
        y += h//2
        if self._depth_box is None:
            self._depth_box = DepthBox(depth, (x, y, w, h), 1, Target.curr_frameidx)
        else:
            self._depth_box.depth = depth
            self._depth_box.xywh = (x, y, w, h)
            self._depth_box.detect_count += 1
            self._depth_box.frameidx = Target.curr_frameidx
        self._post_update((x,y,w,h))
        

    def get_yolo_xywh(self) -> Optional[Tuple]:
        if self._yolo_box is not None:
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

    def aggregate_coords(self, show_latest: bool = False) -> Tuple[List[str], Optional[XYWH]]:
        yolo_coords = self.get_yolo_xywh()
        tracker_coords = self.get_tracker_xywh()
        depth_detect_coords = self.get_depth_detect_xywh()

        if show_latest:
            yolo_is_updated = self._yolo_box is not None and self._yolo_box.frameidx == Target.curr_frameidx
            depth_is_updated = self._depth_box is not None and self._depth_box.frameidx == Target.curr_frameidx
        else:
            yolo_is_updated = self._yolo_box is not None and Target.curr_frameidx - self._yolo_box.frameidx < 2
            depth_is_updated = self._depth_box is not None and Target.curr_frameidx - self._depth_box.frameidx < 2
        

        sources = []

        if yolo_is_updated:
            sources.append("YOLO") 
        if depth_is_updated: 
            sources.append("Depth detection")
        if tracker_coords: 
            sources.append("Tracker")

        if yolo_is_updated and depth_is_updated:
            return (sources, merge_boxes(yolo_coords, YOLO_WEIGHT, depth_detect_coords, max(DEPTH_WEIGHT, DEPTH_WEIGHT/self._depth_box.depth)))
        if yolo_is_updated:
            return (sources, yolo_coords)
        if depth_is_updated:
            return (sources, depth_detect_coords)
        if tracker_coords:
            return (sources, tracker_coords)
        return (sources, None)

    def _post_update(self, xywh: XYWH) -> None:
        Target.updated_targets.add(self)

        if self.frameidx < Target.curr_frameidx:
            self.frameidx = Target.curr_frameidx
            self.detect_count += 1

        if self._tracker is not None:
            tracker_xywh = self._tracker.xywh
            if not is_in_proximity((tracker_xywh[0], tracker_xywh[1]), (xywh[0], xywh[1]), BBOX_PROXIMITY_THRESHOLD):
                self.init_tracker(xywh)

            
