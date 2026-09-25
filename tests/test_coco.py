import cv2
import numpy as np
from pycocotools import mask as mask_utils

from cad2coco.coco import mask_bbox, mask_to_polygons, split_indices


def iou_of_polygon(mask, polygons):
    h, w = mask.shape
    rles = mask_utils.frPyObjects(polygons, h, w)
    decoded = mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    return (decoded & mask).sum() / (decoded | mask).sum()


def test_rectangle_polygon_and_bbox():
    m = np.zeros((100, 120), bool)
    m[20:60, 30:90] = True
    polys = mask_to_polygons(m)
    assert len(polys) == 1
    assert len(polys[0]) == 8  # 4 corners after simplification
    assert mask_bbox(m) == [30.0, 20.0, 60.0, 40.0]
    assert iou_of_polygon(m, polys) > 0.93


def test_disc_is_accurate():
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 60, 1, -1)
    polys = mask_to_polygons(m.astype(bool), tolerance=1.0)
    assert iou_of_polygon(m.astype(bool), polys) > 0.95


def test_occlusion_split_gives_two_parts():
    m = np.zeros((50, 100), bool)
    m[10:40, 5:40] = True
    m[10:40, 60:95] = True
    assert len(mask_to_polygons(m)) == 2


def test_tiny_parts_are_dropped():
    m = np.zeros((50, 50), bool)
    m[5:30, 5:30] = True
    m[45:47, 45:47] = True  # 4 px speck
    assert len(mask_to_polygons(m, min_part_area=12)) == 1


def test_ring_keeps_outer_contour_only():
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (50, 50), 40, 1, -1)
    cv2.circle(m, (50, 50), 20, 0, -1)
    assert len(mask_to_polygons(m.astype(bool))) == 1


def test_split_indices_partition():
    s = split_indices(50, 0.1, 0.2, seed=1)
    all_idx = s["train"] + s["valid"] + s["test"]
    assert sorted(all_idx) == list(range(50))
    assert (len(s["valid"]), len(s["test"])) == (5, 10)


def test_small_datasets_keep_every_split():
    s = split_indices(3, 0.1, 0.1, seed=0)
    assert all(len(v) == 1 for v in s.values())
    assert split_indices(5, 0.0, 0.0, seed=0)["train"] == [0, 1, 2, 3, 4]
