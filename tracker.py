"""
tracker.py
──────────
Centroid-based cross-frame bounding box tracker.

Assigns a stable integer ID to each detected plate across frames.
Replaces the pixel-coordinate dict in the old livecam.py, which broke
the OCR throttle every time a box shifted by even 1 pixel.

Usage
─────
    tracker = CentroidTracker(max_distance=60, max_missing=10)

    # Each frame, pass the list of (x1, y1, x2, y2) boxes from YOLO.
    # Returns dict: { track_id: (cx, cy) }
    active = tracker.update(boxes)
"""

import math
from collections import OrderedDict


class CentroidTracker:
    """
    Nearest-centroid matching tracker.

    Parameters
    ──────────
    max_distance : int
        Maximum pixel distance between a centroid in frame N and a
        candidate in frame N+1 to be considered the same object.
        Set to ~10–15% of your typical plate width in pixels.

    max_missing : int
        How many consecutive frames a track can go unseen before it is
        removed. Keeps the tracker from dropping IDs during brief
        occlusions (e.g. another vehicle crossing in front).
    """

    def __init__(self, max_distance: int = 60, max_missing: int = 10):
        self.next_id      = 0
        self.max_distance = max_distance
        self.max_missing  = max_missing

        # Ordered so iteration order == insertion order (stable for display)
        self.centroids : OrderedDict[int, tuple[int, int]] = OrderedDict()
        self.missing   : dict[int, int]                    = {}   # id → missed frame count
        self.boxes     : dict[int, tuple[int,int,int,int]] = {}   # id → last known box

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        boxes: list[tuple[int, int, int, int]],
    ) -> dict[int, tuple[int, int, int, int]]:
        """
        Update tracker with new YOLO detections.

        Parameters
        ──────────
        boxes : list of (x1, y1, x2, y2)
            Bounding boxes from YOLO for the current frame.

        Returns
        ───────
        dict { track_id: (x1, y1, x2, y2) }
            Active tracks with their current bounding boxes.
            Only tracks seen in recent frames are included.
        """
        input_centroids = [
            (int((x1 + x2) / 2), int((y1 + y2) / 2))
            for x1, y1, x2, y2 in boxes
        ]

        # ── No existing tracks → register all as new ──────────────────────
        if not self.centroids:
            for i, centroid in enumerate(input_centroids):
                self._register(centroid, boxes[i])
            return dict(zip(self.boxes.keys(), self.boxes.values()))

        # ── No detections this frame → age all tracks ─────────────────────
        if not input_centroids:
            self._age_all()
            return {tid: self.boxes[tid] for tid in self.centroids}

        # ── Match incoming centroids to existing tracks ────────────────────
        existing_ids       = list(self.centroids.keys())
        existing_centroids = list(self.centroids.values())

        # Build distance matrix: rows = existing, cols = incoming
        distance_matrix = [
            [_dist(ec, ic) for ic in input_centroids]
            for ec in existing_centroids
        ]

        # Greedy nearest-match (sufficient for 1–3 simultaneous plates)
        matched_existing = set()
        matched_incoming = set()

        # Sort all (row, col, dist) by distance ascending
        pairs = sorted(
            (
                (r, c, distance_matrix[r][c])
                for r in range(len(existing_ids))
                for c in range(len(input_centroids))
            ),
            key=lambda x: x[2],
        )

        for r, c, dist in pairs:
            if r in matched_existing or c in matched_incoming:
                continue
            if dist > self.max_distance:
                break   # sorted, so all remaining are farther
            tid = existing_ids[r]
            self.centroids[tid] = input_centroids[c]
            self.boxes[tid]     = boxes[c]
            self.missing[tid]   = 0
            matched_existing.add(r)
            matched_incoming.add(c)

        # Age unmatched existing tracks
        for r, tid in enumerate(existing_ids):
            if r not in matched_existing:
                self.missing[tid] = self.missing.get(tid, 0) + 1
                if self.missing[tid] > self.max_missing:
                    self._deregister(tid)

        # Register unmatched incoming detections as new tracks
        for c in range(len(input_centroids)):
            if c not in matched_incoming:
                self._register(input_centroids[c], boxes[c])

        return {tid: self.boxes[tid] for tid in self.centroids}

    def get_box(self, track_id: int) -> tuple[int,int,int,int] | None:
        return self.boxes.get(track_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _register(self, centroid: tuple[int, int], box: tuple[int,int,int,int]) -> None:
        self.centroids[self.next_id] = centroid
        self.boxes[self.next_id]     = box
        self.missing[self.next_id]   = 0
        self.next_id += 1

    def _deregister(self, track_id: int) -> None:
        self.centroids.pop(track_id, None)
        self.boxes.pop(track_id, None)
        self.missing.pop(track_id, None)

    def _age_all(self) -> None:
        for tid in list(self.centroids.keys()):
            self.missing[tid] = self.missing.get(tid, 0) + 1
            if self.missing[tid] > self.max_missing:
                self._deregister(tid)


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])