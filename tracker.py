import numpy as np

class Tracker:
    def __init__(self):
        self.next_id = 0

    def update(self, detections):
        """
        Assign a unique ID to each detection.
        detections: numpy array of [x1, y1, x2, y2, score] or similar
        Returns: numpy array of [x1, y1, x2, y2, id] for each detection
        """
        tracked = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]  # ignore score
            tracked.append([x1, y1, x2, y2, self.next_id])
            self.next_id += 1
        return np.array(tracked)
