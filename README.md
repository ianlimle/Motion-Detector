# Motion-Detector
OpenCV is used to calculate frame deltas against the most recent saved frame and the current frame. The frame deltas are then passed through a threshold filter, and bounding boxes are drawn around contours of the thresholded frame. A sufficiently large bounding box derived from the contours of a thresholded frame delta image is considered movement.
