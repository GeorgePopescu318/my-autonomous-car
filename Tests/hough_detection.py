import cv2
import numpy as np

def make_coordinates(image, line_parameters, y_min_fraction=0.4):
    """
    Calculate the coordinates of a line segment, extending it from the bottom to a specified height in the image.
    
    Args:
        image: Input image (used for height reference).
        line_parameters: Tuple containing slope and intercept of the line.
        y_min_fraction: Fraction of the image height to which the line should extend (default is 40%).
    
    Returns:
        Array of coordinates [x1, y1, x2, y2] for the line segment.
    """
    if line_parameters is None:
        return None
    
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * y_min_fraction)  # Extend to a fraction of the image height
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except ZeroDivisionError:
        return None  # Avoid division by zero in case of a vertical line
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines, y_min_fraction=0.4):
    """
    Averages the slope and intercept of detected lines and extends them to specified height ranges.
    
    Args:
        image: Input image (used for height reference).
        lines: Array of lines detected by HoughLinesP.
        y_min_fraction: Fraction of the image height to which the line should extend.
    
    Returns:
        Array of extrapolated line coordinates.
    """
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:  # Handle nested arrays from HoughLinesP
            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Fit a line (slope, intercept)
            slope, intercept = parameters
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    # Calculate the average slope and intercept for each side
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # Extrapolate the lines to desired height ranges
    left_line = make_coordinates(image, left_fit_average, y_min_fraction)
    right_line = make_coordinates(image, right_fit_average, y_min_fraction)

    return np.array([line for line in [left_line, right_line] if line is not None])


def pre_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(3, 3), 0)
    bilateralBlurred = cv2.bilateralFilter(gray, 11, 21, 7)
    return bilateralBlurred

def canny(pre_proc):
    
    canny = cv2.Canny(pre_proc, 50, 150,L2gradient=True)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(0, 175), (318, 175), (180, 25),(0,25)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def get_intersection_x(line, y):
    """
    For a given line (x1,y1,x2,y2) and a row y, compute the x coordinate
    where the line intersects the horizontal line at that y.
    Returns None if y is not within the y-range of the line.
    """
    x1, y1, x2, y2 = line
    # If the line is horizontal, just return the middle x
    if y1 == y2:
        if int(round(y1)) == y:
            return (x1 + x2) / 2.0
        else:
            return None
    # Check if the given y is within the line's vertical span.
    if y < min(y1, y2) or y > max(y1, y2):
        return None
    # Compute the intersection using linear interpolation.
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

def filter_lines_by_row(lines, width=640, height=480):
    """
    For each row (0 to height-1) of the image, retains:
      - the left–side line (intersection x < width/2) closest to the center, and
      - the right–side line (intersection x > width/2) closest to the center.
    
    Returns two lists (of length 'height'):
      best_left[y]  = best left–side line for row y (or None if no line)
      best_right[y] = best right–side line for row y (or None if no line)
    """
    center = width / 2.0
    best_left = [None] * height
    best_right = [None] * height
    # Distance (in x) from the center for the best candidate in that row.
    best_left_dist = [float('inf')] * height
    best_right_dist = [float('inf')] * height

    # Iterate over each line.
    for line in lines:
        x1, y1, x2, y2 = line
        # Determine the range of rows the line covers.
        y_start = int(round(min(y1, y2)))
        y_end   = int(round(max(y1, y2)))
        # Clip the y range to image bounds.
        y_start = max(y_start, 0)
        y_end   = min(y_end, height - 1)
        # For every row the line passes through:
        for y in range(y_start, y_end + 1):
            x_inter = get_intersection_x(line, y)
            if x_inter is None:
                continue
            # If the intersection is to the left of center
            if x_inter < center:
                dist = center - x_inter
                if dist < best_left_dist[y]:
                    best_left_dist[y] = dist
                    best_left[y] = line
            # If to the right of center
            elif x_inter > center:
                dist = x_inter - center
                if dist < best_right_dist[y]:
                    best_right_dist[y] = dist
                    best_right[y] = line
            # If exactly equal to the center, you can decide how to handle it.
    return best_left, best_right

def find_lanes(lines,width,height):
    minLeftLineValue = 1000
    minLeftLine = []
    minRightLineValue = 1000
    minRightLine = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            distanceFromCenter = np.abs((width / 2) - x1)
            if x1 < width / 2:
                if distanceFromCenter < minLeftLineValue:
                    minLeftLine = line[0]
                    minLeftLineValue = distanceFromCenter
            else:
                if distanceFromCenter < minRightLineValue:
                    minRightLine = line[0]
                    minRightLineValue = distanceFromCenter
    return minLeftLine, minRightLine

image = cv2.imread("camera_view_3.jpg")
lane_image = np.copy(image)
pre_proc_img = pre_process(lane_image)
canny_image = canny(pre_proc_img)
# cropped_image = region_of_interest(canny_image)
height,width = canny_image.shape
print(width,height)
cropped_image = canny_image
cv2.imshow('cropp',cropped_image)
cv2.waitKey(0)
raw_lines = cv2.HoughLinesP(cropped_image,rho = 1,theta = 1*np.pi/180,threshold = 30,minLineLength = 1,maxLineGap = 20)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)

# minLeftLine, minRightLine = find_lanes(lines,width,height)

if raw_lines is not None:
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1-width//2):
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# if raw_lines is not None:
#     lines = [line[0] for line in raw_lines]
#     print(lines)
# else:
#     lines = []

# best_left, best_right = filter_lines_by_row(lines, width=640, height=480)
    
#     # For visualization, we can draw the selected lines.
# output = image.copy()
# # Draw best left lines in blue, best right in red.
# for y in range(480):
#     if best_left[y] is not None:
#         x1, y1, x2, y2 = best_left[y]
#         cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     if best_right[y] is not None:
#         x1, y1, x2, y2 = best_right[y]
#         cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)if
# print(minLeftLine)
# print(minRightLine)
# cv2.line(image, (minLeftLine[0], minLeftLine[1]), (minLeftLine[2], minLeftLine[3]), (0, 255, 0), 5)
# cv2.line(image, (minRightLine[0], minRightLine[1]), (minRightLine[2], minRightLine[3]), (0, 255, 0), 5)
# cv2.imwrite("track_hough.jpg",image)
cv2.imshow('result',image)
cv2.waitKey(0)
# cv2.imshow('result',cropped_image)
# cv2.waitKey(0)

#---------------------------------

# cap = cv2.VideoCapture('1.mp4')
# while(cap.isOpened()):
#     _, frame = cap.read()
    
#     pre_proc_img = pre_process(frame)
#     canny_image = canny(pre_proc_img)
#     cropped_image = region_of_interest(canny_image)
#     width,height = cropped_image.shape
#     lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 20, np.array([]), maxLineGap=10)
#     minLeftLine, minRightLine = find_lanes(lines,width,height)

#     # cv2.line(frame, (minLeftLine[0], minLeftLine[1]), (minLeftLine[2], minLeftLine[3]), (0, 255, 0), 5)
#     # cv2.line(frame, (minRightLine[0], minRightLine[1]), (minRightLine[2], minRightLine[3]), (0, 255, 0), 5)
#     # cv2.imshow('result',frame)

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
#     # averaged_lines = average_slope_intercept(frame, lines)
#     # line_image = display_lines(frame, averaged_lines)
#     # combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow('result',frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()