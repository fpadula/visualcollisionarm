import numpy as np
import cv2
import dt_apriltags
import yaml
import argparse
from math import fabs

# Print line between two points with floating type
def draw_line(frame, point_a, point_b, color, width):
    point_ai = (int(point_a[0]), int(point_a[1]))
    point_bi = (int(point_b[0]), int(point_b[1]))
    cv2.line(frame, point_ai, point_bi, color, width)

# Print tag id
def print_tag_info(frame, tag, font, color, scale, thickness):
    point_ai = (int(tag.corners[0][0]), int(tag.corners[0][1]))
    text = "ID:" + str(tag.tag_id)
    cv2.putText(frame, text, (point_ai[0], point_ai[1] - 15),
                font, scale, color, thickness)
    (c_x, c_y) = (int(tag.center[0]), int(tag.center[1]))
    cv2.circle(frame, (c_x, c_y), 5,(0, 0, 255), -1)

def draw_rectangle(frame, points, color, width, diagonals=False):
    # print(type(points))
    # print(len(points))
    for i in range(len(points) -1):
        draw_line(frame, points[i], points[i+1], color, width)
    draw_line(frame, points[-1], points[0], color, width)
    if diagonals:
        draw_line(frame, points[0], points[2], color, width)
        draw_line(frame, points[1], points[3], color, width)

def warp_point(H, point):
    warped_point = np.matmul(H, np.r_[point, [1]])
    warped_point /= warped_point[2]
    return warped_point[:-1]

def draw_axis(img, origin, imgpts):
    # corner = tuple(corners[0].ravel())
    origin_int = (int(origin[0]), int(origin[1]))
    cv2.line(img, origin_int, tuple(imgpts[0].astype(int).ravel()), (0,0,255), 5)
    cv2.line(img, origin_int, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)
    cv2.line(img, origin_int, tuple(imgpts[2].astype(int).ravel()), (255,0,0), 5)


"""
Calculates a homography transformation from image pixels to workspace
pixels.
    ws_corners:     Workspace corners in camera pixel coordinates.

    Returns:
        H:      Homography transform
        size:   Workspace size (width, height)
"""
def calculate_H(ws_corners, to_screen = True, dest_pts = None):
    if to_screen:
        ws_view_height = np.linalg.norm(
            np.array(ws_corners[0])-np.array(ws_corners[1]))
        ws_view_width = np.linalg.norm(
            np.array(ws_corners[2])-np.array(ws_corners[1]))

        dest_pts = np.float32([
            [ws_view_width, 0],
            [ws_view_width, ws_view_height],
            [0, ws_view_height],
            [0, 0]
        ])
        size = (int(ws_view_width), int(ws_view_height))
    else:
        size = (int(dest_pts[1][0]), int(dest_pts[1][1]))
    H, _ = cv2.findHomography(np.array(ws_corners), dest_pts, cv2.RANSAC,5.0)
    return H, size

def callback(value):
    pass

def setup_trackbars(initial_values = [0, 0, 0, 0, 0, 0]):
    cv2.namedWindow("Trackbars", 0)
    cv2.createTrackbar("H_MIN", "Trackbars", int(initial_values[0]*2), 360,
                       callback)
    cv2.createTrackbar("L_MIN", "Trackbars", int(initial_values[1]/2.55), 100,
                       callback)
    cv2.createTrackbar("S_MIN", "Trackbars", int(initial_values[2]/2.55), 100,
                       callback)
    cv2.createTrackbar("H_MAX", "Trackbars", int(initial_values[3]*2), 360,
                       callback)
    cv2.createTrackbar("L_MAX", "Trackbars", int(initial_values[4]/2.55), 100,
                       callback)
    cv2.createTrackbar("S_MAX", "Trackbars", int(initial_values[5]/2.55), 100,
                       callback)

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            v = v/2.0 if j == "H" else v * 2.55
            values.append(v)

    return values

#Mouse Callback function
def pick_color(event,x,y,flags,param): 
    if event == cv2.EVENT_LBUTTONDOWN:        
        hls_value = param[y, x]        
        variables = ["H", "L", "S"]      
        print(f'H: {hls_value[0]*2}, L: {hls_value[1]/255}, S: {hls_value[2]/255}')  
        type = "MIN"
        for i, value_name in enumerate(variables):
            value = hls_value[i]
            # value = max(hsl_value[i]-10, 0)
            value = value*2.0 if value_name == "H" else value / 2.55
            value = int(value)
            cv2.setTrackbarPos(f'{value_name}_{type}', "Trackbars", value)
        type = "MAX"
        cv2.setTrackbarPos("H_MAX", "Trackbars", 360)
        cv2.setTrackbarPos("S_MAX", "Trackbars", 100)
        cv2.setTrackbarPos("L_MAX", "Trackbars", 100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None,
                        help="The configuration file.")
    args = parser.parse_args()
    with open(args.config_file, encoding='UTF-8') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    tag_detector = dt_apriltags.Detector(families='tag36h11')
    camera_matrix = np.array(params["camera_matrix"])
    camera_distortion = np.array(params["distort_coefs"])
    cap = cv2.VideoCapture(2)
    # Same dimensions as calibrated
    # size = (2304, 1536)
    size = params["camera_resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    correct_frames = False

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, camera_distortion, size, 1, size)

    tags_to_id = [0, 1, 2, 3]

    ws_corners = [
        params["ws_corners"]["top_right"],
        params["ws_corners"]["bottom_right"],
        params["ws_corners"]["bottom_left"],
        params["ws_corners"]["top_left"]
    ]
    # ws_corners = np.loadtxt('ws_corners.txt', delimiter=',')
    i = 1
    detecting_ws_corners = True
    print(f"{i}. Setting workspace corners:")
    print( ("\tSet the tags in the workspace corners and press Enter or press"
        "Esc to skip and load from file."))
    while detecting_ws_corners:
        _, frame = cap.read()
        if correct_frames:
            frame = cv2.undistort(frame, camera_matrix, camera_distortion, None,
                                  newcameramtx)
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_tags = tag_detector.detect(gray)
        # if ids is not None and ids[0] == id_to_find:
        for tag in detected_tags:
            if tag.tag_id not in tags_to_id:
                continue
            (cX, cY) = (tag.center[0], tag.center[1])
            ws_corners[tag.tag_id] = [cX, cY]
            draw_rectangle(frame, tag.corners, (0, 255, 0), 2)
            print_tag_info(frame, tag, cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255),
                           0.5, 2)

        if len(detected_tags)>=4:
            draw_rectangle(frame, ws_corners, (0, 255, 0), 2, diagonals=True)
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        # Save to file:
        if key == 13:
            detecting_ws_corners = False
            print("\tSaving workspace specs to file...")
            ws_corners = np.array(ws_corners)
            # np.savetxt('ws_corners.txt', ws_corners, delimiter=',')
            
            #quit()
            params["ws_corners"]["top_right"] = ws_corners[0].tolist()
            params["ws_corners"]["bottom_right"] = ws_corners[1].tolist()
            params["ws_corners"]["bottom_left"] = ws_corners[2].tolist()
            params["ws_corners"]["top_left"] = ws_corners[3].tolist()
            #with open("config.yaml", 'w', encoding='UTF-8') as outfile:
            with open(args.config_file, 'w', encoding='UTF-8') as outfile:
                yaml.dump(params, outfile, default_flow_style=False)
        # Load from file:
        elif key==27:
            detecting_ws_corners = False
            print("\tLoading workspace specs from file...")
            ws_corners = [
                params["ws_corners"]["top_right"],
                params["ws_corners"]["bottom_right"],
                params["ws_corners"]["bottom_left"],
                params["ws_corners"]["top_left"]
            ]
    print(ws_corners)
    w = fabs(ws_corners[0][0] - ws_corners[1][0])
    h = fabs(ws_corners[1][1] - ws_corners[3][1])
    dest_pts = np.float32([
        [w, 0],
        [w, h],
        [0, h],
        [0, 0]
    ])
    print(dest_pts)
    camera_to_workspace_H, ws_view_size = calculate_H(ws_corners,False,
                                                      dest_pts)
    i += 1
    for seg_class in params["segmentation_classes"]:
        print(f"{i}. Setting class \'{seg_class['name']}\' threshold values:")
        print("\tPress Enter to save values or ESC to skip.")
        for color in seg_class['input_colors']:
            print(f"\t\tProcessing color: {color['name']}")
            setting_class_thresholds = True
            setup_trackbars(color["min"] + color["max"])
            while setting_class_thresholds:
                _, frame = cap.read()
                if correct_frames:
                    frame = cv2.undistort(frame, camera_matrix, camera_distortion, None,
                                        newcameramtx)
                    x, y, w, h = roi
                    frame = frame[y:y+h, x:x+w]

                ws_view = cv2.warpPerspective(frame , camera_to_workspace_H,
                                            ws_view_size)
                draw_rectangle(frame, ws_corners, (0, 255, 0), 2, diagonals=True)

                hls_frame = cv2.cvtColor(ws_view, cv2.COLOR_BGR2HLS)

                frame_to_thresh = cv2.medianBlur(hls_frame, ksize=15)
                # frame_to_thresh = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HLS)
                v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(
                    "HLS")

                mask = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min),
                                            (v1_max, v2_max, v3_max))
                preview = cv2.bitwise_and(ws_view, ws_view, mask=mask)
                cv2.imshow("segmented", preview)
                cv2.imshow('frame', frame)
                cv2.imshow('ws', ws_view)                
                cv2.setMouseCallback('ws', pick_color, param = hls_frame)
                key = cv2.waitKey(1) & 0xFF
                # Save to file:
                if key == 13:
                    setting_class_thresholds = False
                    print("\tSaving workspace specs to file...")
                    color["min"] = [v1_min, v2_min, v3_min]
                    color["max"] = [v1_max, v2_max, v3_max]
                    with open(args.config_file, 'w', encoding='UTF-8') as outfile:
                        yaml.dump(params, outfile, default_flow_style=False)
                # Load from file:
                elif key==27:
                    setting_class_thresholds = False
                    print("\tLoading workspace specs from file...")
            cv2.destroyWindow("Trackbars")

    displaying_seg = True
    segmented_out = None
    segmented_shape = (frame_to_thresh.shape[0], frame_to_thresh.shape[1])
    running_avg = np.zeros(segmented_shape, np.float64)
    cv2.namedWindow("Trackbar", 0)
    cv2.createTrackbar("Running avg int:", "Trackbar",
                       int(params["running_average_alpha"]*100), 100, callback)
    for seg_class in params["segmentation_classes"]:
        if ("kernel_size" in seg_class) and ("morph_op" in seg_class):
            cv2.createTrackbar(f"{seg_class['name']} kernell size", "Trackbar",
                               seg_class["kernel_size"], 50, callback)
            cv2.createTrackbar(f"{seg_class['name']} operation", "Trackbar",
                               int(seg_class["morph_op"] == "close"), 1,
                               callback)
    i +=1
    print(f"{i}. Display final segmentation:")
    print("\tDisplaying final segmentation. Press ESC or Enter to exit.")
    while displaying_seg:
        _, frame = cap.read()
        if correct_frames:
            frame = cv2.undistort(frame, camera_matrix, camera_distortion, None,
                                  newcameramtx)
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]

        ws_view = cv2.warpPerspective(frame , camera_to_workspace_H,
                                      ws_view_size)

        draw_rectangle(frame, ws_corners, (0, 255, 0), 2, diagonals=True)
        blurred_frame = cv2.medianBlur(ws_view, ksize=15)
        frame_to_thresh = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HLS)

        segmented_out = np.zeros(segmented_shape, np.uint8)
        for seg_class in params["segmentation_classes"]:
            for i, color in enumerate(seg_class['input_colors']):
                if i == 0:
                    mask = cv2.inRange(frame_to_thresh, tuple(color['min']),
                                       tuple(color['max']))
                else:
                    mask += cv2.inRange(frame_to_thresh, tuple(color['min']),
                                       tuple(color['max']))
            if ("kernel_size" in seg_class) and ("morph_op" in seg_class):
                seg_class["kernel_size"] = cv2.getTrackbarPos(
                    f"{seg_class['name']} kernell size", "Trackbar")
                kernel = np.ones((seg_class["kernel_size"] ,
                                 seg_class["kernel_size"] ),np.uint8)
                op_num = cv2.getTrackbarPos(
                    f"{seg_class['name']} operation", "Trackbar")
                if op_num == 1:
                    seg_class["morph_op"] = "close"
                    op = cv2.MORPH_CLOSE
                else:
                    seg_class["morph_op"] = "open"
                    op = cv2.MORPH_OPEN
                if seg_class["kernel_size"] > 0:
                    mask = cv2.morphologyEx(mask, op, kernel)
            segmented_out[mask>=100] = seg_class['output_color']

        v = cv2.getTrackbarPos("Running avg int:", "Trackbar")
        params["running_average_alpha"] = v/100
        cv2.accumulateWeighted(segmented_out, running_avg,
                               params["running_average_alpha"])
        running_avg_preview = cv2.convertScaleAbs(running_avg)

        downscaled = cv2.resize(running_avg_preview, (64,64), cv2.INTER_NEAREST)
        downscaled = cv2.resize(downscaled, (512,512), cv2.INTER_NEAREST)
        cv2.imshow("to agent", downscaled)
        cv2.imshow("running_avg", running_avg_preview)
        cv2.imshow("segmented", segmented_out)
        cv2.imshow('frame', frame)
        cv2.imshow('ws', ws_view)

        key = cv2.waitKey(1) & 0xFF
        # Wait for enter key:
        # Save to file:
        if key == 13:
            displaying_seg = False
            with open(args.config_file, 'w', encoding='UTF-8') as outfile:
                yaml.dump(params, outfile, default_flow_style=False)
        # Load from file:
        elif key==27:
            displaying_seg = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()