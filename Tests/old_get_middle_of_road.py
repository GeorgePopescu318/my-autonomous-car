import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
def get_middle(frame,displayVideo):
    middleBirdEye = []

    ## Choosing points for perspective transformation

    # cv2.circle(frame, tl, 5, (0,0,255), -1)
    # cv2.circle(frame, bl, 5, (0,0,255), -1)
    # cv2.circle(frame, tr, 5, (0,0,255), -1)
    # cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Aplying perspective transformation
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = 0
    l_s = 0
    l_v = 200
    u_h = 255
    u_s = 50
    u_v = 255

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    # plt.hist(histogram, bins=mask.shape[0], color='skyblue', edgecolor='black')
    # x = np.arange(histogram.shape[0])

    # plt.figure(figsize=(10, 4))
    # plt.plot(x, histogram, color='blue', linewidth=2)
    # plt.title("Histogram (Sum over Bottom Half of Mask)")
    # plt.xlabel("Column Index")
    # plt.ylabel("Sum")
    # plt.grid(True)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    startingY = 472
    y = startingY
    lx = []
    rx = []

    msk = mask.copy()
    windowHeight = 40
    windowWidt = 100

    if left_base != 0:
        if left_base - windowWidt > 0:
            img = mask[y-windowHeight:y, left_base-windowWidt:left_base+windowWidt]
        else:
            img = mask[y-windowHeight:y, 1:left_base+windowWidt]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            largest_contour = 0
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            left_base = 0

    if right_base != mask.shape[1]:
        if right_base + windowWidt < mask.shape[1]:
            img = mask[y-windowHeight:y, right_base-windowWidt:right_base+windowWidt]
        else:
            img = mask[y-windowHeight:y, right_base-windowWidt:mask.shape[1]]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            largest_contour = 0
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            right_base = mask.shape[1]

    while y > 0 and (left_base + windowWidt) < mask.shape[1] and (right_base - windowWidt) > 0:
        drawL = False
        drawR = False
        ## Left threshold
        if left_base != 0:
            if left_base - windowWidt > 0:
                img = mask[y-windowHeight:y, left_base-windowWidt:left_base+windowWidt]
            else:
                img = mask[y-windowHeight:y, 1:left_base+windowWidt]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
            else:
                largest_contour = 0
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    if left_base > windowWidt:
                        lx.append(left_base-windowWidt+cx)
                        left_base =left_base-windowWidt+cx
                    else:
                        lx.append(cx)
                        left_base = cx
                drawL = True
        ## Right threshold
        if right_base != mask.shape[1]:
            if right_base + windowWidt < mask.shape[1]:
                img = mask[y-windowHeight:y, right_base-windowWidt:right_base+windowWidt]
            else:
                img = mask[y-windowHeight:y, right_base-windowWidt:mask.shape[1]]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
            else:
                largest_contour = 0
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base-windowWidt + cx)
                right_base = right_base-windowWidt + cx
                drawR = True

        if left_base != 0 and drawL:
            cv2.rectangle(msk, (left_base-windowWidt,y), (left_base+windowWidt,y-windowHeight), (255,255,255), 2)
        if right_base != img.shape[1] and drawR:
            cv2.rectangle(msk, (right_base-windowWidt,y), (right_base+windowWidt,y-windowHeight), (255,255,255), 2)
        y -= windowHeight


    if len(lx) >= finalMiddlePoints and len(rx) >= finalMiddlePoints:
        # if np.abs(lx[0] - rx[0]) > 50 and list(set(lx) & set(rx)) == []:
        if np.min([np.abs(lx[i]-rx[i]) for i in range(finalMiddlePoints)]) > 50:
            for i in range(finalMiddlePoints):
                # cv2.circle(frame, ((int(original_ptsR[i][0][0]) + int(original_ptsL[i][0][0]))//2,int(original_ptsR[i][0][1])), 5, (0,0,255), -1)
                # middleBirdEye = np.array([[x, startingY - i * windowHeight] for i, x in enumerate(rx)], dtype=np.float32)
                middleBirdEye = np.append(middleBirdEye,[(rx[i] + lx[i])//2,startingY - i * windowHeight])
        elif lx[-1] > mask.shape[1]//2:
            for i in range(finalMiddlePoints):
                # cv2.circle(frame, (int(original_ptsL[i][0][0]) + 250,int(original_ptsL[i][0][1])), 5, (0,0,255), -1)
                middleBirdEye = np.append(middleBirdEye,[lx[i] + singleLaneShift,startingY - i * windowHeight])
        elif rx[-1] < mask.shape[1]//2:
            for i in range(finalMiddlePoints):
                # cv2.circle(frame, (int(original_ptsR[i][0][0]) - 250,int(original_ptsR[i][0][1])), 5, (0,0,255), -1)
                middleBirdEye = np.append(middleBirdEye,[rx[i] - singleLaneShift,startingY - i * windowHeight])
    elif len(lx) >= finalMiddlePoints:
        for i in range(finalMiddlePoints):
            # cv2.circle(frame, (int(original_ptsL[i][0][0]) + 250,int(original_ptsL[i][0][1])), 5, (0,0,255), -1)
            middleBirdEye = np.append(middleBirdEye,[lx[i] + singleLaneShift,startingY - i * windowHeight])
    elif len(rx) >= finalMiddlePoints:
        for i in range(finalMiddlePoints):
            # cv2.circle(frame, (int(original_ptsR[i][0][0]) - 250,int(original_ptsR[i][0][1])), 5, (0,0,255), -1)
            middleBirdEye = np.append(middleBirdEye,[rx[i] - singleLaneShift,startingY - i * windowHeight])

    middleBirdEye = middleBirdEye.reshape(-1, 1, 2)
    originalMiddle = cv2.perspectiveTransform(middleBirdEye, invMatrix)

    
    if displayVideo:
        for i in range(len(originalMiddle)-1):
            cv2.line(frame, (int(originalMiddle[i][0][0]),int(originalMiddle[i][0][1])),(int(originalMiddle[i+1][0][0]),int(originalMiddle[i+1][0][1])), (0,0,255), 2)
        cv2.imshow("Original", frame)
        # cv2.imshow("Bird's Eye View", transformed_frame)
        # cv2.imshow("Lane Detection - Image Thresholding", mask)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
    return originalMiddle

# tl = (133,120)
# bl = (0 ,415)
# tr = (460,120)
# br = (638,415)
# pts1 = np.float32([tl, bl, tr, br]) 
# pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 

# Matrix to warp the image for birdseye window
matrix = cv2.getPerspectiveTransform(pts1, pts2)
invMatrix = cv2.getPerspectiveTransform(pts2, pts1)

finalMiddlePoints = 6

singleLaneShift = 315
# vidcap = cv2.VideoCapture("NewCameraRodNewTrack.mp4")
# success, image = vidcap.read()
# while success:
#     success, frame = vidcap.read()

#     originalMiddle = get_middle(frame,False)

#     originalMiddle = originalMiddle.astype(int)

#     cv2.circle(frame, (originalMiddle[len(originalMiddle)//2][0][0],originalMiddle[len(originalMiddle)//2][0][1]), 5, (0,0,255), -1)
#     cv2.imshow("middlePoint",frame)
#     if cv2.waitKey(10) == 27:
#         # cv2.imwrite("bird_for_track.jpg",frame)
#         break
# # result.release() 
# cv2.destroyAllWindows()