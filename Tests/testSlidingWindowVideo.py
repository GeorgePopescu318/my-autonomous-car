import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
vidcap = cv2.VideoCapture("1.mp4")
success, frame = vidcap.read()

# result = cv2.VideoWriter('test11.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'), 
#                          10, (640,480)) 

tl = (133,120)
bl = (0 ,365)
tr = (460,120)
br = (638,365)
pts1 = np.float32([tl, bl, tr, br]) 
pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 

# Matrix to warp the image for birdseye window
matrix = cv2.getPerspectiveTransform(pts1, pts2)
inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)

finalMiddlePoints = 1

singleLaneShift = 315

lastMiddle = 0

while success:
    # time.sleep(1/15)
    success, frame = vidcap.read()
    frame = cv2.resize(frame,(640,480))
    middleBirdEye = []

    ## Choosing points for perspective transformation

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

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
    # plt.show()
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
#     if left_base != 0:
#         if left_base - windowWidt > 0:
#             img = mask[y-windowHeight:y, left_base-windowWidt:left_base+windowWidt]
#         else:
#             img = mask[y-windowHeight:y, 1:left_base+windowWidt]
#         contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#         else:
#             largest_contour = 0
#         M = cv2.moments(largest_contour)
#         if M["m00"] == 0:
#             left_base = 0

# ## Right threshold
#     if right_base != mask.shape[1]:
#         if right_base + windowWidt < mask.shape[1]:
#             img = mask[y-windowHeight:y, right_base-windowWidt:right_base+windowWidt]
#         else:
#             img = mask[y-windowHeight:y, right_base-windowWidt:mask.shape[1]]
#         contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#         else:
#             largest_contour = 0
#         M = cv2.moments(largest_contour)
#         if M["m00"] == 0:
#             right_base = mask.shape[1]
    # lx.append(left_base)
    # rx.append(right_base)
    while y > 0 and (left_base + windowWidt) < mask.shape[1] and (right_base - windowWidt) > 0:
        okl = 0
        okr = 0
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
                if left_base > windowWidt:
                    lx.append(left_base-windowWidt+cx)
                    left_base =left_base-windowWidt+cx
                else:
                    lx.append(cx)
                    left_base = cx
                # print(left_base)
                okl = 1

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
                okr = 1

        if left_base != 0 and okl == 1:
            cv2.rectangle(msk, (left_base-windowWidt,y), (left_base+windowWidt,y-windowHeight), (255,255,255), 2)
        if right_base != img.shape[1] and okr == 1:
            cv2.rectangle(msk, (right_base-windowWidt,y), (right_base+windowWidt,y-windowHeight), (255,255,255), 2)
        y -= windowHeight
        
    # for point in original_ptsL:
    #     cv2.circle(image, (int(point[0][0]),int(point[0][1])), 5, (0,0,255), -1)
        if len(lx) >= finalMiddlePoints and len(rx) >= finalMiddlePoints:
            # if np.abs(lx[0] - rx[0]) > 50 and list(set(lx) & set(rx)) == []:
            if np.min([np.abs(lx[i]-rx[i]) for i in range(finalMiddlePoints)]) > 200:
                print(np.min([np.abs(lx[i]-rx[i]) for i in range(finalMiddlePoints)]),"__")
                for i in range(finalMiddlePoints):
                    # cv2.circle(frame, ((int(original_ptsR[i][0][0]) + int(original_ptsL[i][0][0]))//2,int(original_ptsR[i][0][1])), 5, (0,0,255), -1)
                    # middleBirdEye = np.array([[x, startingY - i * windowHeight] for i, x in enumerate(rx)], dtype=np.float32)
                    middleBirdEye = np.append(middleBirdEye,[(rx[i] + lx[i])//2,startingY - i * windowHeight])
                    # print("mid")
            elif lx[-1] > mask.shape[1]//2:
                for i in range(finalMiddlePoints):
                    # cv2.circle(frame, (int(original_ptsL[i][0][0]) + 250,int(original_ptsL[i][0][1])), 5, (0,0,255), -1)
                    middleBirdEye = np.append(middleBirdEye,[lx[i] + singleLaneShift,startingY - i * windowHeight])
                    # print("left")
            elif rx[-1] < mask.shape[1]//2:
                for i in range(finalMiddlePoints):
                    # cv2.circle(frame, (int(original_ptsR[i][0][0]) - 250,int(original_ptsR[i][0][1])), 5, (0,0,255), -1)
                    middleBirdEye = np.append(middleBirdEye,[rx[i] - singleLaneShift,startingY - i * windowHeight])
                    # print("right")
        elif len(lx) >= finalMiddlePoints:
            for i in range(finalMiddlePoints):
                # cv2.circle(frame, (int(original_ptsL[i][0][0]) + 250,int(original_ptsL[i][0][1])), 5, (0,0,255), -1)
                middleBirdEye = np.append(middleBirdEye,[lx[i] + singleLaneShift,startingY - i * windowHeight])
                # print("left")
        elif len(rx) >= finalMiddlePoints:
            for i in range(finalMiddlePoints):
                # cv2.circle(frame, (int(original_ptsR[i][0][0]) - 250,int(original_ptsR[i][0][1])), 5, (0,0,255), -1)
                middleBirdEye = np.append(middleBirdEye,[rx[i] - singleLaneShift,startingY - i * windowHeight])
                # print("right")

    if len(middleBirdEye) != 0:
        middleBirdEye = middleBirdEye.reshape(-1, 1, 2)
        originalMiddle = cv2.perspectiveTransform(middleBirdEye, inv_matrix)
        # print(originalMiddle)
        # for i in range(len(originalMiddle)-1):
        #     cv2.line(frame, (int(originalMiddle[i][0][0]),int(originalMiddle[i][0][1])),(int(originalMiddle[i+1][0][0]),int(originalMiddle[i+1][0][1])), (0,0,255), 2)

    if len(middleBirdEye)!= 0:
        middleBirdEye = middleBirdEye.reshape(-1, 1, 2)
        middleBirdEye = cv2.perspectiveTransform(middleBirdEye, inv_matrix)
        middleBirdEye = middleBirdEye.astype(int)
        lastMiddle = middleBirdEye[len(middleBirdEye)//2][0][0]
        cv2.circle(frame, (middleBirdEye[len(middleBirdEye)//2][0][0],middleBirdEye[len(middleBirdEye)//2][0][1]), 5, (0,0,255), -1)
    #     print(middleBirdEye[len(middleBirdEye)//2][0][0],middleBirdEye[len(middleBirdEye)//2][0][1])
    # else:
    #     print(lastMiddle)
    # print(lx)
    # print("_-----------")
    # print(rx)
    cv2.imshow("Original", frame)
    # cv2.imshow("Bird's Eye View", transformed_frame)
    # cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    

    # result.write(frame)

    if cv2.waitKey(10) == 27:
        # cv2.imwrite("bird_for_track.jpg",frame)
        break
# result.release() 
cv2.destroyAllWindows()