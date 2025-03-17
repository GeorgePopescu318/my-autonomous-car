import cv2

vidcap = cv2.VideoCapture('1.mp4')
success, frame = vidcap.read()
while success:
    success, frame = vidcap.read()
    print(frame.shape)
    cv2.imshow("aa",frame)

    if cv2.waitKey(10) == 27:
        # cv2.imwrite("bird_for_track.jpg",frame)
        break

vidcap.release()
cv2.destroyAllWindows()
