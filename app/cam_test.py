import cv2

cam = cv2.VideoCapture(0)

ret, frame = cam.read()
cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
