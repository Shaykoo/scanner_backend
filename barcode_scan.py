import cv2
import numpy as np
from pyzbar.pyzbar import decode


img = cv2.imread("jagota1.jpg")

for code in decode(img):
    print(code.data.decode("utf-8"))

cv2.imshow("image", img)
cv2.waitKey(20000)

# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#
#     if not success:
#         break
#
#     for code in decode(img):
#         decoded_data = code.data.decode("utf-8")
#         print("data", decoded_data)
#         rect_pts = code.rect
#
#         if decoded_data:
#             pts = np.array([code.polygon], np.int32)
#             cv2.polylines(img, [pts], True, (0, 255, 0), 3)
#             cv2.putText(img, str(decoded_data), (rect_pts[0], rect_pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
#
#     cv2.imshow("image", img)
#     cv2.waitKey(1)
#
# cap.release()
