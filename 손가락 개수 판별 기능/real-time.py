import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO
import webbrowser
#import subprocess

#command = "python -m streamlit run test_file1.py"
url = 'http://127.0.0.1:5501/web_expression/48up_kiosk.html'

hand_detection_method = 'yolo'

if hand_detection_method is 'solo':
    hand = SOLO(weights=r'D:\01-폴더-임재윤\##(고)3학년4반\Kiosk_Project\Gender-and-Age-Detection\Unified-Gesture-and-Fingertip-Detection-master\weight\solo.h5', threshold=0.8)
if hand_detection_method is 'yolo':
    hand = YOLO(weights=r'D:\01-폴더-임재윤\##(고)3학년4반\Kiosk_Project\Gender-and-Age-Detection\Unified-Gesture-and-Fingertip-Detection-master\weight\yolo.h5', threshold=0.8)
else:
    assert False, "'" + hand_detection_method + \
                  "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"

fingertips = Fingertips(weights=r'D:\01-폴더-임재윤\##(고)3학년4반\Kiosk_Project\Gender-and-Age-Detection\Unified-Gesture-and-Fingertip-Detection-master\weight\fingertip.h5')

cam = cv2.VideoCapture(0)
print('Unified Gesture & Fingertips Detection')

while True:
    ret, image = cam.read()

    if ret is False:
        break

    # hand detection
    tl, br = hand.detect(image=image)

    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        # gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        # post-processing
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]
        

        # drawing
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        detected_fingers = 0

        for c, p in enumerate(prob):
            if p > 0.5:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
                detected_fingers += 1
            index = index + 2

        if detected_fingers >= 4:
            print("more than 3")
            webbrowser.open(url)
            break
            #result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            #print("stdout:", result.stdout)
            #print("stderr:", result.stderr)
            #print("Return code:", result.returncode)


    if cv2.waitKey(1) & 0xff == 27:
        break

    # display image
    cv2.imshow('Unified Gesture & Fingertips Detection', image)


cam.release()
cv2.destroyAllWindows()
