
import cv2 as cv
import numpy as np
import math
import pyzbar.pyzbar as pyzbar
import betterLook
import mediapipe as mp
import time

# Variable
camID = 0  # camera ID, or pass string as filename. to the camID

# Real world measured Distance and width of QR code
KNOWN_DISTANCE = 8.0708661417  # inches
KNOWN_WIDTH = 1.5748031496  #  inches

# define the fonts
fonts = cv.FONT_HERSHEY_COMPLEX
Pos =(50,50)
# colors (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
GOLD = (0, 255, 215)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 230)

# functions

# finding Distance between two points


def eucaldainDistance(x, y, x1, y1):

    eucaldainDist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return eucaldainDist

# focal length finder function


def focalLengthFinder(knowDistance, knownWidth, widthInImage):
    '''This function calculates the focal length. which is used to find the distance between  object and camera 
    :param1 knownDistance(int/float) : it is Distance form object to camera measured in real world.
    :param2 knownWidth(float): it is the real width of object, in real world
    :param3 widthInImage(float): the width of object in the image, it will be in pixels.
    return FocalLength(float): '''
    
    focalLength = ((widthInImage * knowDistance) / knownWidth)
    print(widthInImage)
    print(knowDistance)
    print(knownWidth)
    print(focalLength)

    return focalLength

def distanceFinder(focalLength, knownWidth, widthInImage):
    '''
    This function basically estimate the distance, it takes the three arguments: focallength, knownwidth, widthInImage
    :param1 focalLength: focal length found through another function .
    param2 knownWidth : it is the width of object in the real world.
    param3 width of object: the width of object in the image .
    :returns the distance:


    '''
    distance = ((knownWidth * focalLength) / widthInImage)
    return distance

def DetectQRcode(image):
    codeWidth = 0
    x, y = 0, 0
    euclaDistance = 0
    global Pos 
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.decode(Gray)
    for obDecoded in objectQRcode:

        points = obDecoded.polygon
        if len(points) > 4:
            hull = cv.convexHull(
                np.array([points for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        n = len(hull)
        # draw the lines on the QR code 
        for j in range(0, n):
            # print(j, "      ", (j + 1) % n, "    ", n)

            cv.line(image, hull[j], hull[(j + 1) % n], ORANGE, 2)
        # finding width of QR code in the image 
        x, x1 = hull[0][0], hull[1][0]
        y, y1 = hull[0][1], hull[1][1]
        
        Pos = hull[3]
        # using Eucaldain distance finder function to find the width 
        euclaDistance = eucaldainDistance(x, y, x1, y1)

        # retruing the Eucaldain distance/ QR code width other words  
        return euclaDistance


# creating camera object
camera = cv.VideoCapture(camID)

refernceImage = cv.imread("referenceImage0.png")
# getting the width of QR code in the reference image 
Rwidth= DetectQRcode(refernceImage)
print("rwidth:  ", Rwidth)

# finding the focal length 
focalLength = focalLengthFinder(KNOWN_DISTANCE, KNOWN_WIDTH, Rwidth)
print("Focal length:  ", focalLength)

counter =0

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.75, trackCon= 0.95):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z*w)
                lmList.append([id, cx, cy])


        return lmList


def main():

    pTime = 0
    cTime = 0
detector = handDetector()





while True:

    success, img = camera.read()
    img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    # finding width of QR code width in the frame
    codeWidth= DetectQRcode(img)
    
    if codeWidth is not None:
        
        # print("not none")
        Distance = distanceFinder(focalLength, KNOWN_WIDTH, codeWidth)
        betterLook.showText(img, f"Distnace: {round(Distance, 2)} Inches", Pos, GOLD, int(Distance / 2))


    key = cv.waitKey(1)

    cv.imshow("Image", img)
    cv.waitKey(1)

    if key == ord('s'):
        # saving frame
        counter += 1
        print("frame saved")
        cv.imwrite(f"frames/frame{counter}.png", frame)
    if key == ord('q'):
        break

        if __name__ == "__main__":
            main()
camera.release()
cv.destroyAllWindows()
out.release()
