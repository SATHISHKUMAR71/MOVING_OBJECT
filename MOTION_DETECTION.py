import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.DataFrame(columns=["Centroid (X)","Centroid (Y)","Radius","Area"])

def ROI(img,verices):

    mask_img = np.zeros_like(img)
    channels = img.shape[2]
    match_mask_color = (255,) * channels
    cv2.fillPoly(mask_img,verices,match_mask_color)
    masked_img = cv2.bitwise_and(img,mask_img)
    return masked_img

regions = [
    (497,296),
    (293,219),
    (196,181),
    (130,163),
    (5,142),
    (20,123),
    (55,127),
    (102,134),
    (155,140),
    (203,141),
    (251,132),
    (288,129),
    (347,114),
    (412,84),
    (453,67),
    (467,85),
    (421,119),
    (431,158),
    (475,192)
]
cap = cv2.VideoCapture('resources/vtest.avi.mp4')
# cap = cv2.VideoCapture(0)
_, frame01 = cap.read()
_, frame02 = cap.read()

while cap.isOpened():
    frame1 = ROI(frame01,np.array([regions],np.int32))
    frame2 = ROI(frame02, np.array([regions], np.int32))
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    ret,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations=3)
    contour,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame1,contour,-1,(0,255,0),2)
    for i in contour:
        # RECTANGLE
        # (x,y,w,h) = cv2.boundingRect(i)
        # print("Area: "+str(cv2.contourArea(i)),end=" ")
        # print("Centroid: "+str((x+x+w)/2)+","+str((y+y+h)/2))
        area = cv2.contourArea(i)
        (x,y),radius = cv2.minEnclosingCircle(i)
        tmp = pd.DataFrame({
            "Centroid (X)":[x],
            "Centroid (Y)":[y],
            "Radius":[radius],
            "Area":[area]
        })
        # print(tmp)
        df = df.append(tmp,ignore_index=True)
        if(area < 290):

            continue

        cv2.putText(frame1,"status: MOVEMENT",(3,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        # RECTANGLE
        # cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(frame1,(int(x),int(y)),int(radius),(0,0,255),2)
    # cv2.imshow("frame",frame1)
    cv2.imshow("frame01", frame01)
    tot_frame = cv2.bitwise_or(frame1,frame01)
    cv2.imshow("tot_frame",tot_frame)
    frame01 = frame02
    _,frame02 = cap.read()

    if(cv2.waitKey(10) == 27):
        break
cap.release()
cv2.destroyAllWindows()
print(df)