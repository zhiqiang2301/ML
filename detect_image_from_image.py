import cv2
from matplotlib import pyplot as plt
import numpy

img = cv2.imread('1.jpg')


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bin_img = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

M = cv2.moments(bin_img)


Cx = int(M['m10'] / M['m00'])
Cy = int(M['m01']/M['m00'])

a = int((M['m20']/M['m00']) - (Cx**2))
b = int(2*((M['m11']/M['m00']) - (Cx*Cy)))
c = int((M['m02']/M['m00'])-(Cy**2))

w = int((8*(a+c-(b**2 + (a-c)**2)**(1/2)))**(1/2))
l = int((8*(a+c+(b**2 + (a-c)**2)**(1/2)))**(1/2))

x = int(Cx-(l/2))
y = int(Cy-(w/2))
cv2.rectangle(img, (x, y), (x+l, y+w), (255, 0, 0), 5)
plt.subplot(1, 1, 1)
plt.imshow(img)
plt.show()
