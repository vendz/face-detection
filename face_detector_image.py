import cv2

# Load some pre-trained data of frontal-face opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect Face
img = cv2.imread('rdj.jpeg')

# Now we will convert the image to GreyScale as it is easier to recognize face in greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

# Draw rectangle around faces
for(x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)  # here (x,y) is upper left coordinates and (x+w,y+h) is lower right coordinates
                                                            # here (0, 255, 0) is for color and `2` is for thickness of rectangle

# this will give show the image and give title to the window that is opened
cv2.imshow('Vendz Face Detector', img)

# if you don't write 'cv2.waitKey()' then the image window will only pop-up for a split second and then close
# 'cv2.waitKey()' will keep the window open until you press a key
cv2.waitKey()


print("code completed!")