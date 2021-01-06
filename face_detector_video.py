import cv2

# Load some pre-trained data of frontal-face opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)    # here (0) means it will select default cam
                                # you can also put a filename insted of (0)

# Iterate forever over frames
while True:
    # read the current frame
    successful_frame_read, frame=webcam.read()


    # Now we will convert the image to GreyScale as it is easier to recognize face in greyscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    # Draw rectangle around faces
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)  # here (x,y) is upper left coordinates and (x+w,y+h) is lower right coordinates
                                                                  # here (0, 255, 0) is for color and `2` is for thickness of rectangle

    # this will give show the image and give title to the window that is opened
    cv2.imshow('Vendz Face Detector', frame)

    # here inside 'waitKey(1)' are putting (1) because if we don't put anything then it won't play video in real-time
    # and display each frame after a key is pressed on keyboard, so by putting (1) it will wait 1 millisecond and then
    # display next frame
    key = cv2.waitKey(1)

    # Stop is Q key is pressed
    if key==81 or key==113:         # here 113 is for 'q' and 81 is for 'Q'
        break
