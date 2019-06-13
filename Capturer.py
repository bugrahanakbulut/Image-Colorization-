import cv2

# frame's count
capturedFrameCount = 0
### MAKE THIS SMALLER IF YOU WANT TO CAPTURE IMAGES MORE FREQUENTLY, OR LARGER OTHERWISE ###
skipFrameSize = 1
# loads given videoFrames file to the memory
video = cv2.VideoCapture('Babysitting Cute Wolf Pups - Snow Wolf Family And Me - BBC.mp4') # set to videoFrames file


while True:
    isFrameValid, frame = video.read() # reads frame from given videoFrames file
    if isFrameValid: # if end of the file is not reached
        if capturedFrameCount % skipFrameSize != 0: # handles skip size
            capturedFrameCount = capturedFrameCount + 1
            continue
        imageDest = 'videoFrames/VideoFramesGray/imageFrame' + str(capturedFrameCount) + '.jpg' # prepares image destination
        print(imageDest)
        cv2.imwrite(imageDest, frame) # saves image to given destination
        capturedFrameCount = capturedFrameCount + 1
    else:
        break
