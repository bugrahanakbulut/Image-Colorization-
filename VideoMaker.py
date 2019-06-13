from glob import glob
import cv2

folder = "DataSets/colorized"
videoName = "video2.avi"

images = glob(folder + "/*.jpg")

frame = cv2.imread(images[0])
height, width, layers = frame.shape
video = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), 30.0,  (width, height))

for imageNum in range(len(images)):
    if imageNum > 2000:
        break
    x = cv2.imread(folder + "/imageFrame" + str(imageNum) + ".jpg")
    video.write(x)

cv2.destroyAllWindows()
video.release()