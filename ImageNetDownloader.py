import urllib.request
import socket

# sets the default timeout for new socket objects to 5 seconds
socket.setdefaulttimeout(5)

# opens the file which contains urls of small portion of imagenet dataset
file = open("DataSets/download.txt", "r")

# iterates through each image URL and downloads the image to the dataset folder
for i, line in enumerate(file):
    try:
        urllib.request.urlretrieve(line, "DataSets/semanticInterpret/" + str(i) + ".jpg")
        print(i)
    except Exception as e:
        print(e)
        continue
