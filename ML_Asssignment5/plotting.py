import matplotlib.pyplot as plt
import matplotlib.image as imread
folder = 'Data/'
# plot first few images
for i in range(9):
 # define subplot
 plt.subplot(330 + 1 + i)
 # define filename
 filename = folder + 'horse' + str(i) + '.jpg'
 # load image pixels
 image = imread.imread(filename)
 # plot raw pixel data
 plt.imshow(image)
# show the figure
plt.show()

folder = 'Data/'
# plot first few images
for i in range(9):
 # define subplot
 plt.subplot(330 + 1 + i)
 # define filename
 filename = folder + 'monkey' + str(i) + '.jpg'
 # load image pixels
 image = imread.imread(filename)
 # plot raw pixel data
 plt.imshow(image)
# show the figure
plt.show()
