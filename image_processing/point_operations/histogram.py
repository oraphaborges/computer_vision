import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg

img = cv2.imread('input/forest.jpeg')
cv2.imshow('foreste',img)

# Histograma da imagem em tons de cinza
plt.hist(img.ravel(),256,[0,256])
plt.xlabel('Tonalidade de Cinza')
plt.ylabel('Quantidade de Pixels')
plt.suptitle('Histograma de tons de cinza')
plt.show()    

# Histograma da imagem colorida
histograma = cv2.calcHist(images=[img],channels=[0],mask=None,histSize=[256],ranges=[0,256])
cores = ('b','g','r')
for i, col in enumerate(cores):
    histograma = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histograma,color = col)
    plt.xlabel(f'Tonalidade de {col}')
    plt.ylabel('Quantidade de Pixels')
    plt.suptitle('Histograma dos canais RGB')
    plt.xlim([0,256])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()