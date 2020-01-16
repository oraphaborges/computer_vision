import cv2
import pytesseract as tesseract

img = cv2.imread('input/texto.jpg',0)
text = tesseract.image_to_string(img,'por')
print(text)