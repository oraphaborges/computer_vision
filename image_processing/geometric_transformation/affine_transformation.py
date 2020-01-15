import cv2
import numpy as np

def identidade(image):
    (h,w) = image.shape[:2]
    M = np.float32([[1, 0, 0],
                    [0, 1, 0]])
    return  cv2.warpAffine(image,M,(w,h))

def reflexao(image,axis=1):
    return  cv2.flip( img, axis )

def rotacao(image, angulo, centro=None, escala=1.0):
    (h,w) = image.shape[:2]
    if centro is None:
        centro =(w/2,h/2)
    M = cv2.getRotationMatrix2D(centro,angulo,escala)
    return cv2.warpAffine(image,M,(w,h))

def traslacao(image,x,y):
    (h,w) = image.shape[:2]
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    return  cv2.warpAffine(image,M,(w,h))

def cisalhamento(image):
    (h,w) = image.shape[:2]
    M = np.float32([[0.75, 0.25, 0],
                    [0.25, 0.75, 0]])
    return  cv2.warpAffine(image,M,(w,h))

def cisalhamento_triangular(image):
    (h,w) = image.shape[:2]
    triangulo_origem = np.array( [   [83,90],
                                     [447,90],
                                     [83,472]
                                   ] ).astype(np.float32)
    triangulo_destino = np.array( [  [83,93],
                                     [447,90],
                                     [150,472]
                                    ] ).astype(np.float32)
    M = cv2.getAffineTransform(triangulo_origem, triangulo_destino)
    return cv2.warpAffine(image, M, (w,h))

img = cv2.imread('input/landscape.jpeg')

# IMAGEM ORIGINAL
cv2.imshow('img',img)

# IMAGEM IDENTIDADE
img_identidade = identidade(img)
cv2.imshow('img_identidade',img_identidade)

# IMAGEM REFLETIDA
img_refletida = reflexao(img)
cv2.imshow('img_refletida',img_refletida)

# IMAGEM ROTACIONADA
img_rotacionada = rotacao(img,60)
cv2.imshow('img_rotacionada',img_rotacionada)

# IMAGEM TRANSLADADA
img_transalada = traslacao(img,200,100)
cv2.imshow('img_transalada',img_transalada)

# IMAGEM CISALHADA
img_cisalhada = cisalhamento(img)
cv2.imshow('img_cisalhada',img_cisalhada)

# DESENHO DDO TRIANGULO
cv2.line(img,(83,90), (447,90) ,(0,255,0))
cv2.line(img,(447,90),(83,472) ,(0,255,0))
cv2.line(img,(83,472),(83,90)  ,(0,255,0))
cv2.circle(img,(83,90),5,(0,0,255),-1)
cv2.circle(img,(447,90),5,(0,0,255),-1)
cv2.circle(img,(83,472),5,(0,0,255),-1)

# IMAGEM CISALHADA TRIANGULAR
cv2.imshow('img pontuada',img)
cisalhamento_triangular = cisalhamento_triangular(img)
cv2.imshow('cisalhamento_triangular',cisalhamento_triangular)

cv2.waitKey(0)
cv2.destroyAllWindows()