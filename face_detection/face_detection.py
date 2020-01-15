import cv2

# Carregando os modelos de classificação para olhos, rosto e sorriso
eye_cascade  = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

imgs = ['friends.jpg','multi.jpg']
for src in imgs:
    # Carregando a imagem
    img = cv2.imread(f'input/{src}')

    # Convertendo para Preto e Branco para a Classificação
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Executand a classficação dos rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor =1.1, minNeighbors = 2)

    for (x, y, w, h) in faces:
        # desenhando um retangulo no rosto detectado
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Reduzindo a região de interesse os para melhor encontrar os olhos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Executando a classificação dos olhos
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor =1.1, minNeighbors = 2)
        for (ex,ey,ew,eh) in eyes:
            # desenhando um retangulo nos olhos detectados
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  

        # Executando a classificação dos olhos
        smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor =2.5, minNeighbors = 8)
        for (sx,sy,sw,sh) in smiles:
            # desenhando um retangulo nos olhos detectados
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)  
    cv2.imshow(src, img)
cv2.waitKey()