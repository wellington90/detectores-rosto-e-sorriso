import cv2

# carrega os detectores pré-treinados de rosto e sorriso
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# inicializa a webcam
cap = cv2.VideoCapture(0)

# define a fonte e a cor do texto
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)

while True:
    # captura o quadro atual da webcam
    ret, frame = cap.read()

    # converte o quadro para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta as faces no quadro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # percorre todas as faces detectadas
    for (x, y, w, h) in faces:
        # desenha um retângulo ao redor de cada face detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # obtém a região de interesse (ROI) da face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # detecta os sorrisos na ROI da face
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        # verifica se há sorrisos na ROI da face
        if len(smiles) > 0:
            # desenha um retângulo ao redor de cada sorriso detectado
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)

            # adiciona o texto "Você está sorrindo!" na imagem
            cv2.putText(frame, 'Voce esta sorrindo!', (x, y - 10), font, 0.8, color, 2)

    # exibe o quadro na janela
    cv2.imshow('Sorriso Detector', frame)

    # espera por uma tecla ser pressionada para encerrar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libera a webcam e fecha a janela
cap.release()
cv2.destroyAllWindows()
