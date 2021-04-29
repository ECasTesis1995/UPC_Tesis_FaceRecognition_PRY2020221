from flask import Flask, render_template, Response, request

import cv2
import numpy as np
import os
import imutils
from datetime import datetime

from flask_mail import Mail, Message

app = Flask(__name__)

mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    #PONER EL USUARIO Y PASWORD DE CORREO ORIGEN:
    "MAIL_USERNAME": 'reconocimientofacialtesis843@gmail.com',
    "MAIL_PASSWORD": 'pollolol'
}
#ON. (https://myaccount.google.com/lesssecureapps)
#Enable IMAP Access (https://mail.google.com/mail/#settings/fwdandpop)
app.config.update(mail_settings)
mail = Mail(app)
#funcion que envia el mensaje
def sendMessage():
    with app.app_context():
        msg = Message(subject="Registro de sospechosos reconocidos",
                      sender=app.config.get("MAIL_USERNAME"),
                      #PONER EL CORREO DEL DESTINO:
                      recipients=["reconocimientofacialtesis843@gmail.com"], # replace with your email for testing
                      body="Hemos reconocido los siguientes usuarios, en nuestra base de sospechosos...!!")
        now = datetime.now()
        fecha = now.strftime("%d-%m-%Y")
        with app.open_resource("Reconocidos_"+fecha+".csv") as fp:
            msg.attach("Reconocidos_"+fecha+".csv", "text/plain", fp.read())
        mail.send(msg)
#funcion que reconoce el rostro
def gen_frames():  # generate frame by frame from camera
    dataPath = './data' #Cambia a la ruta donde hayas almacenado Data
    imagePaths = os.listdir(dataPath)
    print('imagePaths=', imagePaths)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Leyendo el modelo
    face_recognizer.read('./modelo/modeloLBPHFace.xml')
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('./video/video1.mp4')
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    #camera = cv2.VideoCapture('./video/video1.mp4')  # use 0 for web camera
    personasReconocidas = [] 
    while True:
        success,frame = cap.read()
        #frame = imutils.resize(frame, width=1200)
        if success == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            # LBPHFace
            if result[1] < 70:#43:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                #capturo fecha y hora
                now = datetime.now()
                fecha = now.strftime("%d/%m/%Y")
                hora = now.strftime("%H:%M:%S")
                datos = [imagePaths[result[0]], fecha, hora]
                personasReconocidas.append(datos)
                x = len(personasReconocidas)
                #CAMBIAR EL NUMERO 100, PARA QUE ENVIE UNA CANTIDAD DE "X" DE VECES DETECTADOS
                print("... deteccion Nro. : ", x)
                if(x == 100):
                    sendEmail(personasReconocidas)
                    print("Mensaje enviado ...")
                    personasReconocidas = []
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)            
            
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
#Funcion que genera el archivo para ser enviado
def sendEmail(personasReconocidas):
    #print(personasReconocidas)
    now = datetime.now()
    fecha = now.strftime("%d-%m-%Y")
    with open("Reconocidos_"+fecha+".csv", "w") as f:
        for num in personasReconocidas:
            f.write(num[0]+', '+num[1]+', '+num[2]+'\n')
    #send email with data:
    sendMessage()

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    return render_template('index.html')
    #return render_template('index.html')
@app.route("/home", methods=["POST"])
def home():
    user_name = request.form.get("email")
    return render_template('home.html', user_name=user_name)
    #return render_template('index.html')

@app.route("/captura")
def captura():
    return render_template('captura.html')

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_files = request.files.getlist("file[]")
    folder_name = request.form.get("name")
    path = "./data/"
    joinPath = path+folder_name
    createFolder(joinPath)
    #AQUI HACER UN POSIBLE CAMBIO***************************************
    #guardo el contenido de las imagenes seleccionadas
    contador = contarFolder(joinPath)
    for image in uploaded_files:     #image will be the key 
        file_name = image.filename
        image.save(joinPath+"/"+str(contador)+".jpg")
        contador += 1
    return render_template('upload.html', contador=contador)
def contarFolder(path):
    imagePaths = os.listdir(path)
    contador = 0
    for image in imagePaths:     #image will be the key 
        contador += 1
    return contador
#Funcion que crea la carpeta
def createFolder(path):
    #access_rights = 0o755    
    try:
        #os.mkdir(path, access_rights)
        os.mkdir(path)
    except:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)
@app.route("/entrenar")
def entrenar():
    return render_template('entrenar.html')

@app.route("/entrenando", methods=["POST"])
def entrenando():
    if request.form['aceptar'] == '  SI  ':
        #AQUI PONER LA FUNCION QUE HACE QUE ENTRENE AL MODELO.
        resultado = entrenarModelo()
        return render_template('entrenando.html', resultado = resultado)
    elif request.form['aceptar'] == '  NO  ':
        return render_template('captura.html')
    else:
        return ""
#Funcion para hacer el entrenamiento y generar el modelos
def entrenarModelo():
    dataPath = './data' #Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes')

        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            #image = cv2.imread(personPath+'/'+fileName,0)
            #cv2.imshow('image',image)
            #cv2.waitKey(10)
        label = label + 1
    #Recolectamos datos hasta aqui ahora comienzo con el entrenamiento.

    # Métodos para entrenar el reconocedor
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    # Almacenando el modelo obtenido
    #face_recognizer.write('modeloEigenFace.xml')
    #face_recognizer.write('modeloFisherFace.xml')
    face_recognizer.write('./modelo/modeloLBPHFace.xml')
    return "Modelo almacenado..."

@app.route("/reconocimiento")
def reconocimiento():
    return render_template('reconocimiento.html')

if __name__ == "__main__":
    #app.run()
    app.run(debug = True)

    