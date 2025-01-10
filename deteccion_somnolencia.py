import numpy as np
import cv2
import tflite_runtime.interpreter as tflite 
from imutils.video import VideoStream
import time
import RPi.GPIO as GPIO

def load_model_and_cascade():
    print("Cargando modelo ResNeXt-50 y detector de rostros...")
    tflite_model_path = '/home/pi/Marco/ResNeXt_50_Modelo_128x128-L_vf-28.2(5)-TFL.tflite'
    face_cascade = cv2.CascadeClassifier('/home/pi/Marco/haarcascade_frontalface_default.xml')
    
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    time.sleep(2.0)
    print("Modelo y detector de rostros cargados")
    return interpreter, face_cascade, input_details, output_details

def prepare_frame(face):
    face_resized = cv2.resize(face, (128, 128))
    face_normalized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_normalized = face_normalized / 255.0
    face_normalized = np.expand_dims(face_normalized, axis=-1)
    face_normalized = np.expand_dims(face_normalized, axis=0).astype(np.float32)
    return face_normalized

def calculate_ear(eye_points):
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def initialize_buzzer():
    pin = 7 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 500) 
    pwm.start(0) 
    return pwm


def calibrate_ear_threshold(cap, face_cascade, input_details, output_details, interpreter):
    print("Calculando umbral EAR...")
    calibration_frames = 16
    calibration_data = []
    while len(calibration_data) < calibration_frames:
        frame = cap.read()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                expand_top = 0.28
                expand_bottom = 0.20
                expand_width = 0.45

                expand_width_value = int(expand_width * w)
                expand_height_top = int(expand_top * h)
                expand_height_bottom = int(expand_bottom * h)

                new_x = max(0, x - expand_width_value // 2)
                new_y = max(0, y - expand_height_top)
                new_w = w + expand_width_value
                new_h = h + expand_height_top + expand_height_bottom
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 1)

                face = frame[new_y:new_y + new_h, new_x:new_x + new_w]
                prepared_face = prepare_frame(face)

                interpreter.set_tensor(input_details[0]['index'], prepared_face)
                interpreter.invoke()
                regression_output = interpreter.get_tensor(output_details[1]['index'])[0]
                landmarks = np.reshape(regression_output, (-1, 2)) * [new_w, new_h]
                landmarks += [new_x, new_y]

                ear_left = calculate_ear(landmarks[0:6])
                ear_right = calculate_ear(landmarks[6:12])
                ear_avg = (ear_left + ear_right) / 2.0

                calibration_data.append(ear_avg)
                break 

        cv2.imshow("Calibración de Umbral EAR", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    ear_threshold = np.mean(calibration_data) * 0.7
    print(f"Umbral EAR calculado: {ear_threshold}")

    return ear_threshold
    
    


def process_camera(interpreter, face_cascade, input_details, output_details, pwm):
    print("Iniciando el video...")
    cap = VideoStream(src=0).start()
    time.sleep(2.0)
    print("Video iniciado...")
    eye_closure_start=None
    ear_threshold = calibrate_ear_threshold(cap, face_cascade, input_details, output_details, interpreter)
    last_frame_time = time.time()
    while True:
        start_time = time.time()
        frame = cap.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))
        if len(faces) == 0:
             cv2.putText(frame, f"No hay rostros",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        else:
            for (x, y, w, h) in faces:
                
                expand_top=0.28
                expand_bottom=0.20
                expand_width=0.45
                
                expand_width_value=int(expand_width*w)
                expand_height_top=int(expand_top*h)
                expand_height_bottom=int(expand_bottom*h)
                
                new_x=max(0,x-expand_width_value//2)
                new_y=max(0,y-expand_height_top)
                new_w=w+expand_width_value
                new_h=h+expand_height_top+expand_height_bottom
                
                cv2.rectangle(frame,(new_x,new_y),(new_x+new_w,new_y+new_h),(255,0,0),1)
                
                face=frame[new_y:new_y+new_h,new_x:new_x+new_w]
                prepared_face=prepare_frame(face)
                
                interpreter.set_tensor(input_details[0]['index'],prepared_face)
                interpreter.invoke()
                regression_output=interpreter.get_tensor(output_details[1]['index'])[0]
                
                landmarks=np.reshape(regression_output,(-1,2))*[new_w,new_h]
                landmarks+=[new_x,new_y]
                
                ear_left=calculate_ear(landmarks[0:6])
                ear_right=calculate_ear(landmarks[6:12])
                ear_avg=(ear_left+ear_right/2.0)
                
                for (lx,ly) in landmarks:
                    cv2.putText(frame, f"Umbral EAR: {ear_threshold:.3f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
                    cv2.putText(frame, f"EAR Actual: {ear_avg:.3f}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
                    cv2.circle(frame,(int(lx),int(ly)),1,(0,255,0),-1)
                    
                if ear_avg < ear_threshold:
                    if eye_closure_start is None:
                        eye_closure_start=time.time()
                else:
                    eye_closure_start=None
                    
                if eye_closure_start and (time.time()-eye_closure_start)>=1.5:
                    cv2.putText(frame, "SOMNOLENCIA DETECTADA",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    #pwm.ChangeDutyCycle(50)
                    #time.sleep(2)
                    #pwm.changeDutyCycle(0)
                
                pass
                # Muestra los FPS
        #time_diff = time.time() - last_frame_time
        #fps = 1.0 / time_diff if time_diff > 0 else 0
        #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Actualiza el tiempo del último frame procesado
        #last_frame_time = start_time
        cv2.imshow('Detección de Somnolencia', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.stop()
    cv2.destroyAllWindows()
    #pwm.stop()
    #GPIO.cleanup()

def main():
    interpreter, face_cascade, input_details, output_details = load_model_and_cascade()
    pwm = initialize_buzzer()
    process_camera(interpreter, face_cascade, input_details, output_details, pwm)


if __name__ == '__main__':
    main()
