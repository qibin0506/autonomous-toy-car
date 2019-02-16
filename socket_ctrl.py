import socketserver
import serial
import threading
from picamera import PiCamera
import time
import random
import cv2
import io
import numpy as np
import model
import tensorflow as tf

trainMode = True # or False for autonomous model

ip = "192.168.1.1"
image_height = 240
image_width = 320
clipping_pixel = 80
fixed_speed = 25.0

lock = threading.Lock()

camera = PiCamera()
# camera.saturation = 80
# camera.brightness = 50
camera.shutter_speed = 6000000
# camera.iso = 800
camera.resolution = (image_width, image_height)
# camera.framerate = 32
camera.hflip = False
camera.vflip = False

speed = 0.0
angle = 0.0
ser = serial.Serial('/dev/ttyUSB0', 38400, timeout=1)
auto = False

if not trainMode:
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "./model/tf_result.ckpt")

    print("load model complete")


class VideoProcessor(object):
    def write(self, buf):
        global angle, speed
        if speed == 0.0 or speed == -10000.0:
            return

        if not buf.startswith(b'\xff\xd8'):
            return

        file = io.open('./data/img/%s_%s_%s_%s.jpg'
                       % (str(angle), str(speed), str(time.time()), str(random.randint(1, 50))), 'wb')
        file.write(buf)


class CaptureThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # capture data
        try:
            camera.start_preview()
            time.sleep(2)
            camera.start_recording(VideoProcessor(), format='mjpeg')
            camera.wait_recording(60000)
            camera.stop_recording()
        except:
            pass


class AutoDrivingVideoProcessor(object):
    def write(self, buf):
        if not buf.startswith(b'\xff\xd8'):
            return

        global model, sess, angle, speed

        data = np.fromstring(buf, dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        image = image[clipping_pixel:]
        image = image / 255.0 - 0.5

        angle_t = model.result.eval(feed_dict={model.tf_x: [image], model.keep_prob: 1}, session=sess)
        angle_t = angle_t.item((0, 0))
        angle_t = (angle_t + .5) * 180.0
        if angle_t > 180.0:
            angle_t = 180.0
        angle = float(int(angle_t))
        print("angle: ", angle_t, angle)
        speed = fixed_speed


class AutoDrivingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        # capture data
        while True:
            try:
                camera.start_preview()
                time.sleep(3)
                camera.start_recording(AutoDrivingVideoProcessor(), format='mjpeg')
                camera.wait_recording(6000000)
                camera.stop_recording()
            except:
                pass


class CtrlThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while 1:
            response = ser.readline()
            print(response)

            try:
                if speed == 0.0 and angle == 0.0:
                    continue

                if speed == -10000.0:
                    ser.write(bytes("RASPI:0.0,0.0\n", "utf-8"))
                    continue

                if speed < 0.0:
                    real_speed = -fixed_speed
                else:
                    real_speed = fixed_speed

                ser.write(bytes("RASPI:" + str(real_speed) + "," + str(angle) + "\n", "utf-8"))
                # print("RASPI:" + str(real_speed) + "," + str(angle) + "\n")
            except Exception as e:
                print(e)

            time.sleep(0.01)


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global speed, angle, auto
        self.request.send(bytes("connected", "utf-8"))

        flag = True
        while flag:
            data = self.request.recv(4096).strip()
            temp = data.decode()

            if temp == "quit":
                exit()
                return

            if temp == "exit":
                speed = 0.0
                angle = 0.0
                flag = False
                continue

            if temp == "auto":
                print("self driving mode")
                auto = True
                AutoDrivingThread().start()
                continue

            temp = temp.split(",")
            if len(temp) == 2:
                lock.acquire()
                speed = float(temp[0])
                angle = float(temp[1])
                lock.release()


CtrlThread().start()
if trainMode:
    CaptureThread().start()

h, p = ip, 9999
server = socketserver.ThreadingTCPServer((h, p), TCPHandler)
server.serve_forever()
