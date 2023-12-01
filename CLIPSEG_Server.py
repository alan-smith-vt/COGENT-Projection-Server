from CLIPSEG_Engine_ImagePrompt import CLIPSEG_Engine_ImagePrompt
from predsParser import plotData, subsampleCluster
from PIL import Image
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
import os
import sys
import json
import socket
import requests
from threading import Thread
import cv2
import numpy as np

#Preload ML Engine
CLIPSEG_Engine = CLIPSEG_Engine_ImagePrompt()

target = Image.open("photos/PXL_20230705_053948478.jpg")# Load your image
names = ["PXL_20230705_054626285"]# Define your prompts
defaultPrompts = [Image.open(f"photos/{i}.jpg") for i in names]
preds = CLIPSEG_Engine.main(target, defaultPrompts)

#subset, imgOverlayRGB = subsampleCluster(preds, target)
#imgOverlayBGR = cv2.cvtColor(imgOverlayRGB, cv2.COLOR_RGB2BGR)
#cv2.imwrite("results/imgOverlay" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".png",imgOverlayBGR)

img = []#not used currently, in future can facilitate image caching

class Handler(BaseHTTPRequestHandler):
        
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><body><h1>Hello World!</h1></body></html>", "utf-8"))

    def do_POST(self):
        global img, prompts

        length = int(self.headers.get('content-length'))
        request_type = self.headers.get('Request-Type')
        headers = self.headers

        def processImage(self, engine, prompts, cache = False):
            
            global img#one of these isn't needed I swear

            if not cache:
                data = self.rfile.read(length)
                imgRGB = cv2.imdecode(np.frombuffer(data, dtype='uint8'), 1)
                imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
                cv2.imwrite("results/imgHL2-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".png",imgRGB)

                img = Image.fromarray(imgBGR)
                
            #Debugging stuff: force pre-loaded image for now
            img = Image.open("photos/PXL_20230705_053948478.jpg")
            
            #Process Image
            print("starting ML routine " + datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            preds = engine.main(img, prompts)

            #process the first image prompt for now
            subset, imgOverlayRGB = subsampleCluster(preds, img)
            imgOverlayBGR = cv2.cvtColor(imgOverlayRGB, cv2.COLOR_RGB2BGR)
            cv2.imwrite("results/imgOverlay" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".png",imgOverlayBGR)

            print("Sending results to HL2" + datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            self.send_response(200)
            self.send_header("Content-type", "text")
            self.send_header('Request-Timestamp', datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            self.end_headers()
            
            print("subset shape: " + str(subset.shape))
            results_bytes = bytes(json.dumps(subset.tolist()),'utf-8')
            print("Bytes size = " + str(len(results_bytes)))
            print('Sending: (total size)' + str(len(str(json.dumps(subset.tolist())))) + '\nFirst lines: ' + str(json.dumps(subset.tolist())[:200]))
            self.wfile.write(bytes(json.dumps(subset.tolist()),'utf-8'))
      
        if request_type == 'SendImage':
            print('Recieved Image. Processing with CLIPSEG model...' + datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            processImage(self, CLIPSEG_Engine, defaultPrompts)
        elif request_type == 'CacheImage':
            global img
            data = self.rfile.read(length)
            imgRGB = cv2.imdecode(np.frombuffer(data, dtype='uint8'), 1)
            imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
            cv2.imwrite("results/imgHL2-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".png",imgRGB)
            img = Image.fromarray(imgBGR)

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Request-Timestamp', datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            self.end_headers()
            self.wfile.write(bytes("We got yer picture.", "utf-8"))
            
        elif request_type == 'FindWaterBottle':
            print('Recieved request to find water bottle using cached image.\n Processing with CLIPSEG model...' + datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            names = ["PXL_20230705_054626285"]# Define your prompts
            prompt = [Image.open(f"photos/{i}.jpg") for i in names]
            processImage(self, CLIPSEG_Engine, prompt, cache = True)

        elif request_type == 'FindBolt':
            print('Recieved request to find bolt using cached image.\n Processing with CLIPSEG model...' + datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            names = ["PXL_20230717_071533443.NIGHT"]# Define your prompts
            prompt = [Image.open(f"photos/{i}.jpg") for i in names]
            processImage(self, CLIPSEG_Engine, prompt, cache = True)

        elif request_type == 'FindCalculator':
            print('Recieved request to find calculator using cached image.\n Processing with CLIPSEG model...' + datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
            names = ["PXL_20230717_090620012.NIGHT"]# Define your prompts
            prompt = [Image.open(f"photos/{i}.jpg") for i in names]
            processImage(self, CLIPSEG_Engine, prompt, cache = True)

def send_ip_to_hl2(hostName, serverPort, hl2_ip):

    hl2_port = 4444
    hl2_endpoint = "http://{}:{}".format(hl2_ip,hl2_port)

    headers = {
        'Content-Type': 'application/json"',
    }
    data = {"ipAddress":hostName, "port":str(serverPort)}

    connected = False
    while not connected:
        try:
            print("Sending Post! Connected = " + str(connected))
            requests.post(hl2_endpoint, json=data, headers=headers) 
            if not connected:
                print('Connecting to HoLoLens 2 device...')
            connected = True
        except:
            connected = False
            pass
        sleep(1)
        
def get_ip_manually(device,fname):
    default_ip = ''
    out_path = ''
    if os.path.exists(fname):
        with open(fname,'r') as f:
            try:
                default_ip = f.readlines()[0]        
            except:
                pass
    IP = input("\nEnter Your {} local IP then press ENTER (default: {}): ".format(device,default_ip)) 
    if IP == '':
        IP = default_ip           
    with open(fname,'w') as f:
        f.write(IP)

    return IP

editor = False

if editor:
    HOST = '10.0.0.182'#home desktop wifi
    hl2_ip = HOST#Editor (home desktop wifi)
else: 
    HOST = '10.0.0.182'#Netgear offline LAN Desktop
    #hl2_ip = '192.168.1.3'#Actual HL2 (offline LAN)
    hl2_ip = '10.0.0.209'#Actual HL2 (home desktop wifi)


PORT = 8090
time_out = 1e-6

server = HTTPServer((HOST, PORT), Handler)
server.socket.settimeout(time_out)


#hl2_ip = get_ip_manually('HoloLens 2','hl2_ip.txt')

#thread = Thread(target = send_ip_to_hl2, args = (HOST, PORT, hl2_ip))
#thread.start()

try:
    print("\nServer now running on http://%s:%s" % (HOST, PORT)+ datetime.utcnow().strftime('\t%Y-%m-%d %H:%M:%S.%f')[:-3])
    server.serve_forever()
except KeyboardInterrupt:
    pass

server.server_close()

