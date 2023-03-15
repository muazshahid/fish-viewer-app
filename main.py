import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import os
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import numpy as np
import time
import threading
import sys
from time import ctime
global image
from streamlit_webrtc import (AudioProcessorBase,ClientSettings,VideoProcessorBase,WebRtcMode,webrtc_streamer,)
import av
import pydub
import asyncio
import shutil
import queue
from aiortc.contrib.media import MediaPlayer
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from time import ctime
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import string
import random

ss_path="screenshots"
lock = threading.Lock()
img_container = {"img": None}
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
FLAGS(sys.argv) 

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)
    return result_str

######################################################################################################################################
######################################################################################################################################

def fish_track_cam(old_id,ss_clicked):
    # sess = tf.InteractiveSession()
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    # video_path = "data/video/test_video2.mp4"
    # initialize deep sort
    model_filename = 'data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    sei_count=0
    last_length=0
    makrell_count=0
    col1, col2, col3,col4,col5 = st.columns(5)

    col1=st.empty()
    col2=st.empty()
    col3=st.empty()
    col4=st.empty()
    col5=st.empty()

    class Video(VideoProcessorBase):

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            # uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
            if image.any(): 
            # load configuration for object detector
                config = ConfigProto()
                config.gpu_options.allow_growth = True
                session = InteractiveSession(config=config)
                STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
                input_size = FLAGS.size
                # video_path = FLAGS.video

                # load tflite model if flag is set
                if FLAGS.framework == 'tflite':
                    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
                    interpreter.allocate_tensors()
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    print(input_details)
                    print(output_details)
                # otherwise load standard tensorflow saved model
                else:
                    saved_model_loaded = tf.saved_model.load("output/yolov4-tiny-416", tags=[tag_constants.SERVING])
                    infer = saved_model_loaded.signatures['serving_default']

                out = None
                output="output/outvid.mp4"
                # get video ready to save locally if flag is set
                # if FLAGS.output:
                # by default VideoCapture returns float instead of int
                width = 640
                height = 480
                fps = 30
                codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(output, codec, fps, (width, height))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ss_button_man = st.empty()
                frame_num = 0
                if ss_button_man.button('Screenshot'):
                    ss_clicked=1
                # img_placeholder = st.empty()
                # while video is running
                while True:
                    frame_num +=1
                    print('Frame #: ', frame_num)
                    frame_size = frame.shape[:2]
                    image_data = cv2.resize(frame, (input_size, input_size))
                    image_data = image_data / 255.
                    image_data = image_data[np.newaxis, ...].astype(np.float32)
                    start_time = time.time()
                    if(ss_clicked==1):
                        # cv2.imwrite("ss.png",ss_frame)
                        cv2.imwrite(os.path.join(ss_path,get_random_string(5)+'.png'),frame)  
                        ss_clicked=0

                    # run detections on tflite if flag is set
                    if FLAGS.framework == 'tflite':
                        interpreter.set_tensor(input_details[0]['index'], image_data)
                        interpreter.invoke()
                        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                        # run detections using yolov3 if flag is set
                        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                            input_shape=tf.constant([input_size, input_size]))
                        else:
                            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                            input_shape=tf.constant([640, 480]))
                    else:
                        batch_data = tf.constant(image_data)
                        pred_bbox = infer(batch_data)
                        for key, value in pred_bbox.items():
                            boxes = value[:, :, 0:4]
                            pred_conf = value[:, :, 4:]

                    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                        scores=tf.reshape(
                            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                        max_output_size_per_class=50,
                        max_total_size=50,
                        iou_threshold=FLAGS.iou,
                        score_threshold=FLAGS.score
                    )

                    # convert data to numpy arrays and slice out unused elements
                    num_objects = valid_detections.numpy()[0]
                    bboxes = boxes.numpy()[0]
                    bboxes = bboxes[0:int(num_objects)]
                    scores = scores.numpy()[0]
                    scores = scores[0:int(num_objects)]
                    classes = classes.numpy()[0]
                    classes = classes[0:int(num_objects)]

                    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                    original_h, original_w, _ = frame.shape
                    bboxes = utils.format_boxes(bboxes, original_h, original_w)

                    # store all predictions in one parameter for simplicity when calling functions
                    pred_bbox = [bboxes, scores, classes, num_objects]

                    # read in all class names from config
                    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                    # by default allow all classes in .names file
                    allowed_classes = list(class_names.values())

                    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                    names = []
                    deleted_indx = []
                    for i in range(num_objects):
                        class_indx = int(classes[i])
                        class_name = class_names[class_indx]
                        if class_name not in allowed_classes:
                            deleted_indx.append(i)
                        else:
                            names.append(class_name)
                    names = np.array(names)
                    count = len(names)

                    bboxes = np.delete(bboxes, deleted_indx, axis=0)
                    scores = np.delete(scores, deleted_indx, axis=0)

                    # encode yolo detections and feed to tracker
                    features = encoder(frame, bboxes)
                    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                    #initialize color map
                    cmap = plt.get_cmap('tab20b')
                    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                    # run non-maxima supression
                    boxs = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    classes = np.array([d.class_name for d in detections])
                    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]       

                    # Call the tracker
                    tracker.predict()
                    tracker.update(detections)

                    # update tracks
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue 
                        bbox = track.to_tlbr()
                        class_name = track.get_class()
                        cv2.putText(frame, "Fish Count: {}".format(track.track_id), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)  

                       
                        if (bbox[2]>305 and bbox[2]< 345):
                            if (class_name=="makrell"):
                                fish_length=bbox[2]/8.3
                            elif (class_name=="sei"):
                                fish_length=bbox[2]/8.4 
                        elif (bbox[2]>372 and bbox[2]< 412):
                            if (class_name=="makrell"):
                                fish_length=bbox[2]/8.3
                            elif (class_name=="sei"):
                                fish_length=bbox[2]/8.4
                        else:
                            fish_length=0    
                        col1.metric("Count", int(track.track_id),"")    
                    # if enable info flag then print details about each track
                        if FLAGS.info:
                            print("Tracker ID: {}, Class: {}, Fish Length: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, fish_length ,(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                        if (fish_length!=0 and track.track_id-old_id>0):
                            timee=ctime()
                            log_data = pd.read_csv('logs.csv')
                            avg_length=log_data['fish_length'].mean() 
                            col2.metric("Average Length (cm)", "{:.2f}".format(avg_length),"")
                            if (class_name=="sei"):
                                sei_count+=1
                            if (class_name=="makrell"):
                                makrell_count+=1 
                            col3.metric("Total Sei", int(sei_count),"")
                            col4.metric("Total Makrell", int(makrell_count),"") 
                            col5.metric("Length (cm)", "{:.2f}".format(last_length),None )    
                            print("Tracker ID: {}, Class: {}, Fish Length: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, fish_length ,(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                            with open("logs.csv",'r') as fl:
                                dframe=pd.DataFrame({"fish_name":class_name,"fish_length":fish_length,"inference_time":timee, "count":track.track_id},index=[0])
                                dframe.to_csv("logs.csv",index=False, mode='a',header=False)
                                if(fish_length!=0):
                                    last_length=fish_length  
                            old_id=int(track.track_id)
                            if save_fish:
                                cv2.imwrite(os.path.join(ss_path,(class_name+str(frame_num)+'.png')),ss_frame)      
                                dframe=pd.DataFrame({"fish_name":class_name,"fish_length":fish_length,"inference_time":timee, "count":track.track_id, "file_location": ss_path+class_name+"/"+str(frame_num)+'.png'},index=[0])
                                dframe.to_csv("logs.csv",index=False, mode='a',header=False) 

                        color = colors[int(track.track_id) % len(colors)]
                        color = [i * 255 for i in color]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                        
                        cv2.putText(frame, "Length (cm) : {:.2f}".format(last_length), (5, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1) 

                    # calculate frames per second of running detections
                    fps = 1.0 / (time.time() - start_time)
                    print("FPS: %.2f" % fps)
                    result = np.asarray(frame)
                    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return av.VideoFrame.from_ndarray(result, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=Video,
        async_processing=True,
    )        
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def fish_track(old_id,ss_clicked):
    # sess = tf.InteractiveSession()
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # video_path = "data/video/test_video2.mp4"
    # initialize deep sort
    model_filename = 'data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    avg_length=0
    sei_count=0
    makrell_count=0
    last_length=0
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    img_placeholder = st.empty()
    col1,col2,col3,col4,col5= st.columns(5)
    st.empty()
    col1=st.empty()
    col2=st.empty()
    col3=st.empty()
    col4=st.empty()
    col5=st.empty()


    # save_fish = st.checkbox('Save Sceenshots')
    save_fish= False
    if uploaded_video != None: 
    # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size
        # video_path = FLAGS.video

        # load tflite model if flag is set
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load("output/yolov4-tiny-416", tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
               
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        
        vid = cv2.VideoCapture(vid)
        out = None
        output="output/outvid.mp4"
        # get video ready to save locally if flag is set
        # if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(output, codec, fps, (width, height))
        sei_count=0
        makrell_count=0
        frame_num = 0
        ss_button_man = st.empty()
        # ss_button_ph = st.empty()
        orignal_frame="ss.png"
        if ss_button_man.button('Screenshot'):
            ss_clicked=1
        
        
        
        # while video is running
        while True:

            saved_frame="screenshots/"+get_random_string(5) +".png"
            return_value, frame = vid.read()
            if frame is not None:
                ss_frame=frame.copy()
            else:
                break    
            
            if(ss_clicked==1):
                # cv2.imwrite("ss.png",ss_frame)
                cv2.imwrite(os.path.join(ss_path,get_random_string(5)+'.png'),ss_frame)  
                ss_clicked=0
                
            # return_value, frame = vid.read()    
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num +=1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()
            cv2.imwrite("ss.png",ss_frame)
            


            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)
            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                cv2.putText(frame, "Fish Count: {}".format(track.track_id), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                # old_id=int(track.track_id)  
                if (bbox[2]>315 and bbox[2]< 335):
                    if (class_name=="makrell"):
                        fish_length=bbox[2]/8.3
                    elif (class_name=="sei"):
                        fish_length=bbox[2]/8.4 
                elif (bbox[2]>382 and bbox[2]< 402):
                    if (class_name=="makrell"):
                        fish_length=bbox[2]/8.3
                    elif (class_name=="sei"):
                        fish_length=bbox[2]/8.4
                else:
                    fish_length=0    
            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {}, Fish Length: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, fish_length ,(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                col1.metric("Count", int(track.track_id),None)
                col2.metric("Length (cm)", "{:.2f}".format(last_length),None )
                col4.metric("Total Sei", int(sei_count),None)
                col5.metric("Total Makrell", int(makrell_count),None)
                col3.metric("Average Length (cm)", "{:.2f}".format(avg_length),None)
                if (fish_length!=0 and track.track_id-old_id>0):
                    timee=ctime()
                    log_data = pd.read_csv('logs.csv')
                    avg_length=log_data['fish_length'].mean() 
                    
                    if (class_name=="sei"):
                        sei_count+=1
                    if (class_name=="makrell"):
                        makrell_count+=1
                    col3.metric("Total Sei", int(sei_count),"")
                    col4.metric("Total Makrell", int(makrell_count),"")        
                    print("Tracker ID: {}, Class: {}, Fish Length: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, fish_length ,(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    with open("logs.csv",'r') as fl:
                        dframe=pd.DataFrame({"fish_name":class_name,"fish_length":fish_length,"inference_time":timee, "count":track.track_id},index=[0])
                        dframe.to_csv("logs.csv",index=False, mode='a',header=False)
                        if(fish_length!=0):
                            last_length=fish_length      
                        old_id=int(track.track_id)
                        if save_fish:
                            cv2.imwrite(os.path.join(ss_path,(class_name+str(frame_num)+'.png')),ss_frame)      
                            dframe=pd.DataFrame({"fish_name":class_name,"fish_length":fish_length,"inference_time":timee, "count":track.track_id, "file_location": ss_path+class_name+"/"+str(frame_num)+'.png'},index=[0])
                            dframe.to_csv("logs.csv",index=False, mode='a',header=False)              
                
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                # if(fish_length!=0):
                cv2.putText(frame, "Length (cm) : {:.2f}".format(last_length), (5, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
# "Length (cm)", "{:.2f}".format(last_length)

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_placeholder.image(result)
        vid.release()
        cv2.destroyAllWindows()


def main():    
    ss_flag=0
    st.title('Fish Viewer App')
    read_me = st.markdown("""The application will then track each individual fish coming from the right edge traveling towards the left edge. For each fish it will add 1 to a counter, this is for counting the total amount of fish that has been caught on video. While the fish is traveling across the screen, the application will try to estimate the species and measuring the length of the fish from head to tail fin. """
    )
    st.sidebar.title("Select Mode")
    choice  = st.sidebar.selectbox("Mode",("Fish Detection (Video)","Fish Detection (Camera)","About"))
    if choice == "Fish Detection (Video)":
        old_id=0
        fish_track(old_id,0)
    
    elif choice == "Fish Detection (Camera)":
        old_id=0
        fish_track_cam(old_id,0)

    elif choice == "About":
        print()
        

if __name__ == '__main__':
		main()	
