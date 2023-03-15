# Fish Viewer Application
A fish viewer application that uses deep learning models to detect fish types and the length of fish using an image, video or a camera input.
## General description
The application will load the live video from a webcam or similar. The application will then track each individual fish coming from the right edge traveling towards the left edge. For each fish it will add 1 to a counter, this is for counting the total amount of fish that has been caught on video. While the fish is traveling across the screen, the application will try to estimate the species and measuring the length of the fish from head to tail fin, as well as measuring the total area of the contour/silhouette of the fish. The data from this shall be appended to a csv file, with date, time, length, species and displayed on-screen. The species is primarily Gadinae, Gadidae family.
The application using a YOLOv4-tiny model which does the object detection part and then uses yolov4-deepsort, we take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to keep track of fishes that pass through frame in the input. 
For the application design part, the streamlit library is used to design the application and interface the camera and video input for the app.

## Demo
The GIF below shows the working of application.
![fish_viewer_app](https://user-images.githubusercontent.com/65394988/225229463-0f0a7553-9308-4b3f-9fe0-09d3ac750b27.gif)

## Installation
To run the application, there are a couple of dependenices the user must have on the system in order for full functionality. 
The first thing you need is to have Microsoft Visual Studio with C++ Build-tools installed (This is required for the application to interface the camera libraries that are
are needed for running the web camera input. 
Dowload it here: https://visualstudio.microsoft.com/vs/community/
A setup of the the above application is provided along with the application for directly installing.

After you are done setting that up, the next thing is to have a python environment setup on your host device.

There are two ways to apporach the environment setup, the user can either use a python environemnt that is natively setup on the host device 
or they can use a virtual environment using Anaconda.


### The First Way - PIP
Download Python version 3.9 and install it on your host device.
Navigate to the application directory which has the requirements.txt file ( requirements-gpu.txt is the same but use this only if you have an Nvidia GPU setup with CUDA)
Open a command prompt in this folder and run the following command (You can do that by typing "cmd" in the address bar and hitting "Enter"
```
pip install -r requirements.txt
```
If you happen to have an **Nvidia GPU with CUDA** configured on your system, you may use the Tensorflow-gpu to speed up your application
Open a command prompt in this folder and run the following command (You can do that by typing "cmd" in the address bar and hitting "Enter"
```
pip install -r requirements-gpu.txt
```
(On Linux : streamlit run main.py)

Once that is done, open the file named **fish-viewer.vbs** and you should see the application running.


### The Second Way - Conda (Recommended)
Download and install anaconda on your Windows device.

Download Anaconda here: https://www.anaconda.com/
Once you have anaconda installed. You should open anaconda prompt and run the following command which will create a virtual environment for you

```
conda env create -f conda-cpu.yml
```
If you happen to have an **Nvidia GPU with CUDA** configured on your system, you may use
```
conda env create -f conda-gpu.yml
```

After that is done, activate your conda environment
```
conda activate fishnet-cpu
```
and that's it. You can now exit the cmd prompt and go to application folder and **open fish-viewer_conda.vbs** to run the application.


