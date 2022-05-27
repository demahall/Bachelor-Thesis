#Import Properties 

import cv2
import mediapipe as mp
import numpy as np
import csv
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
import datetime
import math
import time


# global variable
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
keypoints = []

# Read video and extract the coordinate

def readvideo (filename,landmarkname):


    
    #read the video
    cap = cv2.VideoCapture(filename)
    
    #count frames
    counter=0
    
    # extract landmark coordinate with list variable
    landmark=[]
    
    # extract important landmark to define initialposition
    global keypoints
    fillin=[]
    
    # Check if keypoints are already filled in and it will be reset
    if keypoints or None:
        keypoints=[]
    
    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
         while cap.isOpened():
              
                ret, frame = cap.read()
                
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                    

                counter=counter+1
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
        
                # Declare image shape 
                image_height, image_width, _ = image.shape  
                
                # Make detection
                results = pose.process(image)

                #extract landmark
                landmarkname_dict={
                    "R_S":"RIGHT_SHOULDER",
                    "L_S":"LEFT_SHOULDER",
                    "R_H":"RIGHT_HIP",
                    "L_H":"LEFT_HIP",
                    "R_K":"RIGHT_KNEE",
                    "L_K":"LEFT_KNEE",
                    "R_A":"RIGHT_ANKLE",
                    "L_A":"LEFT_ANKLE"

                }

                varlist=[getattr(mp_pose.PoseLandmark,landmarkname_dict["R_S"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["L_S"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["R_H"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["L_H"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["R_K"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["L_K"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["R_A"]),
                         getattr(mp_pose.PoseLandmark,landmarkname_dict["L_A"])
                         ]
  
                for i in varlist:
                    fillin.append([results.pose_landmarks.landmark[i].x*image_width,
                                 results.pose_landmarks.landmark[i].y*image_height])
                
                keypoints.append(fillin)
                fillin=[]
                                
                # extract landmark
                landmark.append([results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark,landmarkname)].x*image_width,
                                                 results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark,landmarkname)].y*image_height])


                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    


    keypoints = np.array(keypoints)
    keypoints = np.transpose(keypoints)
    keypoints = list(keypoints)
    
    # count the number of frames'
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames=int(frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS= ",fps)

    # calculate dusration of the video
    seconds = frames/fps
    video_time = str(datetime.timedelta(seconds=seconds))
    print("video time:", video_time)
    
    #landmark array transposed
    landmark=np.array(landmark)
    landmark=np.transpose(landmark)
    
    #Count size of Image
    print("Höhe der Image: ",image_height)
    print("Breite der Image: ",image_width)
       
    cap.release()
    cv2.destroyAllWindows()
    
    return landmark,fps,seconds

# Rearrange matrix keypoints

def rearrange_matrix():
    global keypoints  
    """
      # X-axis
        "RIGHT_SHOULDER.X":keypoints[0][0],
        "LEFT_SHOULDER.X":keypoints[0][1],
        "RIGHT_HIP.X":keypoints[0][2],
        "LEFT_HIP.X":keypoints[0][3],
        "RIGHT_KNEE.X":keypoints[0][4],
        "LEFT_KNEE.X":keypoints[0][5],
        "RIGHT_ANKLE.X":keypoints[0][6],
        "LEFT_ANKLE.X":keypoints[0][7],
        
        # Y-Axis
        
        "RIGHT_SHOULDER.Y":keypoints[1][0],
        "LEFT_SHOULDER.Y":keypoints[1][1],
        "RIGHT_HIP.Y":keypoints[1][2],
        "LEFT_HIP.Y":keypoints[1][3],
        "RIGHT_KNEE.Y":keypoints[1][4],
        "LEFT_KNEE.Y":keypoints[1][5],
        "RIGHT_ANKLE.Y":keypoints[1][6],
        "LEFT_ANKLE.Y":keypoints[1][7],
    """
    ergebnis_matrix = [[keypoints[0][0],keypoints[1][0]], #RIGHT_SHOULDER
                       [keypoints[0][1],keypoints[1][1]], #LEFT_SHOULDER
                       [keypoints[0][2],keypoints[1][2]], #RIGHT_HIP
                       [keypoints[0][3],keypoints[1][3]], #LEFT_HIP
                       [keypoints[0][4],keypoints[1][4]], #RIGHT_KNEE
                       [keypoints[0][5],keypoints[1][5]], #LEFT_KNEE
                       [keypoints[0][6],keypoints[1][6]], #RIGHT_ANKLE
                       [keypoints[0][7],keypoints[1][7]]  #LEFT_ANKLE
                      ]
    keypoints = ergebnis_matrix
    
# Calculate angle between three points

def calculateAngle(landmark1,landmark2,landmark3):
    
    # Get the required landmarks coordinates
    x1,y1 = landmark1
    x2,y2 = landmark2
    x3,y3 = landmark3
    
    sample = int(fps_standard*zeit_standard)
    # Calculate the angle between the three points
    ang = []
    
    for i in range(sample):
        ang.append(math.degrees(math.atan2(y3[i]-y2[i], x3[i]-x2[i]) - math.atan2(y1[i]-y2[i],
                                                             x1[i]-x2[i])))
    
    for i in range(sample):
        if ang[i] < 0 :
            ang[i] = ang[i]+360
                                                                    
    return ang   

# Calculate Initial Position

def calc_init_position():
    
    """ Hier wird erstmal die Liste von Coordinates aufgelistet, um
    die Code lesbar zu machen. Dazu wird erstmal mit der Abkürzung
    die Elemente von Coordinates definieren
    """
    global keypoints
    
    sample = int(fps_standard*zeit_standard)
    left_knee_winkel = []
    right_knee_winkel=[]
    hip_winkel = []
    
    middle_hip = []
    middle_knee = []
    middle_ankle = []
    
    for i in range(sample):
        middle_hip.append([min(keypoints[2][0][i],keypoints[3][0][i])+
                           np.abs(keypoints[2][0][i]-keypoints[3][0][i])/2,
                           min(keypoints[2][1][i],keypoints[3][1][i])+
                           np.abs(keypoints[2][1][i]-keypoints[3][1][i])/2]
                          )
        middle_knee.append([min(keypoints[4][0][i],keypoints[5][0][i])+
                           np.abs(keypoints[4][0][i]-keypoints[5][0][i])/2,
                           min(keypoints[4][1][i],keypoints[5][1][i])+
                           np.abs(keypoints[4][1][i]-keypoints[5][1][i])/2]
                          )
        middle_ankle.append([min(keypoints[6][0][i],keypoints[7][0][i])+
                           np.abs(keypoints[6][0][i]-keypoints[7][0][i])/2,
                           min(keypoints[6][1][i],keypoints[7][1][i])+
                           np.abs(keypoints[6][1][i]-keypoints[7][1][i])/2]
                          )
    
    middle_hip=np.transpose(middle_hip)
    middle_knee=np.transpose(middle_knee)
    middle_ankle=np.transpose(middle_ankle)
    
    
    left_knee_winkel = calculateAngle(keypoints[3],keypoints[5],keypoints[7])
    right_knee_winkel = calculateAngle(keypoints[2],keypoints[4],keypoints[6])
    hip_winkel = calculateAngle(middle_hip,middle_knee,middle_ankle)
    
    left_knee_winkel = list(map(int,left_knee_winkel))
    right_knee_winkel = list(map(int,right_knee_winkel))
    hip_winkel = list(map(int,hip_winkel))
    
    
    sample_periode = 1/fps_standard
    
    w_tol = range(178,182)
    
    zeitpunkt=[]
    
    for i in range(sample):
        if left_knee_winkel[i] in w_tol and right_knee_winkel[i] in w_tol:
            zeitpunkt.append((i+1)*sample_periode)
            
    return right_knee_winkel

# Cost Matrix

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    # 
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1],cost_mat,N,M)

#Plotting coordinate Landmark

def plottingcoordinate (signal1,signal2):
    
    
    xlist=np.linspace(0,zeit_standard,len(signal1[0]))
    xlist2=np.linspace(0,zeit_sample,len(signal2[0]))
    
    y_landmarks1=signal1
    y_landmarks2=signal2
    
    #plot x,y,z axis from  landmarks
    
    fig, axs = plt.subplots(2)
    subtitle=["Landmark in x-axis","Landmark  in y-axis"]
    for i in range(2):
        axs[i].set_title(subtitle[i])
        axs[i].plot(xlist,y_landmarks1[i],label="Standard")
        axs[i].plot(xlist2,y_landmarks2[i],label="Sample")
        axs[i].set(xlabel="time in s",ylabel="position in cm")
        axs[i].legend(loc='upper left')   

# Distance matrix

def distance_matrix(signal1,signal2):
        N = signal1.shape[0]
        M = signal2.shape[0]
        dist_mat = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                dist_mat[i, j] = abs(signal1[i] - signal2[j])
        return dist_mat

# DTW Plotting
def dtw_plotting(i,signal1,signal2):
    dist_mat=distance_matrix(signal1[i],signal2[i])
    path, cost_mat,N,M = dp(dist_mat)
   

    print("Alignment cost: {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("Normalized alignment cost: {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))

    plt.figure(figsize=(10, 8))
    plt.subplot(121)
    plt.title("Distance matrix")
    plt.xlabel("Signal_sample")
    plt.ylabel("Signal_standard")
    plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    plt.subplot(122)
    plt.title("Cost matrix")
    plt.xlabel("Signal_sample")
    plt.ylabel("Signal_standard")
    plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path)
    plt.plot(y_path, x_path); # i for Coordinate, x_axis=0, y_axis=1

# Mittelwertoptimierung

def mittelwertopt(signal1,signal2):

    for i in range(2):
        signal1[i]=signal1[i]-np.mean(signal1[i])
        signal2[i]=signal2[i]-np.mean(signal2[i])
        
    return signal1,signal2

# Frequency analysis

def fft_analysis(signal1,signal2,i,fps1,fps2):
    
    samplingFrequency = [int(fps1),int(fps2)]
    samplingInterval = []
    for i in range(len(samplingFrequency)):
        samplingInterval.append(1/samplingFrequency[i])
    
    
    fouriersignal1 = np.fft.fft(signal1[i])/len(signal1[i])
    fouriersignal2 = np.fft.fft(signal2[i])/len(signal2[i]) 
    
    fouriersignal1 = fouriersignal1[range(int(len(signal1[i])/2))]
    fouriersignal2 = fouriersignal2[range(int(len(signal2[i])/2))]
    
    fouriersignal1 = np.abs(fouriersignal1)
    fouriersignal2 = np.abs(fouriersignal2)
    
    
    tpcount=[len(signal1[i]),len(signal2[i])]
    values=[]
    timeperiod=[]
    frequencies=[]
    
    for i in range (len(tpcount)):
        values.append(np.arange(int(tpcount[i]/2)))
        timeperiod.append(tpcount[i]/samplingFrequency[i])
        frequencies.append(values[i]/timeperiod[i])
    
    peaks1,_ = find_peaks(fouriersignal1,height = 20)
    peaks2,_ = find_peaks(fouriersignal2,height = 20)

  
    fig,(axs1,axs2)=plt.subplots(2)
    fig.suptitle("FFT Analysis")
    axs1.plot(frequencies[0],fouriersignal1,'tab:blue')
    axs1.plot([frequencies[0][i] for i in peaks1],fouriersignal1[peaks1],"x")
    axs1.set_ylim(0,50)
    axs1.set_xlim(0,10)
    axs1.set(xlabel="Frequency in Hz")
    axs2.plot(frequencies[1],fouriersignal2,'tab:orange')
    axs2.plot([frequencies[1][j] for j in peaks2],fouriersignal2[peaks2],"x")
    axs2.set(xlabel="Frequency in Hz")
    axs2.set_ylim(0,50)
    axs2.set_xlim(0,10)
    
    return str([frequencies[0][i] for i in peaks1]),str([frequencies[1][i] for i in peaks2])

# Read Standard Video
landmark_standard,fps_standard,zeit_standard=readvideo("squat_standard_ohneFPSrecons.mp4","RIGHT_ANKLE")
print("Count frame: ", int(fps_standard*zeit_standard))

# Rearrange Element from keypoints
rearrange_matrix()