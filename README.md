# Engagement_Detection_OpenFace_Bi-LSTM

# **Introduction & Problem Definition**

With the Covid-19 outbreak, the online working and learning environments became essential in our lives. For this reason, automatic analysis of non-verbal communication becomes crucial in online environments. 

Engagement level is a type of social signal that can be predicted from facial expression and body pose.  To this end, we propose an end-to-end deep learning-based system that detects the engagement level of the subject in an e-learning environment.

The engagement level feedback is important because:

- Make aware students of their performance in classes.
- Will help instructors to detect confusing or unclear parts of the teaching material.

# Dataset

- 78 students, 195 videos, each 5 min.
- Only Engagement labels.
- Label score between 0-1.
- Challenges:
    - Very few samples.
        
        ![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled.png)
        

# Model Design

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%201.png)

First, the input videos with  <!-- $l$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=l">  number of frames are divided into video segments with a window size of <!-- $m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m"> and with <!-- $j$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=j"> overlapping frames where <!-- $1 \leq m \leq l, m \in \mathbb{Z}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=1%20%5Cleq%20m%20%5Cleq%20l%2C%20m%20%5Cin%20%5Cmathbb%7BZ%7D">  and <!-- $0 \leq j \leq m-1, j \in \mathbb{Z}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=0%20%5Cleq%20j%20%5Cleq%20m-1%2C%20j%20%5Cin%20%5Cmathbb%7BZ%7D"> . Second, the video segments are passed to OpenFace and OpenPose tools for frame-level feature extraction. OpenFace and OpenPose generate $n$ and $m$different features respectively for all <!-- $m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m"> frames. The resulting matrices with shapes <!-- $m \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m%20%5Ctimes%20n"> and <!-- $m \times k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m%20%5Ctimes%20k"> are aggregated by using a subset of the following functions, {mean, variance, standard deviation, minimum, maximum, length, mean, and variance of the absolute Fourier transform spectrum and top 3 Fourier coefficients of the one-dimensional discrete Fourier Transform} and an aggregation frame size of <!-- $z$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=z"> where <!-- $z\leq m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=z%5Cleq%20m">. The aggregation process generates matrices with new shapes such that <!-- $a \leq m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a%20%5Cleq%20m">, <!-- $b \geq n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=b%20%5Cgeq%20n"> and <!-- $c\geq k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=c%5Cgeq%20k">. Third, the aggregation matrices are fed into Bidirectional LSTM and Bidirectional GRU units for sequence modeling. Finally, a fully connected network is used for regression and classification tasks and the predictions are fused with weighted averaging for regression and majority voting for classification. The equations from shows the same flow in functional form.

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%202.png)

## OpenFace Features

- **The eye gaze related features**:
    - gaze_0_x, gaze_0_y,gaze_0_z which are eye gaze direction vectors in world coordinates for the left eye.
    - gaze_1_x, gaze_1_y, gaze_1_zfor right eye.
- **The head pose and rotation related features:**
    - pose_Tx, pose_Ty, pose_Tz representing the location of the head with respect to the camera in millimeters.
    - pose_Rx, pose_Ry, pose_Rzindicates the rotation of the head in radians around x,y,z axes.
- **Facial Action Unit Intensities:**
    - AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r,AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r.
        
        ![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%203.png)
        
    
    ## **Feature Aggregation & Bi-LSTM**
    

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%204.png)

# Experiments & Results

- LSTM parameters:
    - number of hidden units: 512
    - number of layers: 2
- MLP parameters:
    - num neurons 1st layer: 128
    - num neurons 2nd layer: 32
    - num neurons 3rd layer: 4
- Training Parameters:
    - Batch size: 8
    - Learning rate: 0.0005
    - Number of epochs: 350
    - Dropout probability: 0.2
        
        ![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%205.png)
        
        ![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%206.png)
        

# **Feature Importance with Integrated gradients**

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%207.png)

# Real-Life Performance

## Very High Engagement

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%208.png)

## High Engagement

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%209.png)

## Low Engagement

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%2010.png)

## Very Low Engagement

![Untitled](Engagement_Detection_OpenFace_Bi-LSTM%2017cfcb2170c14f12b810380dd47e05cc/Untitled%2011.png)

# Conclusion

- We proposed an end-to-end deep learning-based system that detects the engagement level of the subject.
- The model is able to distinguish between different levels of engagement.
- Some possible directions to extend and improve this work;
    - A training procedure that will use both datasets.
    - Learnable aggregation functions.
    - A self-supervised based method to avoid the reliability problem of the labels.