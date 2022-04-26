# 3D-CNN-Speaker-Verification
Text Independent Speaker Verification using 3D CNN

Similar works:
- STATISTICAL PARAMETRIC SPEECH SYNTHESIS USING GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/pdf/1707.01670.pdf  
- MLSP 2013 Bird Classification Challenge: https://www.kaggle.com/c/mlsp-2013-birds/data  
- Last year abstract link: https://docs.google.com/spreadsheets/d/1Qd2FOkX5KxxL6-l4PjiFezBY-61cNbQ9ra7dmvy-u38/edit#gid=0  


-------------------------------Final Project on Speaker Verification -----------------------------------------------

- Paper Link →  https://arxiv.org/pdf/1705.09422.pdf
- Link of implementation Tensorflow → https://github.com/astorfi/3D-convolutional-speaker-recognition
- Voice activity code → https://github.com/mvansegbroeck/vad
- Data Set Link (Interesting Datasets):
  - King Speech Corpus
  - Yoho
  - Speaker Identification Research(SPIDRE0) NIST 1994
  - CSLU
  - IIT Guwahati(Phase 1 and 2)
  - Grid corpus




### Abstract(Submitted on 14th August, 2017)

#### [Text-Independent Speaker Verification]  

Speaker Verification (SV) refers to verifying the identity of a speaker by using their recorded voice characteristics. The main aim of this project is to learn robust features that can distinguish individual speakers from just their recorded utterances. We would be working on text independent speaker utterances.

#### [Implementation Details ]

Our recognition system will be built upon the following three phases ::- 

a.) Development Phase -  Here we would like to train a 3D convolutional neural network that would capture robust features that are well distinguishable. This would be learnt in a supervised setting in which one-hot encoded speaker id vs speaker utterances will be provided during training. The 3d convolution comes from the fact that our input is three dimensional. We would be stacking MFEC features of multiple utterances of fixed durations from the same user. The input features will be DxHxW where the dimensions H and W are time and frequency of the MFEC. The dimension D represents the fixed number of utterances from the same person. This completes the development phase.

b.) Enrollment Phase -  During the Enrollment Phase each user’s features will be computed. The features obtained from the layer just before the final classification softmax layer encodes the distinguishable features that we are looking for. These feature are then stored as speaker representative features. 

c.) Test Phase -  During testing phase we would compute the features of the test user and then compared it with the already enrolled speakers. The one with the least distance will be the identity of the speaker.

#### [Interesting Data Sets ]

- VoxCeleb 
- King Speech Corpus
- Yoho
- Speaker Identification Research - NIST 1994
- CSLU
- IIT Guwahati (Phase 1 and 2)
- Grid corpus


#### [Reference]

https://arxiv.org/pdf/1705.09422.pdf


Hey for mythology, we can write whatever we have written in the implementation part of abstract submission  

For implementation detail,  
- Downloaded the entire dataset of audios from youtube
- using the given annotation, cut the audios to make training examples 
- took the 40 dimentional feature (MFEC) at every 10ms of audios


#### Results:
Training codebook
- SER : 6.09375 % (error classification)
- EER = 16.875 % ( equal error rate for verification task)  
![image](https://user-images.githubusercontent.com/17162465/165372306-0e2ce1d9-e6ee-4bcf-8099-de97e4c997ce.png)


Confusion matrix for training codebook  

![image](https://user-images.githubusercontent.com/17162465/165372421-935840a7-c1b7-4096-a733-50ba0c287b87.png)


Test codebook
- SER : 22.5  (error classification)
- EER = 27.8125 % ( equal error rate for verification task)  
![image](https://user-images.githubusercontent.com/17162465/165372533-f91c84e6-1ee8-4faf-b26d-f04c25414fa7.png)


Confusion matrix for test codebook  

![image](https://user-images.githubusercontent.com/17162465/165372607-9fa91de8-b1ec-442f-a1df-07722051a315.png)

