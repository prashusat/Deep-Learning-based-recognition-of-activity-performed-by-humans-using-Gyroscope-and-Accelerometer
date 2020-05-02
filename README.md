<ul><h1>Aim:</h1></ul><br/>
The aim of the project is to use data collected from mobile gyroscome and accelerometers and build a sequence model using LSTM's to classify the type of activity being performed by a human being within a given interval.<br/>
<br/>
<ul><h1> Description of the dataset: </h1></ul><br/>

This project was performed with the help of data obtained from UC Irvine dataset. 


The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain. See 'features_info.txt' for more details. 

<ul><h1> Labels: </h1></ul><br/>
The labels folder which is y_train.txt and y_test has 6 different labels and they are as follows:<br/>
1 WALKING<br/>
2 WALKING_UPSTAIRS<br/>
3 WALKING_DOWNSTAIRS<br/>
4 SITTING<br/>
5 STANDING<br/>
6 LAYING<br/>

<ul><h1> Conclusions:  </h1></ul><br/>
Scroll down the ipynb file for conclusions.
