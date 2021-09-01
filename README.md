# Emotion-Recogntion-using-SVM
- P/S
  - The dataset used in this project are taken from JAFFE databased with 7 types of emotion
  - This programming are modified from https://github.com/gomanajah/Emotion-Estimation-From-Facial-Images (please refer this github for more expression and method)

- What is modified in my work ?
  - The expression is reduce into 5 emotion - Angry, Disgusted, Fear, Happy, Neutral
  - The method used is :
    - LPB as feature extraction:
      - LBP is 
    - SVM as classification
      - SVM is 

- How to used this program : 
  - Set path for the databased in DBPath
  - run the main.m 
  - Test the data invidually to predict the expression shows by the image choose
 
- The output of this program is represent in confusion matrix and testing data
  - Confusian Matrix
<p align="center">
  <img width="80%" height="80%" src= "https://github.com/masyitah-abu/Emotion-Recogntion-using-SVM/blob/main/ConfusionMatrix.png">
</p>
  
  - Predict image
    - For predict image: one image will be choose in the folder and the expression is predict
    - The image choose is:
    <p align="center">
      <img width="80%" height="80%" src= "https://github.com/masyitah-abu/Emotion-Recogntion-using-SVM/blob/main/Predict.png">
    </p> 
    
    - The image will be predict based on the highest score from the 5 expression.
    - From the image Angry exprestion show the highest probability
    <p align="center">
      <img width="60%" height="60%" src= "https://github.com/masyitah-abu/Emotion-Recogntion-using-SVM/blob/main/Predict.png">
    </p>  
 
