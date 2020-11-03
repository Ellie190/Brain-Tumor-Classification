# Brain Tumor Classification: Project Overview 
- Created a Convolutional Neural Network model to classify if a patient has Brain Tumor or not from Brain MRI scans. 
- Downloaded the MRI dataset on Kaggle.
- Made use of Transfer learning to compensate for the dataset size and Data Augmentation to allow the model to generalise better. 
- ImageNet dataset and VGG-16 architecture was utilized for transfer learning via fine-tuning.
- Six Statistical metrics was used to evaluate model performance.
- Metrics: Accuracy, Precision, Recall, F1-score, Macro-average and Weighted-average. 
- Built an application to automate the process of Brain Tumor classification. 

## Resources
**Python Version:** 3.8 <br>
**Packages:** Tensorflow, Keras, Sklearn, imutils, matplotlib, numpy, argparse, pickle, cv2, os <br>
**Detecting COVID-19:** https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/ <br>
**Undersatnding CNNs:** https://www.analyticsvidhya.com/blog/2019/05/understanding-visualizing-neural-networks/ <br>

## Brain MRI scan (Gray Image)
Source: [Wikipedia](https://en.wikipedia.org/wiki/Brain_tumor)

## Statistical Evaluation 
![Metric](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/Statistical%20Metrics.png)

## Filter and Feature (Hot Colornmap)
- A hot colormap was applied, a sequential black-red-yellow-white, to emulate blackbody radiation from an object at increasing temperatures
- This was to illustrate features in brain MRI scans learned by the Neural Network
![Hot Filter](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/Hot%20Filter.png) 
![Feature](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/Feature%20(Hot).png)
![Feature2](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/Feature%202%20(Hot).png)

## Classifications from test samples
- The images below got presevered to be used to test model. <br>
- Outputs from the Classification.py <br> 
![Tumor](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/TumorClassfication.png)
![Normal](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/TumorClassfication.png)

## Application to Classify Brain MRI screenshot
![App](https://github.com/Ellie190/Brain-Tumor-Classification/blob/main/Figures/AppScreenshot1.png)

