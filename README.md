# Project Name
> Outline a brief description of your project.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
### General Information About the Project
- **Objective**: To develop a machine learning model capable of accurately detecting melanoma in medical images using Convolutional Neural Networks (CNNs).
- **Tools and Libraries**: TensorFlow, Keras, Matplotlib, NumPy, and Augmentor.
- **Approach**: The project follows a structured workflow from dataset understanding, preprocessing, and model training to final evaluation.

---

### Background of the Project
- **Motivation**: Melanoma is a severe form of skin cancer, and early detection is critical for effective treatment. Automated detection systems can assist dermatologists in identifying potential cases quickly.
- **Challenges**: Medical image datasets often exhibit class imbalance, overfitting, and variations in lighting, angles, or noise, requiring robust preprocessing and model designs.
- **Goal**: Address challenges through data augmentation, class balancing, and a carefully designed CNN architecture.

---

### Business Problem
- **Problem Statement**: Skin cancer diagnosis is time-intensive and relies heavily on expert interpretation. A scalable and efficient AI system can improve diagnostic rates, especially in resource-limited settings.
- **Impact**: Automating melanoma detection reduces diagnostic time, enhances accuracy, and ensures equitable healthcare access.
- **Key Metrics**: Model accuracy, sensitivity (recall), and class fairness.

---

### Dataset Being Used
- **Source**: A directory-structured dataset containing training and testing images classified by labels.
- **Content**: Images of skin lesions with categories indicating whether a lesion is malignant or benign.
- **Properties**:
  - Images are stored in subdirectories named after class labels.
  - Approximately 80% of data is allocated for training, and 20% is reserved for validation/testing.
  - Images have varied resolutions and need resizing to a consistent shape for model training.


<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
### 1. Model Evolution Shows Clear Improvement

- **Model 1**: Showed significant overfitting with large accuracy gap  
- **Model 2**: Better generalization but lower overall accuracy (~53%)  
- **Model 3**: Best performance with high accuracy (~80-85%) and manageable overfitting  

### 2. Optimal Performance Found in Model 3

- Achieved highest accuracy of all models  
- Maintained good generalization until epoch 20  
- Demonstrated better learning stability  
- Showed effective training with both metrics improving consistently  

### 3. Early Stopping Emerges as Critical

- **Model 1**: Should stop around epoch 10  
- **Model 2**: Could continue training  
- **Model 3**: Optimal stopping point around epoch 20 before overfitting begins  

### 4. Architecture Complexity Alignment

- Model 3's architecture appears best suited for the problem  
- Shows balanced capacity to learn without immediate overfitting  
- Maintains higher validation accuracy while managing the bias-variance tradeoff  
- Suggests right level of model complexity for the task at hand  







<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Technologies Used
- TensorFlow - version 2.18.0
- Keras - version 3.8.0
- Matplotlib - version 3.0
- NumPy - version 1.26.4
- Augmentor - version 0.2.12

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by the Case Study which Upgrad offered
- This project was based on detecting Melanoma


## Contact
Created by [@AnkitSalian] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->