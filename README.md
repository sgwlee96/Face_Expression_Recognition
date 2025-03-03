# 😊 Face Expression Recognition
## Introduction

This project focuses on developing and optimizing a Facial Expression Recognition (FER) system tailored for educational environments to assess student engagement and emotional responses during instructional sessions. The system leverages advanced deep learning architectures, particularly You Only Look Once version 7 (YOLOv7), alongside insights from prior research on VGG16, R-CNN, and AlexNet to enhance accuracy and reliability.

## Objective

The primary objective of this project is to improve FER accuracy beyond 90%, addressing challenges related to real-time detection, data imbalance, and multi-label classification. By integrating novel data balancing and augmentation techniques, this research enhances the model’s robustness and ensures reliable engagement assessment in academic settings.

## Technologies & Frameworks Used
- Deep Learning Architectures: YOLOv7, VGG16, R-CNN, AlexNet
- Frameworks & Libraries: PyTorch, TensorFlow, Keras, OpenCV
- Programming Language: Python
- Data Processing: NumPy, Pandas
- Visualization Tools: Matplotlib, Seaborn

## Methodology
1. Data Collection & Preprocessing:
- Collected facial expression datasets for training and testing.
- Applied data augmentation techniques to handle class imbalance.
- Preprocessed images using normalization and noise reduction.

2. Model Training & Optimization:
- Implemented deep learning architectures (YOLOv7, VGG16, R-CNN, AlexNet) with Rectified Linear Unit (ReLU) activation.
- Fine-tuned hyperparameters (learning rate, batch size, epochs) for optimal performance.

3. Performance Evaluation:
- Compared model accuracy with previous research findings.
- Assessed real-time detection performance in classroom-like environments.

## Key Outcomes & Results
- YOLOv7 outperformed VGG16, R-CNN, and AlexNet in real-time facial expression detection.
- Achieved an accuracy of over 90%, surpassing prior studies (VGG16 - 82%, R-CNN - 85%, AlexNet - 80%).
- Proposed data preparation techniques significantly improved model reliability.
- System successfully detects and categorizes student engagement levels during instructional sessions.

## Future Improvements
- Expanding Dataset: Increase dataset size with diverse expressions from multiple demographics.
- Edge Deployment: Optimize model for integration into low-power edge devices (e.g., Raspberry Pi, Jetson Nano).
- Multimodal Learning: Incorporate additional student engagement indicators (e.g., voice, posture analysis).
- Explainability & Interpretability: Implement techniques like Grad-CAM for better model interpretability.

## Conclusion
This Facial Expression Recognition (FER) system enhances educational settings by analyzing student engagement and emotional responses in real time. The integration of YOLOv7 with novel data balancing strategies improves accuracy, efficiency, and practical usability compared to traditional FER models. This research contributes to advancing AI-powered engagement monitoring tools for personalized and adaptive learning environments.


-------------------------------------------------------------------------------------
# Project Structure

- **[src/](./src/)**: source code and data

    - **[cvmodels/](./src/cvmodels/)**
        - **[yolov8/](https://github.com/WongKinYiu/yolov7)**: Yolo version 7. Model is to large, so it is not pushed. You can check it go to the link provided.

        - **[rcnn/](./src/cvmodels/rcnn/)** : R-CNN.

           - **[helper_functions/](./src/cvmodels/rcnn/helper_functions/)**: Function used to prepare data.
                - **[collate.py](./src/cvmodels/rcnn/helper_functions/collate.py)**: Function to collate batch before giving to model as an input.
                - **[helper.py](./src/cvmodels/rcnn/helper_functions/helper.py)** : Functions to get image annotations in the model input format.
                - **[split_roots.py](./src/cvmodels/rcnn/helper_functions/split_roots.py)**: Function to split data into train, test, and validation sets.
                - **[visual.py](./src/cvmodels/rcnn/helper_functions/visual.py)**: Function to plot images with annotations.

           - **[model/](./src/cvmodels/rcnn/model/)**: Files containing functions to train R-CNN.
                - **[__init__.py](./src/cvmodels/rcnn/model/__init__.py)**: Package initializer.
                - **[dataset.py](./src/cvmodels/rcnn/model/dataset.py)** : A custom Dataset object (inherited torch Dataset).
                - **[model_training.py](./src/cvmodels/rcnn/model/model_training.py)**: The main function for model training.
                - **[predict.py](./src/cvmodels/rcnn/model/predict.py)** : Function for prediction.
                - **[train.py](./src/cvmodels/rcnn/model/train.py)** : Functions for model training.
                - **[validate.py](./src/cvmodels/rcnn/model/validate.py)** : Function to evaluate model performance.

           - **[main.py](./src/cvmodels/rcnn/main.py)** : The main function where model is training.

        
    - **[data_engineering/](./src/data_engineering/)**

        - **[utils/](./src/data_engineering/utils/)**: Utility functions for data engineering.

          - **[__init__.py](./src/data_engineering/utils/__init__.py)**: Package initializer.
          - **[concat.py](./src/data_engineering/utils/concat.py)**: Utility functions to concat concat datasets.
          - **[convert.py](./src/data_engineering/utils/convert.py)** : Utility functions to convert data set into YOLO format.
          - **[data_preparation.py](./src/data_engineering/utils/data_preparation.py)** : Utitliy functions to solve balancing problem.
          - **[functions.py](./src/data_engineering/utils/functions.py)**: Utility functions to automate simple tasks.
          - **[resplit.py](./src/data_engineering/utils/resplit.py)** : Utility functions to resplit and combines by labels.
          - **[visual.py](./src/data_engineering/utils/visual.py)**: Utility functions to automate visualizations.
        - **[data_engineering_results.ipynb](./src/data_engineering/data_engineering_results.ipynb)**: Steps of data preparation with visualizations.
        - **[dem_reduction.ipynb](./src/data_engineering/dem_reduction.ipynb)**: Examples of dimensionality reduction.
        - **[eda.ipynb](./src/data_engineering/eda.ipynb)** : EDA.
        - **[main.py](./src/data_engineering/main.py)**: The main file for data engineering.
- **[requirements.txt](/requirements.txt)**: The inclusion of a requirements.txt file makes it easier to recreate the project's environment and install the necessary dependencies.
    
