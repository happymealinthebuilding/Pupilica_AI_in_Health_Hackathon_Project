# Derm-AI: Skin Lesion Triage System

## Project Overview

Derm-AI is a deep learning-powered application designed to provide preliminary triage for common skin lesions. The system classifies dermoscopic images into one of eight categories using a fine-tuned EfficientNet-V2-M model. By leveraging interpretability techniques like Grad-CAM++ and Score-CAM, the application provides visual heatmaps that highlight the specific regions of an image the AI model focused on to make its prediction. This transparency aims to build user trust and provide a robust, explainable tool for educational and informational purposes.

The project is structured into a modular, professional codebase, separating concerns such as data preparation, model training, and the web application into distinct files. This design ensures the system is easy to maintain, scale, and understand for future development.

---

## Features

* **Deep Learning Classification:** Utilizes a fine-tuned EfficientNet-V2-M model, pre-trained on ImageNet, for high-accuracy skin lesion classification.
* **Explainable AI (XAI):** Implements **Grad-CAM++** and **Score-CAM** to generate visual heatmaps, allowing users to see the model's decision-making process. 
* **Modular Codebase:** The project is organized into logical components (`app.py`, `train.py`, `data_loader.py`, `model.py`, `utils.py`), promoting code reusability and clarity.
* **Streamlit Web Application:** A user-friendly web interface built with Streamlit for easy image upload and real-time analysis.
* **Ethical AI Disclaimer:** A prominent disclaimer is included in the application to ensure users understand that the tool is for informational purposes only and is not a substitute for professional medical advice.

---

## Installation and Setup

#### Prerequisites

* Python 3.8+
* Git (for cloning the repository)
* Access to a GPU is recommended for training but not required for inference.

#### Step-by-step Guide

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Copy from this repo
    cd your-repo-name 
    ```
2.  **Set up the Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Dataset:**
    The project uses the ISIC 2019 dataset. Please download `ISIC_2019_Training_Input.zip` and `ISIC_2019_Training_GroundTruth.csv` from the official [ISIC Archive](https://www.isic-archive.com/#!/topWithMenu/home).

5.  **Prepare the Dataset:**
    Place the downloaded files in a designated directory (e.g., `data/raw`). Then, run the `data_preparation.py` script to organize the images and split them into training, validation, and test sets.
    ```bash
    python data_preparation.py
    ```

6.  **Train the Model:**
    Once the data is prepared, you can begin training the model. The script will save the trained weights to `derm_ai_model.pth`.
    ```bash
    python train.py
    ```

7.  **Run the Streamlit Application:**
    After training, you can launch the interactive web application to classify new images.
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---


## Code Structure

Derm-AI-Skin-Lesion-Triage-System/

├── app.py     # The Streamlit web application.

├── data_preparation.py     # Script for organizing and splitting the raw dataset.

├── data_loader.py     # Defines PyTorch DataLoaders and data transformations.

├── model.py     # Defines the EfficientNet-V2-M model architecture.

├── train.py     # Main script for training and evaluating the model.

├── utils.py     # Contains utility functions, including XAI implementations.

├── requirements.txt     # Python package dependencies.

├── .gitignore     # Files and directories to be ignored by Git.

├── README.md     # Project overview and documentation.

└── derm_ai_model.pth     # The trained model weights (ignored by Git).


---

## Dataset

The Dataset used to train, test and validate the model was from the International Skin Imaging Collaboration (ISIC)  public dataset.
