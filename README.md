Here's the full project documentation with your Streamlit deployment included:

# Disaster Prediction Model using Hugging Face Transformers and NLP Techniques

## Project Overview

This project aims to classify tweets related to different types of disasters using Natural Language Processing (NLP) techniques. The model is built using the Hugging Face Transformers library, specifically leveraging a pre-trained BERT model. This project aligns with **Sustainable Development Goal (SDG) 13: Climate Action** by enabling the classification and analysis of disaster-related content on social media, which can assist in disaster response and management.

## **Table of Contents**

1. [Project Idea](#project-idea)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Setup and Installation](#setup-and-installation)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Model Selection and Loading](#model-selection-and-loading)
7. [Data Preparation](#data-preparation)
8. [Fine-Tuning the Model](#fine-tuning-the-model)
9. [Evaluation](#evaluation)
10. [Prediction Example](#prediction-example)
11. [Results](#results)
12. [Conclusion](#conclusion)
13. [References](#references)
14. [Streamlit Deployment](#streamlit-deployment)

## **Project Idea**

### **I. Project Alignment with SDG 13**

This project addresses SDG 13 by creating a tool to predict and classify disaster-related tweets. The model helps identify the type of disaster, which can aid in disaster response and resource allocation, ultimately supporting climate action and disaster management efforts.

### **II. Project Goal**

The primary goal is to fine-tune a pre-trained BERT model to accurately classify tweets into six disaster categories: Drought, Earthquake, Wildfire, Floods, Hurricanes, and Tornadoes. This classification can be used to improve the response to and management of disasters as they occur.

## **Objectives**

1. **Familiarize with Hugging Face Transformers**: Gain hands-on experience with the Hugging Face library and its pre-trained models for NLP tasks.
2. **Model Selection**: Identify and select an appropriate pre-trained model (BERT) that can be fine-tuned for the task of disaster tweet classification.
3. **Fine-Tuning**: Fine-tune the selected model on a labeled dataset of disaster-related tweets to enhance its performance.
4. **Evaluation**: Rigorously evaluate the model’s performance before and after fine-tuning, focusing on key metrics like accuracy, precision, recall, and F1-score.
5. **Real-World Application**: Demonstrate the model’s effectiveness by predicting disaster types in new, unseen tweets.

## **Dataset**

- **Source**: The dataset consists of tweets related to various types of disasters, sourced from [Kaggle](https://www.kaggle.com/datasets/harshithvarma007/disaster-tweets).
- **Features**:

  - `Tweets`: The text of the tweet.
  - `Disaster`: The type of disaster mentioned in the tweet, categorized into six labels: Drought, Earthquake, Wildfire, Floods, Hurricanes, and Tornadoes.

- **Dataset Mapping**:
  - Drought: 0
  - Earthquake: 1
  - Wildfire: 2
  - Floods: 3
  - Hurricanes: 4
  - Tornadoes: 5

## **Setup and Installation**

### **I. Install the Necessary Libraries**

Ensure that you have the required libraries installed in your Python environment:

```bash
pip install transformers
pip install torch
pip install pandas
pip install scikit-learn
```

### **II. Clone the Repository**

Clone the project repository to your local machine:

```bash
git clone https://github.com/yourusername/disaster-prediction.git
cd disaster-prediction
```

### **III. Download the Dataset**

Make sure the `Disaster.csv` file is present in the project directory. This file contains the tweets and their corresponding disaster labels.

## **Exploratory Data Analysis (EDA)**

Before training the model, it’s important to understand the dataset through Exploratory Data Analysis (EDA):

- **Loading the Dataset**:
  - Start by loading the dataset into a pandas DataFrame.
  - Inspect the dataset structure, including columns, data types, and checking for any missing values.
- **Visualizing Data Distribution**:
  - Analyze the distribution of tweets across different disaster categories to understand the class balance.
  - Visualizations such as bar plots or pie charts can be used to represent this distribution.

## **Model Selection and Loading**

### **I. Model Selection**

- **Model**: We selected the `bert-base-uncased` model from the Hugging Face model hub due to its robustness in handling a wide range of Natural Language Processing (NLP) tasks, including text classification.
- **Tokenizer**: The `AutoTokenizer` associated with BERT is used to tokenize the input tweets, ensuring that they are in a format suitable for the model.

### **II. Load the Pre-trained Model**

Load the pre-trained BERT model and its tokenizer using the following code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
```

This model will be fine-tuned to adapt it specifically to the disaster tweet classification task.

## **Data Preparation**

### **I. Data Sampling**

For demonstration purposes, we sampled 1000 tweets from the dataset. This sample size is manageable for running experiments while still being large enough to train a reliable model.

### **II. Data Splitting**

Split the dataset into training and testing sets:

- **Training Set**: 80% of the data is used to train the model.
- **Test Set**: 20% of the data is set aside for evaluating the model's performance.

### **III. Tokenization**

The tweets are tokenized using the BERT tokenizer, which converts the raw text into tokens that the model can process:

```python
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)
```

Tokenization includes truncating the text to a specific length and padding shorter texts to ensure uniform input size.

## **Fine-Tuning the Model**

### **I. Define Training Arguments**

Fine-tuning is done by setting specific training parameters:

- **Epochs**: 3 (The number of times the model will pass through the entire training dataset)
- **Batch Size**: 16 (The number of samples processed before the model’s parameters are updated)
- **Learning Rate**: 5e-5 (Controls how much to change the model in response to the estimated error each time the model weights are updated)

### **II. Training Loop**

The model is trained using a custom loop, which optimizes the model's parameters to minimize the loss function:

```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Implement the training loop here
```

The `AdamW` optimizer is used as it’s well-suited for fine-tuning transformer models like BERT.

## **Evaluation**

### **I. Model Evaluation**

The model’s performance is evaluated on the test set using various metrics:

- **Accuracy**: The proportion of correctly predicted tweets out of the total number of tweets.
- **Precision**: The proportion of correctly predicted positive observations to the total predicted positives.
- **Recall**: The proportion of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The weighted average of Precision and Recall, providing a balance between the two.

### **II. Performance Comparison**

The model’s performance before and after fine-tuning is compared to assess the effectiveness of the fine-tuning process. This comparison helps to understand how much the model has improved after being adapted specifically to the disaster tweet classification task.

## **Prediction Example**

After fine-tuning, the model can be used to predict the type of disaster mentioned in new, unseen tweets:

```python
new_texts = ["The smoke from the wildfire is affecting air quality in nearby cities."]
new_encodings = tokenizer(new_texts, truncation=True, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**new_encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(predictions.item())  # Outputs the predicted disaster type as an integer
```

This example demonstrates how the model can be applied to real-world scenarios by analyzing new tweets and classifying them into disaster categories.

## **Results**

### **I. Model Performance**

- **Accuracy**: 97.5%
- **Precision**: 97.60%
- **Recall**: 97.5%
- **F1-Score**: 97.49%

The high accuracy and F1-score indicate that the model performs exceptionally well in classifying disaster-related tweets.

### **II. Predictions on Sample Texts**

The model was tested on various disaster-related texts and correctly identified the disaster types, showcasing its potential utility in disaster response and management.

## **Conclusion**

This project successfully demonstrates the application of Hugging Face Transformers to fine-tune a BERT model for disaster tweet classification. The model achieved high accuracy, making it a valuable tool in the context of disaster response. The project also highlights the potential of AI and NLP in contributing to **SDG 13: Climate Action**, providing a practical example of how technology can aid in managing and responding to disasters.

## **References**

- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)

- [Kaggle Dataset: Disaster Tweets](https://www.kaggle.com/datasets/harshithvarma007/disaster-tweets)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## **Streamlit Deployment**

### **I. Application Interface**

The application is deployed using Streamlit, providing an interactive interface where users can input tweets and receive predictions on the type of disaster mentioned in the tweet.

### **II. Running the Application**

To run the Streamlit application, use the following command in your terminal:

```bash
streamlit run app.py
```

This command will start a local server where you can interact with the disaster tweet classification model through a user-friendly interface.
