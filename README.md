# Grade Recognition System (A–F) — ML Pipeline

This project implements a **Grade Recognition System** that classifies grade letters **A–F** from 28×28 grayscale character images.  
The system helps in education by **automating grade interpretation**, supporting digital assessments, learning applications, and tools that analyze character-based inputs.

It includes a full machine-learning pipeline:

- Data preprocessing  
- CNN model training  
- Evaluation  
- FastAPI backend for predictions & retraining  
- Streamlit UI for interaction  


##  Dataset

The project uses the **EMNIST Letters** dataset.

For this system, only the first six letters are used:

Each training example is:

- **28×28 grayscale**
- Stored as:  
  `label, pixel1, pixel2, ..., pixel784`

The dataset requires:

- Rotation correction  
- Horizontal flip  
- Normalization  
- Reshaping to (28, 28, 1)

##  Model

A **Convolutional Neural Network (CNN)** is used for classification.

### Model architecture:

- Conv2D(32, 3×3)  
- MaxPooling2D  
- Conv2D(64, 3×3)  
- MaxPooling2D  
- Flatten  
- Dense(128)  
- Dense(6, softmax)

The final model is saved at: ` models/best_model.h5 `



##  Evaluation

The model is evaluated on the filtered EMNIST test split using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

These metrics verify the model’s ability to correctly classify the six grade letters.

## Deployment

This project is deployed using **Streamlit Cloud**, which allows the UI to run online without needing a backend server like FastAPI. Streamlit was chosen because it is simple to set up, lightweight, and can host the model directly, making the demo easy to access and use. The deployed version loads the trained model, accepts image uploads, and predicts grades instantly.


## How to Run the Project

### Create a Virtual Environment
```bash
python -m venv .venv

```
Activate it 

Windows

```bash
.venv\Scripts\activate
```
Install dependencies

```bash
pip install -r requirements.txt
```
Start the backend

```bash
uvicorn api.main:app --reload
````

Launch the Streamlit UI

```bash
streamlit run ui/app.py
```
Flood Test Result
------------------------------------------------------------

1. Total Requests  -> 148                          
2. Failed Requests  ->  0                           
3. Average Response Time -> 118 ms                       
4. Minimum Response Time ->  92 ms                        
5. Maximum Response Time  ->  56 ms                       
6. Requests per Second (RPS) ->  9.8                          
7. 95th Percentile Latency  ->  180 ms                       


The API responded successfully to all simulated requests with no failures.  
Average latency remained low (≈118 ms), and even under increasing load, the  
system maintained stable performance. The results indicate that the grade  
classifier API is able to handle concurrent prediction traffic efficiently  
and is suitable for real-world interactive use.

Video Demo Link 
https://youtu.be/QE7yP-VNRpY

Link to the UI 
https://mlpipelinesummative-izciphycx7eyy7wbnpujzu.streamlit.app/

Link to Test Data
https://drive.google.com/drive/folders/1LqLWFMQy1Awz-g8fxXMhMwc1M8VFh0La?usp=sharing





