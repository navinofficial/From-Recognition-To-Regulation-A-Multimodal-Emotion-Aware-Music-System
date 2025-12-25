# From Recognition to Regulation: A Multimodal Emotion-Aware Music System

## Project Overview

Emotions significantly influence human cognition, behavior, and overall well-being. Music has long been recognized as an effective medium for emotional expression, support, and regulation. This project presents a Multimodal Emotion-Aware Music Recommendation System that detects a user’s emotional state using facial expressions and voice signals, and then provides personalized music recommendations or emotion regulation strategies.

Unlike unimodal approaches, the proposed system fuses visual and auditory emotion cues, improving robustness and accuracy under real-world conditions. The system not only recommends music that aligns with the user’s current emotional state but also allows users to regulate their mood through curated playlists.

---

## Objectives

- Accurately recognize human emotions using multiple modalities  
- Reduce emotion misclassification through multimodal fusion  
- Provide emotion-based music recommendation  
- Support emotion regulation for mood improvement  
- Personalize recommendations using the user’s own playlist  

---

## System Architecture

The system consists of the following main components:

### Facial Emotion Recognition
- Captures facial expressions via webcam or image input  
- Uses a Swin Transformer model  
- Pretrained on the FER-2013 dataset  
- Fine-tuned on the CK+ dataset  

### Speech Emotion Recognition
- Processes short voice samples  
- Uses a WavLM-based speech emotion recognition model  
- Trained on datasets such as RAVDESS, CREMA-D, TESS, and EMO-DB  

### Multimodal Emotion Fusion
- Combines predictions from facial and speech emotion models  
- Selects the final emotion based on confidence and performance measures  
- Enhances robustness when one modality is unreliable  

### Music Recommendation and Regulation
- Emotion-based music recommendation mode  
- Emotion regulation mode for improving emotional state  
- Songs are extracted from the user’s personal playlist  

---

## Project Structure

From-Recognition-To-Regulation-A-Multimodal-Emotion-Aware-Music-System/
├── design/
├── notebook/
├── static/
├── templates/
├── app.py
├── pr.ipynb
├── requirement.txt
├── .gitignore
└── README.md

---

## Datasets Used

Facial Emotion Recognition:
- FER-2013
- CK+

Speech Emotion Recognition:
- RAVDESS
- CREMA-D
- EMO-DB

Datasets are not included in the repository due to size and licensing constraints.

---

## Models Used

Facial Emotion Recognition:
- Swin Transformer

Speech Emotion Recognition:
- WavLM Base Model

Emotion Fusion:
- Confidence-based multimodal fusion strategy

---

## Installation and Setup

Clone the repository:
git clone https://github.com/Bharathraj2006/From-Recognition-To-Regulation-A-Multimodal-Emotion-Aware-Music-System.git

Install dependencies:
pip install -r requirement.txt

---

## Running the Application

python app.py

Open a browser and navigate to:
http://127.0.0.1:5000

---

## Results and Observations

- Facial emotion recognition performs well under suitable visual conditions  
- Speech emotion recognition provides complementary emotional cues  
- Multimodal emotion fusion reduces misclassification  
- Personalized playlists enhance user engagement  

---

## Output 

### Login Page:
![Login](https://github.com/user-attachments/assets/23eaa155-f8cd-442f-a607-f3ec5cbb0ba9)

### Home Page:
![Home](https://github.com/user-attachments/assets/385e937d-903b-4c04-a2c1-4bf6dbb1581a)

### Input Page:
![input](https://github.com/user-attachments/assets/f2a69797-071b-4742-94c9-f35e476858c7)

### Output Page:
<img width="1886" height="901" alt="Screenshot 2025-12-24 104017" src="https://github.com/user-attachments/assets/122cf453-27c6-44e1-92e2-c48457465647" />

### Recommended Page :
<img width="1792" height="772" alt="Screenshot 2025-12-24 104216" src="https://github.com/user-attachments/assets/73d82d71-db2b-4214-a6fa-c0777a28e6e2" />


## Future Enhancements

- Text sentiment analysis
- Physiological signal integration
- Real-time feedback
- Reinforcement learning
- User studies

---

## License

Academic and research use only.

---

## Author

Bharath Raj
Third Year, Artificial Intelligence and Data Science
