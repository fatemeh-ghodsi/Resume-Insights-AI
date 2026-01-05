# Smart_Resume_Categorizer

This project implements a complete Natural Language Processing (NLP) pipeline to solve the high-volume task of resume screening. Using Scikit-Learn, it cleans unstructured text from resumes and classifies them into specific professional categories (e.g., Data Science, Java Developer, HR) with high precision.

✨ Key Features
Custom Regex Cleaning: A robust preprocessing function that strips URLs, hashtags, mentions, special characters, and non-ASCII symbols.

TF-IDF Vectorization: Converts cleaned text into weighted numerical features, capturing the importance of industry-specific keywords.

Automated Pipeline: Uses a Scikit-Learn Pipeline to bundle the vectorizer and the classifier together for seamless deployment.

Multi-Class Prediction: Capable of distinguishing between dozens of job categories using a Multinomial Naive Bayes architecture.

Visual Analytics: Generates Pie Charts and Count Plots to analyze dataset balance and a WordCloud to identify dominant skills.

🛠️ Technical Stack
Languages: Python

Data Science: Pandas, NumPy

Machine Learning: Scikit-Learn (TfidfVectorizer, MultinomialNB, Pipeline)

NLP: Regex, WordCloud, NLTK/STOPWORDS

Visualization: Matplotlib, Seaborn

📊 Project Workflow
1. Preprocessing & EDA
The raw resume text is cleaned of "noise" (Twitter handles, RTs, URLs). A distribution analysis is performed to visualize the frequency of different job categories.

2. Feature Extraction
The project uses Term Frequency-Inverse Document Frequency (TF-IDF) to transform text into a format the model can understand, ensuring that common words are down-weighted while unique technical skills are highlighted.

3. Model Training & Evaluation
The data is split into 80% training and 20% testing sets.

Model: Multinomial Naive Bayes (within a Pipeline).

Metrics: Accuracy and a full Classification Report (Precision, Recall, F1-Score).

🚀 Installation & Usage
1. Clone the repository
Bash

git clone https://github.com/your-username/Resume-Categorizer-AI.git
cd Resume-Categorizer-AI
2. Install Dependencies
Bash

pip install pandas scikit-learn matplotlib seaborn wordcloud
3. Predict a Category
Python

sample = ["Experienced Java developer with Spring Boot and Microservices expertise."]
prediction = pipeline.predict(sample)
print(f"Predicted Category: {prediction[0]}")
📂 Repository Structure
data/: Contains the resume dataset (CSV).

Resume_Scanner.ipynb: The main notebook containing the cleaning logic and ML model.

requirements.txt: List of necessary Python libraries.

README.md: Project documentation.

💡 Portfolio Highlight
This project demonstrates the ability to take unstructured text and turn it into structured business value. It showcases proficiency in the entire machine learning lifecycle, from regex-based cleaning to model evaluation.
