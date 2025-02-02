# AI-Powered Resume Parser 🤖📄

![Resume Parser](https://media.licdn.com/dms/image/v2/D4D12AQGeVOhx27jdqQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1732253495805?e=2147483647&v=beta&t=BNfCukca2GVw7Vhv6QUe2jxW2kxX_-LDH_TLBSCiozE)

In the fast-paced world of recruitment, manually sifting through resumes can be a tedious and time-consuming task. This project aims to automate the process of parsing resumes, extracting key information, and categorizing them efficiently using AI. By leveraging Natural Language Processing (NLP) and Machine Learning, the Resume Parser ensures that recruiters can focus on the most relevant details, optimizing the hiring process. 🧑‍💻💼

This repository contains the code, model, and tools necessary for building a state-of-the-art resume parser, capable of extracting personal information, skills, education, experience, and other key elements from resumes in various formats. 📂📝


## 📚 Table of Contents  
- [🔍 Overview](#-overview)  
- [🔧 Project Structure](#-project-structure)  
- [💻 Technologies Used](#-technologies-used)  
- [✔️ Current Work](#-current-work)  
- [🎯 Planned Future Enhancements](#-planned-future-enhancements)  
- [🚀 Getting Started](#-getting-started)  
- [🔄 Prerequisites](#-prerequisites)  
- [📚 Acknowledgments](#-acknowledgments)  

---

## Problem Statement 🚨

Recruitment teams often struggle with the overwhelming volume of resumes they receive for job openings. Parsing these resumes manually to extract key information like contact details, skills, and work experience is inefficient and prone to errors. This project uses AI techniques to automatically parse resumes, extract relevant data, and present it in an organized manner for recruiters. 🎯

By using this tool, the time spent on screening resumes can be drastically reduced, enabling hiring managers to focus more on high-priority tasks such as candidate interviews and decision-making. 🕒🔍

---

## Methodology 🔧

1. **Data Collection and Preparation:** 📊
   - Collected a dataset consisting of resumes in various formats (PDF, DOCX, TXT).
   - Preprocessed the data by converting these resumes into structured text for further analysis.

2. **Text Preprocessing:** 🧹
   - Performed tasks such as tokenization, stop word removal, stemming, and lemmatization.
   - Cleaned and standardized the text to ensure accurate extraction.

3. **Resume Categorization:** 🗂️
   - Categorized resumes based on job titles using semantic similarity scoring (SBERT).
   - Developed a job title classification function for automated categorization of resumes into broad job categories.

4. **Model Deployment:** 🚀
   - The model was deployed as a [Streamlit app](https://ai-powered-resume-parser.streamlit.app/), providing an intuitive interface for resume parsing and job matching.

---

## Data Insights 📊

Explore profound insights and analytics gained from our extensive dataset. Uncover a deeper understanding of customer behaviors, patterns in service usage, and the pivotal factors influencing churn dynamics.

| Feature                                      | Visualization                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Category Distribution                        | ![Category distribution](https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher/blob/main/imgs/category_distribution.png)   |
| WordCloud Resume                             | ![WordCloud Resume](https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher/blob/main/imgs/wordcloud_resume.png)  |
| WordCloud Job Description                    | ![WordCloud Job Description](https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher/blob/main/imgs/wordcloud_job.png)   |

---

## Technologies Used 🧑‍💻

- **Python** 🐍 for text processing and feature extraction
- **Streamlit** 🌐 for web application deployment
- **SBERT** 🤖 for semantic textual similarity (job title matching)
- **spaCy** 📚 for NLP tasks like tokenization, named entity recognition, and lemmatization
- **Pandas** 📊 for data manipulation and analysis
- **Regular Expressions** 🧩 for extracting structured data and removing irrelevant text

---

## Current Work 🛠️

Currently, the AI-powered Resume Parser is designed to:
- Parse resumes in multiple formats (PDF, DOCX, TXT) and extract key features like name, contact details, skills, and experience. 📝
- Match resumes to job descriptions by categorizing resumes based on job titles using SBERT-based semantic similarity. 🧠
- Provide a web interface using Streamlit for easy interaction. 🌐

The project has been successfully deployed as a **Streamlit app**, making it easy for recruiters to upload resumes and obtain relevant extracted information. The model is fine-tuned to handle different resume formats and provides job title matching capabilities.

---

## Planned Future Enhancements 🔮

- **Multilingual Support:** 🌍  
  Enable the parser to handle resumes in multiple languages to cater to a broader range of users.

- **Advanced Resume Scoring:** 🏆  
  Implement a scoring system that evaluates the relevance of a resume to a job description, helping recruiters prioritize candidates more effectively.

- **Expanded Feature Extraction:** 🧩  
  Extract more detailed features from resumes, such as achievements, certifications, and hobbies, to provide a more comprehensive analysis.

- **Integration with ATS (Applicant Tracking Systems):** 🔗  
  Integrate the resume parser with popular ATS platforms to allow for seamless resume processing in recruitment pipelines.

- **Model Improvements:** 📈  
  Continuously improve the model’s accuracy by training it on more diverse datasets, including resumes from different industries and regions.

---

## Usage Instructions 🏁

## 🚀 Getting Started  

To set up this project locally:  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher.git
   ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run resume_parser.py
    ```

---  

## 🔄 Prerequisites  
- Python 3.x
- Required packages are listed in requirements.txt.

---  

## 📚 Acknowledgments  

- **Datasets:**  
   - [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
   - [Job Description Dataset](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description).
