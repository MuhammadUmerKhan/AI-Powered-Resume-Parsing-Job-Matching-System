# AI-Powered Resume Parser 🤖📝

![Resume Parser](https://media.licdn.com/dms/image/v2/D4D12AQGeVOhx27jdqQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1732253495805?e=2147483647&v=beta&t=BNfCukca2GVw7Vhv6QUe2jxW2kxX_-LDH_TLBSCiozE)

In the fast-paced world of recruitment, manually sifting through resumes can be a tedious and time-consuming task. This project aims to automate the process of parsing resumes, extracting key information, and categorizing them efficiently using AI. By leveraging Natural Language Processing (NLP) and Machine Learning, the Resume Parser ensures that recruiters can focus on the most relevant details, optimizing the hiring process. 🧑‍💻🌟

This repository contains the code, model, and tools necessary for building a state-of-the-art resume parser, capable of extracting personal information, skills, education, experience, and other key elements from resumes in various formats. 📂📝

## 📚 Table of Contents  
- [🔍 Overview](#-overview)  
- [🔧 Project Structure](#-project-structure)  
- [💻 Technologies Used](#-technologies-used)  
- [✔️ Current Work](#-current-work)  
- [🎯 Planned Future Enhancements](#-planned-future-enhancements)  
- [🚀 Getting Started](#-getting-started)  
- [📛 Prerequisites](#-prerequisites)  
- [📚 Acknowledgments](#-acknowledgments)  

---

## 🛡️ Problem Statement 
Recruitment teams often struggle with the overwhelming volume of resumes they receive for job openings. Parsing these resumes manually to extract key information like contact details, skills, and work experience is inefficient and prone to errors. This project uses AI techniques to automatically parse resumes, extract relevant data, and present it in an organized manner for recruiters. 🎯

By using this tool, the time spent on screening resumes can be drastically reduced, enabling hiring managers to focus more on high-priority tasks such as candidate interviews and decision-making. 🕛🔍

---

## 🔧 Methodology 
1. **Data Collection and Preparation:** 📊
   - Collected a dataset consisting of resumes in various formats (PDF, DOCX, TXT).
   - Preprocessed the data by converting these resumes into structured text for further analysis.

2. **Text Preprocessing:** 🧺
   - Performed tasks such as tokenization, stop word removal, stemming, and lemmatization.
   - Cleaned and standardized the text to ensure accurate extraction.

3. **Resume Categorization:** 📂
   - Categorized resumes based on job titles using semantic similarity scoring (SBERT).
   - Developed a job title classification function for automated categorization of resumes into broad job categories.

4. **Model Deployment:** 🚀
   - The model was deployed as a [Streamlit app](https://ai-powered-resume-parser.streamlit.app/), providing an intuitive interface for resume parsing and job matching.

---

## 📊 Data Insights 
Explore profound insights and analytics gained from our extensive dataset. Uncover a deeper understanding of customer behaviors, patterns in service usage, and the pivotal factors influencing hiring decisions.

| Feature                                      | Visualization                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Category Distribution                        | ![Category distribution](https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher/blob/main/imgs/category_distribution.png)   |
| WordCloud Resume                             | ![WordCloud Resume](https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher/blob/main/imgs/wordcloud_resume.png)  |
| WordCloud Job Description                    | ![WordCloud Job Description](https://github.com/MuhammadUmerKhan/-AI-Powered-Resume-Parser---Job-Matcher/blob/main/imgs/wordcloud_job.png)   |

---

## 💻 Technologies Used 
- **Python** 🐍 for text processing and feature extraction
- **Streamlit** 🌐 for web application deployment
- **SBERT** 🤖 for semantic textual similarity (job title matching)
- **spaCy** 📚 for NLP tasks like tokenization, named entity recognition, and lemmatization
- **Pandas** 📊 for data manipulation and analysis
- **Regular Expressions** 🧉 for extracting structured data and removing irrelevant text

---

## ✔️ Current Work 
- **LLM Integration for Job Matching & Resume Screening Chatbot:** 🤖
  - The chatbot analyzes uploaded resumes and job descriptions.
  - Provides real-time feedback on resume improvements.
  - Identifies missing skills and suggests enhancements.

- **Enhancing Resume Matching System:** 🎯
  - Integrated an LLM to extract matching skills from resumes and job descriptions.
  - Suggests improvements by identifying missing skills for a better job match.

---

## 🎯 Planned Future Enhancements 
- **Multilingual Support:** 🌍
  - Enable parsing of resumes in multiple languages.

- **Integration with ATS (Applicant Tracking Systems):** 🔗
  - Allow seamless integration into recruitment workflows.

- **Improved LLM Response Handling:** 🤖
  - Ensure structured and clean JSON output from the LLM.

---

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

## 📛 Prerequisites  
- Python 3.x
- Required packages are listed in requirements.txt.

---  

## 📚 Acknowledgments  
- **Datasets:**  
   - [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
   - [Job Description Dataset](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description).

