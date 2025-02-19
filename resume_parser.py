import PyPDF2
import docx
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import spacy
import re
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
groq_api_key = os.getenv('GROK_CLOUD_API_KEY_LLAMA')

# Preprocessing function
def preprocessor(text):
    with nlp.disable_pipes('ner', 'parser', 'textcat'):
        text = re.sub(r'http\S+', ' ', text)  # Remove URLs
        text = re.sub(r'RT|cc', ' ', text)  # Remove RT and cc
        text = re.sub(r'#\S+', ' ', text)  # Remove hashtags
        text = re.sub(r'@\S+', ' ', text)  # Remove mentions
        text = re.sub(r'[^\x00-\x7f]', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces

        doc = nlp(text.lower())

        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

    return ' '.join(tokens)

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text
        return text
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        return None

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted the figsize to medium size
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def compute_similarity(resume_embeddings, job_embeddings):
    similarity_scores = cosine_similarity(resume_embeddings, job_embeddings)
    return np.array(similarity_scores)


sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit page configuration
st.set_page_config(
    page_title="AI Powered Resume Parser",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            margin-top: -10px;
        }
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #808080;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1.8em;
            color: #808080;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        .stTab {
            font-size: 1.4em;
            font-weight: bold;
            color: #2980B9;
        }
        .section-content{
            text-align: center;
        }
        .intro-title {
            font-size: 2.5em;
            color: #00ce39;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #017721;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            font-weight: bold;
        }
        .separator {
            height: 2px;
            background-color: #BDC3C7;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .prediction-text-good {
            font-size: 2em;
            font-weight: bold;
            color: #2980B9;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .prediction-text-bad {
            font-size: 2em;
            font-weight: bold;
            color: #2980B9;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="main-title">💼 AI Powered Resume Parser 💼</div>', unsafe_allow_html=True)

# Tab layout
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📝 Resume Parsing", "Strengthen Your Resume"])


# Tab content
with tab1:
    # About Me
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring AI/Data Scientist passionate about 
            <span class="highlight">🤖 Natural Language Processing (NLP)</span> and 🧠 Machine Learning. 
            Currently pursuing my Bachelor’s in Computer Science, I bring hands-on experience in developing 
            intelligent recommendation systems, performing data analysis, and building machine learning models. 🚀
        </div>
    """, unsafe_allow_html=True)

    # Project Overview
    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project focuses on developing an <span class="highlight">AI-powered resume parser</span> that matches resumes 
            to job descriptions based on <span class="highlight">semantic similarity</span>. 
            The system extracts key information from resumes (such as contact details, skills, experience) 
            and compares them to job descriptions to assess the compatibility between the two. 
        </div>
    """, unsafe_allow_html=True)

    # Dataset Information
    st.markdown('<div class="section-title">📊 Dataset Information</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">Resume Data:</span> The <a href="https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset">dataset</a> consists of various resumes, collected from Kaggle.</li>
                <li><span class="highlight">Job Description Data:</span> The <a href="https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description">dataset</a> is sourced from <span class="highlight">public job postings and resume repositories</span>, providing a diverse collection of job descriptions.</li>
                <li>This datasets is crucial for training the model to understand the semantics of job titles, experience, and skills.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)



    # Steps Performed
    st.markdown('<div class="section-title">🔬 Steps Performed</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li>🚀 <b>Data Preprocessing:</b> Cleaned textual data (removed special characters, stopwords, etc.), and applied <b>lemmatization</b> to reduce words to their base forms.</li>
                <li>📌 <b>Feature Extraction:</b> Used <b>SBERT embeddings</b> for extracting semantic information from resumes and job descriptions, allowing the model to understand context and intent.</li>
                <li>📑 <b>Job Categorization:</b> Implemented an <b>NLP-based job title classification system</b> to categorize resumes into broad job categories, improving the matching accuracy.</li>
                <li>📈 <b>Semantic Matching:</b> Measured compatibility between the resume and job description using <b>SBERT embeddings & Cosine Similarity</b>, providing an accurate measure of similarity.</li>
                <li>🌐 <b>Interactive UI:</b> Built an intuitive <b>Streamlit</b> application to allow users to upload resumes and job descriptions and instantly see compatibility scores.</li>
                <li>🚀 <b>Deployment:</b> Designed and deployed a real-time resume analysis tool with <b>Streamlit</b> for easy access and use.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Technologies & Tools
    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries:</span> Python, NumPy, Pandas, Matplotlib, Streamlit, TensorFlow, SBERT (Sentence-BERT).</li>
                <li><span class="highlight">⚙️ Approaches:</span> Natural Language Processing (NLP), Text Preprocessing, Semantic Textual Similarity, and Cosine Similarity for comparing embeddings.</li>
                <li><span class="highlight">📊 Machine Learning Models:</span> SBERT (Sentence-BERT) for extracting high-quality, dense text embeddings; Cosine Similarity for calculating semantic matching.</li>
                <li><span class="highlight">🌐 Deployment:</span> Streamlit for building an interactive, user-friendly web-based system for resume parsing and matching.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Data Visualizations (WordClouds, Category Distribution)
    st.markdown('<div class="section-title">📊 Data Insights & Visualizations</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Here are some insights from the dataset that provide a better understanding of the data and its distribution:
            <ul>
                <li>💡 <b>Word Cloud of Resume Content:</b> Shows the most frequent words in the resumes that help identify common skills and experiences.</li>
                <li>💼 <b>Category Distribution:</b> Illustrates how resumes are categorized into various job titles, helping improve the classifier's performance.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
# Second Tab: Resume Parsing
with tab2:
    st.markdown('<div class="section-title">📝 Upload and Parse Resume</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Upload a resume in .txt, .pdf, or .docx format, and the system will parse it to extract key details, 
            match it against a job description, and provide a compatibility score.
        </div><br/>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📄 Upload Resume:", type=["txt", "pdf", "docx"])
    job_description = st.text_area("📄 Enter Job Description:", height=200)

    # Similarity Score Button
    if st.button("🔍 Get ATS Score"):
        if uploaded_file and job_description:
            resume_content = extract_text_from_file(uploaded_file)
            if resume_content:
                preprocessed_resume = preprocessor(resume_content)
                preprocessed_job_description = preprocessor(job_description)

                resume_embedding = sbert_model.encode([preprocessed_resume])
                job_embedding = sbert_model.encode([preprocessed_job_description])

                similarity_score = compute_similarity(resume_embedding, job_embedding)[0][0]

                # Define color for numeric score
                if similarity_score >= 0.7:
                    score_color = "#00ce39"  # Green color for good score
                    score_text = "ATS Score: 👍"
                else:
                    score_color = "#E74C3C"  # Red color for bad score
                    score_text = "ATS Score: 👎"

                st.markdown(f'''
                    <div style="font-size: 2em; font-weight: bold; color: #808080; text-align: center; display: inline-block;">
                        {score_text}
                    </div>
                    <div style="font-size: 2em; font-weight: bold; color: {score_color}; text-align: center; display: inline-block; padding-left: 10px;">
                        {similarity_score:.2f}
                    </div>
                ''', unsafe_allow_html=True)


            else:
                st.error("Could not extract text from the uploaded file.")
        else:
            if not uploaded_file:
                st.warning("⚠️ Please upload a resume file before proceeding.")
            if not job_description:
                st.warning("⚠️ Please enter a job description before proceeding.")
with tab3:
    llm = ChatGroq(
    temperature=0, 
    groq_api_key=groq_api_key, 
    model_name="llama-3.3-70b-versatile"
    )
    prompt_extract = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_desc}

        ### RESUME:
        {resume}

        ### TASK:
        Extract skills from the job description that are NOT mentioned in the resume.
        Return them as a JSON list with no extra text.

        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
    )
    prompt_improvement = PromptTemplate.from_template(
        """
        ### MISSING SKILLS:
        {missing_skills}
        ### INSTRUCTIONS:
        Provide actionable bullet points on how to gain these missing skills, including online courses, books, or hands-on experience.
        ### BULLET POINTS:
        """
    )
    resume_file = st.file_uploader("📄 Upload Resume Here:", type=["txt", "pdf", "docx"])
    job_description = st.text_area("📄 Enter Job Description Here:", height=200)
    
    if st.button("🔍 Improve My Resume"):
        if resume_file and job_description:
            resume_content = extract_text_from_file(resume_file)
            if resume_content:
                preprocessed_resume = preprocessor(resume_content)
                preprocessed_job_description = preprocessor(job_description)

                # Extract missing skills
                chain_extract = prompt_extract | llm 
                res = chain_extract.invoke(input={'job_desc': preprocessed_job_description, "resume": preprocessed_resume})
                
                try:
                    # Extract the content from AIMessage and parse it as JSON
                    response_content = res.content  # Extracting content from AIMessage
                    missing_skills_data = json.loads(response_content)  # Convert JSON string to Python dictionary

                    if missing_skills_data:
                        # Convert to DataFrame and display
                        df = pd.DataFrame({"Missing Skills": missing_skills_data})
                        # st.subheader("🚀 Missing Skills")
                        # st.dataframe(df)
                        
                        chain_improvement = prompt_improvement | llm
                        improvement_res = chain_improvement.invoke(input={"missing_skills": ", ".join(missing_skills_data)})
                        
                        st.subheader("💡 Missig Skills & How to Improve These Skills")
                        response_content = improvement_res.content
                        st.write(response_content)
                    else:
                        st.success("✅ Your resume matches all required skills!")

                except json.JSONDecodeError:
                    st.error("⚠️ Unexpected response format. Please try again.")
        
st.markdown('<div class="footer">© 2025 AI Powered Resume Parser. All rights reserved.</div>', unsafe_allow_html=True)