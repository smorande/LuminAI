import streamlit as st
from openai import OpenAI
import os
import io
import base64
from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader as DocxLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from textblob import TextBlob
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from docx import Document
from pptx import Presentation
import csv
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize OpenAI clients
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

openai_client = OpenAI(
    api_key="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"
)

# Initialize Claude client
claude_api_key = "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"

# Performance Metrics Class
class PerformanceMetrics:
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def calculate_perplexity(self, text):
        n = 3
        chars = list(text.lower())
        if len(chars) < n:
            return 0
            
        ngrams = [''.join(chars[i:i+n]) for i in range(len(chars)-n+1)]
        unique_ngrams = set(ngrams)
        ngram_counts = {ngram: ngrams.count(ngram) for ngram in unique_ngrams}
        
        log_prob = sum(np.log(ngram_counts[ngram]/len(ngrams)) for ngram in ngrams)
        perplexity = np.exp(-log_prob/len(ngrams))
        return min(perplexity, 1000)
    
    def calculate_bleu(self, reference, candidate):
        if not reference or not candidate:
            return 0
        
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        if not reference_tokens or not candidate_tokens:
            return 0
            
        return sentence_bleu([reference_tokens], candidate_tokens, 
                           smoothing_function=self.smoothing.method1)

# LLM Engine Class
class LuminaAI:
    def __init__(self):
        self.name = "Lumina"
        self.model = "nvidia/llama-3.1-nemotron-70b-instruct"
        self.conversation_history = []
        self.document_store = None
        self.claude_api_key = claude_api_key
        self.metrics = PerformanceMetrics()
        self.current_metrics = {
            'perplexity': 0,
            'bleu': 0
        }

    def update_metrics(self, response, reference=None):
        self.current_metrics['perplexity'] = self.metrics.calculate_perplexity(response)
        
        if reference:
            self.current_metrics['bleu'] = self.metrics.calculate_bleu(reference, response)

    def generate_response(self, user_input):
        previous_response = self.conversation_history[-1]['content'] if self.conversation_history else ""
        
        messages = [
            *self.conversation_history,
            {"role": "user", "content": user_input}
        ]

        if self.document_store:
            relevant_docs = self.document_store.similarity_search(user_input, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            messages.append({"role": "system", "content": f"Relevant context:\n{context}"})

        completion = nvidia_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=True
        )

        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Update metrics after response generation
        self.update_metrics(full_response, previous_response)

    def generate_image(self, prompt):
        url = "https://api.anthropic.com/v1/images"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.claude_api_key
        }
        data = {
            "model": "claude-3-opus-20240229",
            "prompt": prompt,
            "num_images": 1,
            "size": "1024x1024"
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['images'][0]['url']
        else:
            st.error(f"Error generating image: {response.text}")
            return None

    def set_document_store(self, document_store):
        self.document_store = document_store

    def clear_conversation(self):
        self.conversation_history = []
        self.current_metrics = {
            'perplexity': 0,
            'bleu': 0
        }

def detect_download_request(prompt):
    messages = [
        {"role": "system", "content": "You are an AI assistant that determines if a user is requesting to download content in a specific format. The possible formats are: Image, MS-Word, CSV, MS-PowerPoint, Text, and Python. Respond with a JSON object containing boolean values for each format."},
        {"role": "user", "content": prompt}
    ]
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=150
    )
    
    return json.loads(response.choices[0].message.content)

def process_document(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    temp_file_path = f"temp{file_extension}"
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.getvalue())
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == '.docx':
        loader = DocxLoader(temp_file_path)
    elif file_extension in ['.txt', '.py']:
        loader = TextLoader(temp_file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        image = Image.open(temp_file_path)
        text = pytesseract.image_to_string(image)
        os.remove(temp_file_path)
        return [Document(page_content=text, metadata={"source": file.name})]
    else:
        os.remove(temp_file_path)
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    os.remove(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return texts

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

def update_document_store():
    if st.session_state.get('processed_documents'):
        all_texts = [text for doc in st.session_state.processed_documents for text in doc['texts']]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text.page_content for text in all_texts])
        
        embeddings = tfidf_matrix.toarray()
        
        def embed_func(text):
            return vectorizer.transform([text]).toarray()[0]
        
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip([text.page_content for text in all_texts], embeddings)),
            embedding=embed_func,
            metadatas=[text.metadata for text in all_texts]
        )
        
        st.session_state.lumina.set_document_store(faiss_index)
        st.sidebar.success("Document store updated with new documents.")

def download_as_docx(content):
    doc = Document()
    doc.add_paragraph(content)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def download_as_pptx(content):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    txBox = slide.shapes.add_textbox(10, 10, prs.slide_width-20, prs.slide_height-20)
    tf = txBox.text_frame
    tf.text = content
    bio = io.BytesIO()
    prs.save(bio)
    return bio.getvalue()

def download_as_csv(content):
    bio = io.StringIO()
    writer = csv.writer(bio)
    writer.writerow([content])
    return bio.getvalue()

def needs_image_generation(prompt):
    image_keywords = ['draw', 'create', 'generate', 'make', 'produce', 'design']
    image_subjects = ['image', 'picture', 'diagram', 'figure', 'schematic', 'illustration', 'visual']
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in image_keywords) and any(subject in prompt_lower for subject in image_subjects)

def display_performance_metrics(lumina):
    st.sidebar.subheader("🎯 Performance Metrics")
    
    # Perplexity
    perplexity = lumina.current_metrics['perplexity']
    st.sidebar.metric(
        "Perplexity",
        f"{perplexity:.2f}",
        help="Lower perplexity indicates more confident and fluent responses"
    )
    
    # BLEU Score
    bleu = lumina.current_metrics['bleu']
    st.sidebar.metric(
        "BLEU Score",
        f"{bleu:.2f}",
        help="Higher BLEU score indicates better translation quality"
    )

# Streamlit UI Setup
st.set_page_config(page_title="LuminaAI", page_icon="🌟", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        color: #E0E0E0;
        background-color: #1E1E1E;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        color: #00BFFF;
        border-color: #00BFFF;
        border-radius: 20px;
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput>div>div>input {
        color: #E0E0E0;
        background-color: #2E2E2E;
        border-radius: 20px;
        font-family: 'Roboto', sans-serif;
    }
    .stMarkdown {
        color: #B0B0B0;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }
    p {
        font-family: 'Roboto', sans-serif;
        font-weight: 300;
    }
    @media (max-width: 768px) {
        .stButton>button {
            font-size: 14px;
            padding: 10px;
        }
        .stTextInput>div>div>input {
            font-size: 14px;
        }
    }
    .footer-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .footer-text {
        flex-grow: 1;
    }
    .clear-button {
        width: auto !important;
    }
    .sentiment-highlight {
        background-color: rgba(255, 255, 0, 0.3);
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌟 LuminaAI - Advanced Cognitive Interface")

# Initialize LuminaAI and session state
if 'lumina' not in st.session_state:
    st.session_state.lumina = LuminaAI()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'download_options' not in st.session_state:
    st.session_state.download_options = {}

lumina = st.session_state.lumina

# Sidebar
st.sidebar.title("LuminaAI Control Panel")

# LuminaAI Specs
st.sidebar.subheader("🧠 LuminaAI Specs")
st.sidebar.markdown(f"**Model:** {lumina.model}")
st.sidebar.markdown("**Context Window:** 300,000 tokens")

# Add Performance Metrics Display
if 'lumina' in st.session_state:
    display_performance_metrics(st.session_state.lumina)

# Document Upload in Sidebar
st.sidebar.subheader("📁 Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload documents for context (max 25)", type=['pdf', 'docx', 'txt', 'py', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 25:
        st.sidebar.warning("Maximum 25 files allowed. Only the first 25 will be processed.")
        uploaded_files = uploaded_files[:25]
    
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []

    # Process new documents
    new_docs = [file for file in uploaded_files if file not in [doc['file'] for doc in st.session_state.processed_documents]]
    
    if new_docs:
        with st.spinner("Processing new documents..."):
            for file in new_docs:
                texts = process_document(file)
                st.session_state.processed_documents.append({'file': file, 'texts': texts})
        
        st.sidebar.success(f"{len(new_docs)} new document(s) processed and added to the knowledge base.")
        
        # Update document store after processing
        update_document_store()
    else:
        st.sidebar.info("No new documents to process.")

# Display processed documents
if st.session_state.get('processed_documents'):
    st.sidebar.subheader("📚 Processed Documents")
    for doc in st.session_state.processed_documents:
        st.sidebar.markdown(f"- {doc['file'].name}")

# Conversation Analytics in Sidebar
st.sidebar.subheader("📊 Conversation Analytics")
if st.session_state.messages:
    sentiments = [analyze_sentiment(msg["content"]) for msg in st.session_state.messages]
    
    # Sentiment Trend
    fig_trend = go.Figure(data=go.Scatter(
        x=list(range(len(sentiments))),
        y=sentiments,
        mode='lines+markers',
        marker=dict(
            size=8,
            color=sentiments,
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    fig_trend.update_layout(
        title='Sentiment Trend',
        xaxis_title='Message',
        yaxis_title='Sentiment',
        height=300
    )
    st.sidebar.plotly_chart(fig_trend, use_container_width=True)
else:
    st.sidebar.info("Start a conversation to see analytics.")

# Main content area
st.subheader("💬 Interaction Panel")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image_url" in message:
            st.image(message["image_url"], caption="Generated Image")

# Text input
if prompt := st.chat_input("What would you like to know or create?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Check if image generation is needed
        if needs_image_generation(prompt):
            with st.spinner("Generating image..."):
                image_url = lumina.generate_image(prompt)
            if image_url:
                st.image(image_url, caption="Generated Image")
                full_response = f"I've generated an image based on your request. Here it is!"
                st.session_state.messages.append({"role": "assistant", "content": full_response, "image_url": image_url})
            else:
                full_response = "I'm sorry, I couldn't generate the image. Please try again with a different prompt."
        else:
            for chunk in lumina.generate_response(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    if not needs_image_generation(prompt):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Sentiment analysis with soft yellow highlight
    sentiment = analyze_sentiment(full_response)
    st.markdown(f'<span class="sentiment-highlight">Response sentiment: {sentiment:.2f}</span>', unsafe_allow_html=True)

    # Detect download requests using GPT-4
    download_options = detect_download_request(prompt)
    st.session_state.download_options = download_options

    # Display download options if requested
    if any(download_options.values()):
        st.subheader("Download Options")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        if download_options.get("Text", False):
            with col1:
                if st.button("Download as Text"):
                    st.download_button(
                        label="Download Text",
                        data=full_response,
                        file_name="response.txt",
                        mime="text/plain"
                    )
        
        if download_options.get("MS-Word", False):
            with col2:
                if st.button("Download as Word"):
                    docx_file = download_as_docx(full_response)
                    st.download_button(
                        label="Download Word",
                        data=docx_file,
                        file_name="response.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
        
        if download_options.get("CSV", False):
            with col3:
                if st.button("Download as CSV"):
                    csv_file = download_as_csv(full_response)
                    st.download_button(
                        label="Download CSV",
                        data=csv_file,
                        file_name="response.csv",
                        mime="text/csv"
                    )
        
        if download_options.get("MS-PowerPoint", False):
            with col4:
                if st.button("Download as PowerPoint"):
                    pptx_file = download_as_pptx(full_response)
                    st.download_button(
                        label="Download PowerPoint",
                        data=pptx_file,
                        file_name="response.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
        
        if download_options.get("Python", False):
            with col5:
                if st.button("Download as Python"):
                    st.download_button(
                        label="Download Python",
                        data=full_response,
                        file_name="response.py",
                        mime="text/x-python"
                    )

        if download_options.get("Image", False) and "image_url" in st.session_state.messages[-1]:
            with col6:
                image_url = st.session_state.messages[-1]["image_url"]
                response = requests.get(image_url)
                image_data = response.content
                st.download_button(
                    label="Download Image",
                    data=image_data,
                    file_name="generated_image.png",
                    mime="image/png"
                )

# Footer with Clear Screen button
st.markdown("---")
footer_container = st.container()
with footer_container:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<p class="footer-text">Powered by LuminaAI - Illuminating Your Path to Knowledge</p>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear Screen", key="clear_screen", help="Reset the conversation context", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.download_options = {}
            lumina.clear_conversation()
            st.rerun()