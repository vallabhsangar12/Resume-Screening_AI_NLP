import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load model and vectorizers
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Extract text from different file types
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type.")

# Predict the resume category
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Streamlit App
def main():
    st.set_page_config(page_title="Resume Category Predictor", page_icon="🧾", layout="centered")

    # Custom CSS for better UI
    st.markdown("""
        <style>
            .main-title {
                font-size: 2.5rem;
                color: #2c3e50;
                text-align: center;
                font-weight: bold;
            }
            .sub-title {
                font-size: 1.2rem;
                color: #555;
                text-align: center;
            }
            .footer {
                font-size: 0.85rem;
                color: #333;
                background-color: #f2f2f2;
                text-align: center;
                padding: 12px;
                margin-top: 3rem;
                border-radius: 8px;
            }
            .result-box {
                background-color: #ffe5e5;
                color: #d62828;
                padding: 15px;
                border-radius: 8px;
                border-left: 5px solid #d62828;
                font-size: 1.2rem;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>📄 Resume Category Prediction App</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Upload your resume and let AI predict the job category!</div>", unsafe_allow_html=True)
    st.markdown("")

    uploaded_file = st.file_uploader("📤 Upload your Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted successfully.")

            if st.checkbox("🔍 Show Extracted Resume Text"):
                st.text_area("Extracted Text", resume_text, height=300)

            st.markdown("### Predicted Category")
            category = pred(resume_text)
            st.markdown(f"<div class='result-box'>🔎 The resume is categorized as: <b>{category}</b></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    st.markdown("<div class='footer'>❤ Developed during Microsoft x Edunet Foundation AICTE Internship</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
