import faiss
import numpy as np
import json
import os
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from fpdf import FPDF
from functools import lru_cache

# FAISS Vector Database
class FAISSVectorDB:
    def __init__(self, index_path="./faiss_index", metadata_path="./faiss_metadata.json", model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = 384
        self.index_path = index_path
        self.metadata_path = metadata_path

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded existing FAISS index and metadata from {self.index_path}")
        except:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print(f"Created new FAISS index at {self.index_path}")

    def add_embeddings(self, texts, ids=None):
        embeddings = self.encoder.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        if ids is None:
            ids = list(range(len(self.metadata), len(self.metadata) + len(texts)))
        self.metadata.extend(zip(ids, texts))
        print(f"Added {len(texts)} embeddings to the FAISS index.")

    def search(self, query_text, k=5):
        query_embedding = self.encoder.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:
                id_, text = self.metadata[idx]
                results.append((id_, text, distance))
        return results

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        print(f"Saved FAISS index and metadata to {self.index_path}")

# Groq Generator
class GroqGenerator:
    def __init__(self, model_name='mixtral-8x7b-32768'):
        self.model_name = model_name
        self.client = Groq()

    def generate_lesson(self, topic, retrieved_content):
        prompt = f"""
        Create an engaging English lesson about {topic}. Use the following information:
        {retrieved_content}

        Structure the lesson as follows:
        1. **Introduction**: Briefly introduce the topic.
        2. **Key Concepts**: Explain the main ideas in detail.
        3. **Examples**: Provide real-world examples to illustrate the concepts.
        4. **Practice Exercises**: Include interactive exercises for the student.
        5. **Conclusion**: Summarize the lesson and encourage further practice.

        Lesson:
        """
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an AI English teacher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content

# RAG English Teacher
class RAGEnglishTeacher:
    def __init__(self, vector_db, generator):
        self.vector_db = vector_db
        self.generator = generator

    @lru_cache(maxsize=32)
    def teach(self, topic):
        relevant_content = self.vector_db.search(topic)
        lesson = self.generator.generate_lesson(topic, relevant_content)
        return lesson

# Helper Functions
def extract_text_from_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def add_pdf_to_vector_db(pdf_path, chunk_size=500, chunk_overlap=50):
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text:
        chunks = chunk_text(pdf_text, chunk_size, chunk_overlap)
        vector_db.add_embeddings(chunks)
        print(f"Content from {pdf_path} added to the vector database.")
    else:
        print(f"Failed to extract text from {pdf_path}")

def save_lesson_as_pdf(topic, lesson, output_dir='lessons'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Lesson on {topic}", ln=1, align="C")
    pdf.multi_cell(0, 10, txt=lesson)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{output_dir}/lesson_{topic.replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# Streamlit App
def main():
    st.title("AI English Tutoring App")
    lesson_name = st.text_input("Lesson Name")

    if st.button("Generate Lesson"):
        st.write(f"Generating lesson for {lesson_name}...")
        lesson = teacher.teach(lesson_name)
        st.markdown(f"### Lesson on {lesson_name}")
        st.write(lesson)

        if st.button("Save as PDF"):
            save_lesson_as_pdf(lesson_name, lesson)

# Initialize the RAG English Teacher system
vector_db = FAISSVectorDB()
generator = GroqGenerator()
teacher = RAGEnglishTeacher(vector_db, generator)

# Add content from a PDF file
add_pdf_to_vector_db('./English_Grammar_in_Use_Intermediate.pdf')

# Run the Streamlit app
if __name__ == "__main__":
    main()