# AI_tutor_using_RAG_GROQ

The code provided implements an AI English Tutoring App using a Retrieval-Augmented Generation (RAG) system. It combines a vector database (FAISS) for storing and retrieving relevant content, a language model (Groq API) for generating lessons, and a Streamlit-based UI for user interaction. Below is a detailed explanation of the code:

1. FAISSVectorDB Class
This class handles the storage and retrieval of text embeddings using FAISS, a library for efficient similarity search.

Key Components:
__init__ Method:

Initializes the FAISS index and metadata storage.

Loads an existing index and metadata if available; otherwise, creates a new one.

Uses SentenceTransformer to generate embeddings for text.

add_embeddings Method:

Encodes a list of texts into embeddings using the SentenceTransformer model.

Adds the embeddings to the FAISS index.

Stores metadata (e.g., text and IDs) for retrieval.

search Method:

Encodes a query text into an embedding.

Searches the FAISS index for the k nearest neighbors.

Retrieves metadata (e.g., text and IDs) for the search results.

save_index Method:

Saves the FAISS index and metadata to disk for persistence.

2. GroqGenerator Class
This class generates lessons using the Groq API, which leverages a large language model (e.g., Mixtral).

Key Components:
__init__ Method:

Initializes the Groq client and sets the model name.

generate_lesson Method:

Constructs a prompt for the Groq API using the topic and retrieved content.

Sends the prompt to the Groq API and retrieves the generated lesson.

Returns the generated lesson as a string.

3. RAGEnglishTeacher Class
This class combines the FAISSVectorDB and GroqGenerator to create an AI English teacher.

Key Components:
__init__ Method:

Initializes the vector database and generator.

teach Method:

Retrieves relevant content from the vector database using the topic.

Generates a lesson using the Groq API.

Returns the generated lesson.

4. Helper Functions
These functions support the main functionality of the app.

Key Functions:
extract_text_from_pdf:

Extracts text from a PDF file using PyPDF2.

chunk_text:

Splits text into smaller chunks using RecursiveCharacterTextSplitter from LangChain.

add_pdf_to_vector_db:

Extracts text from a PDF, splits it into chunks, and adds the chunks to the vector database.

save_lesson_as_pdf:

Saves a generated lesson as a PDF file using FPDF.

5. Streamlit App
The Streamlit app provides a user interface for generating and saving lessons.

Key Components:
main Function:

Displays a title and a text input for the lesson topic.

Generates a lesson when the "Generate Lesson" button is clicked.

Displays the generated lesson and allows the user to save it as a PDF.

6. Workflow
Initialize the System:

Create an instance of FAISSVectorDB to store and retrieve embeddings.

Create an instance of GroqGenerator to generate lessons.

Combine them into an RAGEnglishTeacher instance.

Add Content to the Vector Database:

Extract text from a PDF and split it into chunks.

Add the chunks to the vector database as embeddings.

Generate a Lesson:

Retrieve relevant content from the vector database using a topic.

Generate a lesson using the Groq API.

Display the lesson in the Streamlit app.

Save the Lesson:

Save the generated lesson as a PDF file.

Example Usage
Add Content:

The user uploads a PDF (e.g., English_Grammar_in_Use_Intermediate.pdf).

The app extracts text from the PDF, splits it into chunks, and adds it to the vector database.

Generate a Lesson:

The user enters a topic (e.g., "Present Continuous Tense").

The app retrieves relevant content from the vector database and generates a lesson using the Groq API.

The lesson is displayed in the Streamlit app.

Save the Lesson:

The user clicks "Save as PDF" to save the lesson as a PDF file.




