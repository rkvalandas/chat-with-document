# PDF Document Query System

A powerful application that allows you to chat with your PDF documents using advanced AI technologies. Upload any PDF document and ask questions about its content in natural language.

## Features

- **PDF Processing**: Upload and process PDF documents of any size
- **Natural Language Queries**: Ask questions about your document in plain English
- **Accurate Answers**: Get precise answers with relevant context from the document
- **Source Transparency**: View the exact sections of the document used to generate answers
- **Efficient Processing**: PDFs are processed only once after upload, not with each query

## Technology Stack

This application leverages state-of-the-art technologies:

- **Streamlit**: For the interactive web interface
- **LangChain**: For orchestrating the RAG workflow
- **LangGraph**: For creating a structured retrieval and generation pipeline
- **HuggingFace Embeddings**: Using the "thenlper/gte-large" model for semantic understanding
- **Groq LLM**: Using Llama 4 models for high-quality answer generation
- **PyPDF Loader**: For extracting text from PDF documents
- **Vector Storage**: For efficient semantic search capabilities

## How It Works

The application uses a Retrieval Augmented Generation (RAG) approach:

1. **Document Processing**:
   - When you upload a PDF, the document is processed and converted to vector embeddings
   - The embeddings capture the semantic meaning of the document's content
   - The processed document is stored in-memory for quick retrieval

2. **Query Processing**:
   - When you ask a question, the system finds the most relevant sections from your document
   - These sections are used as context for the AI to generate accurate answers
   - The system ranks and sorts results by page number for better context coherence

3. **Answer Generation**:
   - A Llama 4 model from Groq evaluates your question against the retrieved context
   - The model generates a precise answer based solely on the document's content
   - You can view both the answer and the source context used to generate it

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Groq API key

### Installation

1. Clone the repository
   ```
   git clone <repository-url>
   cd chat-with-document
   ```

2. Install the required packages
   ```
   pip install -r requirements.txt
   ```

3. Set up your Groq API key
   - Create a `.streamlit/secrets.toml` file with your API key:
     ```
     GROQ_API_KEY = "your-api-key-here"
     ```

### Running the Application

```
streamlit run app.py
```

## Usage

1. **Upload a PDF**: Use the file uploader to select your PDF document
2. **Ask a Question**: Type your question in the text input field
3. **View the Answer**: The system will display the answer based on the document content
4. **Explore Context**: Expand the "Show source context" section to see which parts of the document were used

## Limitations

- The application works best with text-based PDFs; scanned documents may not process correctly
- Very large PDFs may take longer to process initially
- The quality of answers depends on the clarity and structure of the source document

## Future Improvements

- Support for additional document formats (DOCX, TXT, etc.)
- Multi-document querying
- Persistent storage for processed documents
- Chat history to maintain conversation context


## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG implementation
- [Groq](https://groq.com/) for the powerful LLM API
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Streamlit](https://streamlit.io/) for the web interface framework