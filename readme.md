# PDF Data Extractor and Query System

This project is a **Streamlit-based web application** that extracts data (text, tables, and images) from PDF files. It uses **Groq's LLaMA models** to analyze and extract structured information from images, and **ChromaDB** for storing and querying extracted content. 

 ## Checkout the live demo here: [Chat-with-document](https://chat-with-document1.streamlit.app/)

---

## Features 🚀

1. **PDF Upload and Data Extraction**:
   - Upload a PDF file to the application.
   - Extract text, tables, and image data from each page of the PDF.

2. **AI-Powered Image Analysis**:
   - Uses **Groq's LLaMA Vision model** to analyze images extracted from the PDF pages.
   - Structured data is extracted from tables, graphs, and other visual content.

3. **Data Storage with ChromaDB**:
   - Extracted content is stored in a **ChromaDB collection** for querying.

4. **Intelligent Query System**:
   - Users can input queries to search and retrieve relevant information from the PDF data.
   - Responses are generated with the help of **Groq's LLaMA-3 models**.

5. **Dynamic Cleanup**:
   - ChromaDB collections are cleaned up after querying for efficiency.

---

## Tech Stack 🛠️

- **Python**: Core programming language
- **Streamlit**: Web framework for building interactive UI
- **Groq API**: LLaMA-3 models for image and text processing
- **pdfplumber**: Extracts text and tables from PDFs
- **PyMuPDF (fitz)**: Extracts images from PDF files
- **Pillow**: Image processing library
- **ChromaDB**: Vector database for storage and querying

---

## Installation 🛠️

1. Clone this repository:

   ```bash
   git clone https://github.com/rkvalandas/chat-with-document.git
   cd chat-with-document
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Add **GROQ API Key** to your Streamlit secrets:

   - Run this command:

     ```bash
     streamlit secrets set GROQ_API_KEY "your-groq-api-key"
     ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

---

## How It Works 🔍

1. **Upload PDF**:
   - The application accepts a PDF file.
   - Text, tables, and images are extracted page by page.

2. **Image Analysis**:
   - Extracted images are encoded to Base64 and sent to the **Groq Vision model**.
   - The model returns structured data for any tables, graphs, or diagrams.

3. **Data Storage**:
   - The extracted data is stored in a **ChromaDB collection** for efficient querying.

4. **Query Data**:
   - Users can input natural language queries.
   - The application retrieves relevant information from the stored data and uses **Groq LLaMA-3 models** to generate a detailed response.

---

## Usage Instructions 📋

1. Upload your PDF file via the Streamlit interface.
2. Wait for data extraction and ChromaDB setup to complete.
3. Enter a query in the query box and click **Run Query**.
4. View the AI-generated response based on the PDF content.

---

## Example Workflow 🧩

1. **Upload**: Upload a sample PDF file containing text, tables, and images.
2. **Data Extraction**:
   - Extracted Text: "Page 1: Introduction to AI..."
   - Extracted Table: "Row 1: Data1, Data2..."
   - Extracted Image Analysis: "Graph showing yearly trends..."
3. **Query**: Enter "What are the key trends in the graph?".
4. **Result**: "The key trends in the graph show an increase in revenue from 2020 to 2023."

---

## Dependencies 📦

- `streamlit`
- `groq`
- `langchain`
- `chromadb`
- `pdfplumber`
- `PyMuPDF`
- `Pillow`

---

## Future Improvements 🌟

- Support for multiple PDFs.
- Enhanced error handling and progress tracking.
- Integration with other vector databases.

---

## License 📜

This project is licensed under the MIT License.

---

**Happy Querying! 🎉**