import streamlit as st
import os
import groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import fitz
import pdfplumber
from io import BytesIO
from PIL import Image
import base64
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from database import Vectordb

groq.api_key = st.secrets["GROQ_API_KEY"]

# Disable parallelism for HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

vectordb = Vectordb()

# Helper functions
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def nested_list_to_string(nested_list):
    result = []
    for sublist in nested_list:
        for inner_list in sublist:
            formatted_row = ', '.join(str(item) if item is not None else '' for item in inner_list)
            result.append(formatted_row)
    return '\n'.join(result)

def get_image_data_from_groq(base64_image):
    client = groq.Groq(api_key=groq.api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 
                        """
                        ### INSTRUCTION:
                        The image is from a page in a document.
                        Your job is to extract complete information from the image and return the data in a structured format. It may contain tables, graphs, and piecharts or anything.
                        ### WITHOUT ANY ADDITIONAL COMMENT, INTRODUCTORY OR CONCLUDING REMARKS (NO PREAMBLE):
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-90b-vision-preview",
    )

    return chat_completion.choices[0].message.content

def extract_pdf_text_and_images(pdf_path):
    pdf_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            doc = fitz.open(pdf_path)
            for count, page in enumerate(pdf.pages, start=1):
                page_number = count
                text = page.extract_text()
                tables = nested_list_to_string(page.extract_tables())

                page_data = {
                    'page_number': page_number,
                    'text': text,
                    'tables': "",
                    'images': tables
                }

                fitz_page = doc.load_page(page_number - 1)
                images = fitz_page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))

                    if image.width < 2 or image.height < 2:
                        continue

                    try:
                        encoded_image = encode_image(image_bytes)
                        llm_img_data = get_image_data_from_groq(encoded_image)
                        page_data['images'] += f"Image: {llm_img_data}"
                    except Exception as e:
                        print(f"Error extracting data from image on page {page_number}, image {img_index + 1}: {e}")

                pdf_texts.append(page_data)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return pdf_texts


# Streamlit UI
st.title("PDF Data Extractor and Query System")

# File Upload
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf is not None:
    pdf_path = f"temp_{uploaded_pdf.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    # Extract data from PDF (only once)
    if 'extracted_data' not in st.session_state:
        st.write("Extracting data from PDF...")
        extracted_data = extract_pdf_text_and_images(pdf_path)
        st.session_state.extracted_data = extracted_data  # Store the extracted data in session_state

        # Delete the PDF file after extracting data
        if os.path.exists(pdf_path):  # Check if the file exists
            os.remove(pdf_path)
        else:
            st.warning("Temporary file not found.")

        vectordb.delete_all_collections()
        # Add data to ChromaDB
        st.write("Adding data to ChromaDB...")
        st.session_state.collection = vectordb.load_data(st.session_state.extracted_data)
        st.success("Data successfully added to ChromaDB!")

# Ensure the extracted data is loaded into the session state
if 'extracted_data' in st.session_state:
    extracted_data = st.session_state.extracted_data

    # Query section
    st.subheader("Query the extracted data")
    query = st.text_input("Enter your query")

    if st.button("Run Query"):
        if 'collection' in st.session_state:
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq.api_key)

            # Execute the query on the collection
            page = st.session_state.collection.query(
                query_texts=[query],
                n_results=9
            )['documents'][0]

            prompt = ChatPromptTemplate.from_template(
                """
                ### PAGE CONTENT:
                {page}
                ### INSTRUCTION:
                You are an assistant tasked with providing relevant information from the above page content based on the following query: {query}.
                ### (NO PREAMBLE):
                """
            )
            query_chain = prompt | llm | StrOutputParser()
            query_result = query_chain.invoke({"page": page, "query": query})
            st.write("Query Result", query_result)
        else:
            st.warning("No collection found. Please upload a PDF first.")
else:
    st.warning("No PDF data extracted. Please upload a PDF.")


