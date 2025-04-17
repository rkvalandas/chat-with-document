import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

# Set TOKENIZERS_PARALLELISM to False to avoid deadlocks and warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the RAG state structure
class RAGState(TypedDict):
    """State for the RAG application."""
    question: str
    retrieved_documents: Optional[List[Document]]
    context: Optional[str]
    answer: Optional[str]

# Function to process the uploaded PDF file
def process_pdf(pdf_file):
    """Process the uploaded PDF and create a vector store."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
        
    # Split text into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=600)
    # all_splits = text_splitter.split_documents(pages)
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",  # This model produces embeddings
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
    )
    
    # Create vector store from documents
    vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
    
    # Create a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Clean up temp file
    os.unlink(pdf_path)
    
    return retriever

# Retrieve relevant documents based on the question
def retrieve(state: RAGState, retriever):
    """Retrieve relevant documents based on the question."""
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    sorted_results = sorted(
        retrieved_docs,
        key=lambda doc: doc.metadata.get('page', float('inf'))
    )
    context = "\n\n".join([doc.page_content for doc in sorted_results])
    
    return {
        "question": question,
        "retrieved_documents": retrieved_docs,
        "context": context,
        "answer": None
    }

# Generate answer based on retrieved documents
def generate_answer(state: RAGState):
    """Generate an answer based on the question and retrieved context."""
    # Check if Groq API key is available
    groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        return {
            "question": state["question"],
            "retrieved_documents": state["retrieved_documents"],
            "context": state["context"],
            "answer": "Error: GROQ_API_KEY not found in environment variables. Please set up your API key."
        }
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Using Llama 3.3 for better reasoning
        api_key=groq_api_key,
        temperature=0.2  # Lower temperature for more factual responses
    )
    
    prompt_template = """
    You are a helpful AI assistant that answers questions based on the provided context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer the question based only on the provided context. Be concise, accurate, and helpful.
    If the answer cannot be determined from the context, say so.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Generate answer
    answer = llm.invoke(
        prompt.format(
            context=state["context"],
            question=state["question"]
        )
    ).content
    
    return {
        "question": state["question"],
        "retrieved_documents": state["retrieved_documents"],
        "context": state["context"],
        "answer": answer
    }

# Create the RAG application
def create_rag_app(retriever):
    # Set up the state graph
    workflow = StateGraph(RAGState)
    
    # Add the retrieve node
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    
    # Add the generate_answer node
    workflow.add_node("generate_answer", generate_answer)
    
    # Set up the edges between nodes
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    
    # Compile the graph
    return workflow.compile()

# Main Streamlit UI
def main():
    st.set_page_config(page_title="PDF Query App", layout="wide")
    
    st.title("PDF Document Query System")
    st.markdown("""
    Upload a PDF document and ask questions about its content.
    The system will use Retrieval Augmented Generation (RAG) to provide accurate answers based on the document.
    """)
    
    # Initialize session state for PDF processing
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
        
    # File uploader
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
    
    # Process PDF only if a new file is uploaded and not already processed
    if pdf_file is not None and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            retriever = process_pdf(pdf_file)
            st.session_state.retriever = retriever
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = pdf_file.name
            st.success(f"PDF '{pdf_file.name}' processed successfully!")
    
    # Show processing status if PDF has been processed
    if st.session_state.pdf_processed and 'pdf_name' in st.session_state:
        st.info(f"Using processed PDF: {st.session_state.pdf_name}")
        
    # Reset processing state if a different file is uploaded
    if pdf_file is not None and hasattr(st.session_state, 'pdf_name') and pdf_file.name != st.session_state.pdf_name:
        with st.spinner("Processing new PDF..."):
            retriever = process_pdf(pdf_file)
            st.session_state.retriever = retriever
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = pdf_file.name
            st.success(f"PDF '{pdf_file.name}' processed successfully!")
    
    # Query section
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question about the document:")
    
    if query and 'retriever' in st.session_state:
        with st.spinner("Generating answer..."):
            # Set up initial state
            initial_state = {
                "question": query,
                "retrieved_documents": None,
                "context": None,
                "answer": None
            }
            
            # Create and run the RAG application
            rag_app = create_rag_app(st.session_state.retriever)
            result = rag_app.invoke(initial_state)
            
            # Display the results
            st.subheader("Answer")
            st.write(result["answer"])
            
            # Optionally show the context (can be toggled)
            with st.expander("Show source context"):
                st.markdown(result["context"])
    
    elif query:
        st.warning("Please upload a PDF document first.")

if __name__ == "__main__":
    main()