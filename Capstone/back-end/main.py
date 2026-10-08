from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from back-end/.env
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)
# Flask-CORS adds the CORS headers to every response
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "DELETE", "OPTIONS"], "allow_headers": "*"}})

# Configuration - use absolute path so it works regardless of CWD
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"txt", "pdf"}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Global variables

vector_store = None
qa_chain = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# Process documents, loading data inside loader on basis of file extension
def process_documents():
    """Process all documents in uploads folder and create vector store"""
    global vector_store, qa_chain

    documents = []

    # Load all documents from upload folder

    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        print(f"Upload Folder does not existis: {upload_folder}")
        return False

    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder,filename)
        if not os.path.isfile(file_path):
            continue
        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error Loading {filename}: {str(e)}")

    if not documents:
        return False

        #split documents in to chunks

    try:
        text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create embeddigns

        embeddings = HuggingFaceBgeEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(splits,embeddings)

    except Exception as e:
        print(f"Error creating embedding/vector stores: {e}")
        raise

    #initializing Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model = "openai/gpt-oss-20b",
        temperature=0.3
    )

    #Creating Custom Prompt template

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context","question"]
    )

    #Creating QA_chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents = True
    )

    return True

@app.route('/upload',methods=['POST'])
def upload_file():
    """Upload document endpoint"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}),400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}),400
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only .txt and .pdf are allowed"}),400

    try:
        #ensure upload folder exists
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder,exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder,filename)
        file.save(file_path)

        #success process docs

        success = process_documents()

        if success:
            return jsonify({
                "message": "File uploaded and processed successfully",
                "filename": filename
            }),200
        else:
            return jsonify({"error": "Error processing documents"}),500

    except Exception as e:
        print(f"Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error":f"Failed to upload document: {str(e)}"}),500

@app.route("/query", methods=['POST'])
def query():
    """Query the knowledge base"""
    global qa_chain

    if qa_chain is None:
        # Server may have restarted - rebuild from files on disk
        try:
            process_documents()
        except Exception as e:
            print(f"Rebuild error: {e}")
            return jsonify({'error': f'Failed to load documents: {e}'}),500
    if qa_chain is None:
        return jsonify({'error':'No documents uploaded yet. Please upload a document'}),400
    data = request.json
    question = data.get('question','')

    if not question:
        return jsonify({'error':"No question provided"}),400
    try:
        #Get answer from QA chain

        result = qa_chain.invoke({"query": question})
        return jsonify ({
            "answer": result['result'],
            "sources": [doc.metadata.get('source','Unknown') for doc in result['source_documents']]
        }),200
    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}),500

@app.route("/documents", methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    files = []
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        return jsonify({"documents":files}),200
    for filename in os.listdir(upload_folder):
        if allowed_file(filename):
            file_path = os.path.join(upload_folder,filename)
            files.append({
                "name": filename,
                "size": os.path.getsize(file_path)
            })
    return jsonify({"documents":files}),200

@app.route("/delete/<filename>",methods=['DELETE'])
def delete_documents(filename):
    """Delete a document"""
    global vector_store, qa_chain

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)

            if not process_documents():
                vector_store = None
                qa_chain = None
            return jsonify({"message":f"{filename} deleted successfully"}),200

        else:
            return jsonify({
                "error": "File not found"
            }),404
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route("/clear",methods=['POST'])
def clear_all():
    """Clear all documents"""
    global vector_store, qa_chain

    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            vector_store = None
            qa_chain = None
            return jsonify({'message': 'All documents cleared'}), 200
        # Remove all files
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Reset vector store and QA chain
        vector_store = None
        qa_chain = None
        
        return jsonify({'message': 'All documents cleared'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_loaded': qa_chain is not None
    }), 200

if __name__ == "__main__":
    # create upload folder if it doesnot exist

    os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)
    app.run(debug=True,host="0.0.0.0", port=5002, use_reloader=False)