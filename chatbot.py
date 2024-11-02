import streamlit as st
import os
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage


# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables (if any)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Set up page configuration
#logo_url = "https://aldom.gr/wp-content/uploads/2020/05/alumil.png"
logo_url = "data:image/png;base64,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"  # Replace this with the actual URL of the Alumil logo


# Add custom CSS for the starting message and divider
st.markdown("""
<style>
.starting-message {
    color: #2c3e50;
    background-color: #ecf0f1;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #bdc3c7;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    line-height: 1.5;
    text-align: center;
    margin-bottom: 15px;
}
.starting-message h2 {
    color: #2980b9;
    font-size: 24px;
    margin-bottom: 5px;
}
.starting-message p {
    margin: 0;
    color: #34495e;
}
.divider {
    border-top: 1px solid #bdc3c7;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

# Add the starting message and divider to the sidebar
st.sidebar.markdown("""
<div class="starting-message">
    <h2>Welcome!</h2>
    <p>Upload files, ask questions, and get instant insights based on your data!</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)



# Custom CSS to style the header with logo
st.markdown("""
<style>
.custom-header {
    color: #4a4a4a;
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    border: 1px solid #e1e4e8;
}
.custom-header img {
    margin-right: 20px;
    width: 200px;
}
.custom-header h1 {
    color: #333;
    font-size: 40px;
    margin: 0;
}
.custom-header p {
    color: #555;
    font-size: 16px;
    margin: 0;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Header content with logo
st.markdown(f"""
<div class="custom-header">
    <img src="{logo_url}" alt="Alumil Logo">
    <div>
        <h1>🤖 AI Assistant</h1>
        <p>Empowering ESG with AI-Driven Insights & Document Management</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Instructions in an expander
instructions = """
### **AI ESG Assistant: User Guide**

Welcome to the **AI ESG Assistant**! This guide will help you navigate through the app's features, from uploading your documents to getting AI-powered insights on ESG data.

---

#### **1. Initial Setup**

1. **Open the App**: Start by launching the AI ESG Assistant app in Streamlit.
2. **Enter Your OpenAI API Key**:
   - Locate the **API Key Input** section in the sidebar.
   - Input your OpenAI API key in the provided text box.
   - **Note**: The key is required for document embedding and enabling AI functionalities.
   
---

#### **2. App Layout and Components**

- **Sidebar**: Holds settings, instructions, and upload options.
- **Main Panel**: Displays chat interactions and results.

---

#### **3. Configuring Settings**

- **Select Model**:
  - Choose from available models (e.g., `gpt-3.5-turbo`, `gpt-4`).
- **Adjust Temperature**:
  - Control response creativity by adjusting the temperature slider (`0.0 - 1.0`).
  - Lower values yield more focused answers, while higher values provide creative responses.
- **Set Max Tokens**:
  - Specify a max token limit (recommended between `1000–2048`) for the response length.
  
---

#### **4. Uploading Documents**

1. **Choose Files to Upload**:
   - Use the **file uploader** in the sidebar to upload supported file types:
     - PDF (`.pdf`): Reports or static documents
     - DOCX (`.docx`): Word documents
     - TXT (`.txt`): Plain text files
     - CSV (`.csv`): Tabular data, converted into strings for querying
2. **Adjust Processing Parameters (Optional)**:
   - **Chunk Size**: Controls how the document is segmented. Recommended range is `750–1000` for better context preservation.
   - **Chunk Overlap**: Ensures continuity across chunks by allowing a small overlap (recommended: `10-20%`).
   - **Number of Results to Retrieve (`k`)**: Sets the number of top chunks retrieved for answering a query. Start with `k=8`.
3. **Process Documents**:
   - Click **"Add Data"** to begin processing the uploaded documents.
   - Progress bar and status indicators will show processing completion.
   - After processing, embedding details such as the number of chunks and estimated embedding cost are displayed.
   
---

#### **5. Asking Questions**

1. **Enter Your Question**:
   - Type a question about your uploaded documents in the chat input at the bottom of the page.
   - **Example Questions**:
     - "What are the key ESG initiatives from the 2022 report?"
     - "Provide profit details for Q1 2021."

2. **Review the Answer**:
   - The assistant uses AI to retrieve relevant document chunks and generate an answer.
   - Responses appear in the chat area, allowing you to review each answer.
   
---

#### **6. Reviewing Session History**

- **Chat History**:
   - The chat section maintains a history of your questions and the assistant's responses for the duration of your session.
   - To clear the history, use the **Clear Conversation History** button.

---

#### **7. Managing Sessions and Data**

1. **Clear Vector Store**:
   - Click **Clear Vector Store** to delete all embeddings, freeing up memory for new documents.

2. **Clear Conversation History**:
   - Clear conversation history for a fresh start without resetting embeddings.
   
3. **Start a New Conversation**:
   - If a session has ended, refresh the page or click **Start a New Conversation** to reset the conversation.

---

#### **8. Additional Tips for Optimal Use**

- **Multiple Documents**:
   - Upload multiple documents to query across various contents at once.
   
- **Answer Accuracy**:
   - **Temperature**: Use a lower temperature (e.g., `0.1–0.3`) for accurate answers.
   - **Max Tokens**: Increase the max tokens if the answer requires more context (`1500–2048`).

- **Security Note**:
   - Ensure your OpenAI API key remains confidential. Do not share it publicly.

---

Enjoy using the AI ESG Assistant for faster, smarter insights into your ESG data!
"""


# Unique key for this page
PAGE_KEY = 'Chatbot_5'

# Function to check for API Key
def check_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please provide a valid OpenAI API Key.")
        return False
    return True

# Function to load different document types
def load_document(file_path):
    import os
    name, extension = os.path.splitext(file_path)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
    elif extension == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)  # Handle NaN values
        text = df.to_string()
        from langchain.docstore.document import Document
        data = [Document(page_content=text)]
        return data
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Function to chunk data
def chunk_data(data, chunk_size=512, chunk_overlap=50):
    #from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        #separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Function to create embeddings
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory='vector_store'
    )
    vector_store.persist()
    return vector_store

# Function to load existing vector store
def load_vector_store():
    if os.path.exists('vector_store'):
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma(
            persist_directory='vector_store',
            embedding_function=embeddings
        )
        return vector_store
    return None

# Function to calculate embedding cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    cost = total_tokens / 1000 * 0.0004
    if total_tokens > 50000:
        st.warning("Warning: High token count detected! The embedding process might be costly or time-consuming.")
    return total_tokens, cost

# Function to clear history
def clear_history():
    if f'{PAGE_KEY}_history' in st.session_state:
        del st.session_state[f'{PAGE_KEY}_history']
    if 'conversation_history' in st.session_state:
        del st.session_state['conversation_history']
    if 'conversation_memory' in st.session_state:
        del st.session_state['conversation_memory']
    if 'conversation_ended' in st.session_state:
        del st.session_state['conversation_ended']

# Updated Function to clear vector store
def clear_vector_store():
    if f'{PAGE_KEY}_vs' in st.session_state:
        vector_store = st.session_state[f'{PAGE_KEY}_vs']
        try:
            # Retrieve all IDs from the vector store
            ids = vector_store.get()['ids']
            # Delete all embeddings
            vector_store.delete(ids=ids)
            # Remove from session state
            del st.session_state[f'{PAGE_KEY}_vs']
            st.success('Vector store cleared. You can start a new session now.')
        except Exception as e:
            st.error(f"Error clearing vector store: {e}")
    else:
        st.info('No vector store found to clear.')

# Function to ask question and get answer with conversational context, allowing choice between RAG and LLM-only
def ask_and_get_answer(vector_store, question, answer_mode="RAG", k=3, model_name='gpt-4', temperature=0.7, max_tokens=1500):
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    # Initialize LLM model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Retrieve or initialize conversation memory
    if 'conversation_memory' not in st.session_state:
        st.session_state['conversation_memory'] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    memory = st.session_state['conversation_memory']

    # Handle answer mode: RAG (Retrieve & Generate) or LLM Only
    if answer_mode == "RAG":
        # Ensure vector_store is provided
        if vector_store is None:
            st.error("Vector store is not available. Please upload and process documents.")
            return None

        # Define the retriever for RAG mode
        retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': k}
        )

        # Initialize the Conversational Retrieval Chain for RAG
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )

        try:
            with st.spinner("AI Thinking..."):
                result = qa_chain({"question": question})
                answer = result['answer']
                return answer
        except Exception as e:
            st.error(f"Error during RAG processing: {e}")
            return None

    elif answer_mode == "LLM Only":
    # LLM Only mode: generate answer without document retrieval
        try:
            with st.spinner("AI Thinking..."):
                response = llm([HumanMessage(content=question)])
                answer = response.content
                return answer
        except Exception as e:
            st.error(f"Error during LLM processing: {e}")
            return None


# Main application code
def main():
    st.subheader('📂 Upload your documents and ask away. 💡 Quick insights from your ESG data are just a question away! ✨')

    with st.sidebar:
        st.expander("Instructions").markdown(instructions)
        st.header("Settings")

        # API Key Input
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        if not check_api_key():
            st.stop()

        # Model Selection
        model_name = st.selectbox('Choose a model:', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'])
        temperature = st.slider('Temperature:', min_value=0.0, max_value=1.0, value=0.7)
        max_tokens = st.number_input('Max Tokens:', min_value=100, max_value=2048, value=1500)

        # Answer Mode Selection
        answer_mode = st.radio(
            "Choose Answer Mode",
            ("RAG", "LLM Only")
        )

        # File Upload
        uploaded_files = st.file_uploader(
            'Upload files:',
            type=['pdf', 'docx', 'txt', 'csv'],
            accept_multiple_files=True
        )
        chunk_size = st.number_input(
            'Chunk size:',
            min_value=100,
            max_value=2048,
            value=1024,
            on_change=clear_history
        )
        chunk_overlap = st.number_input(
            'Chunk overlap:',
            min_value=0,
            max_value=500,
            value=150,
            on_change=clear_history
        )
        k = st.number_input(
            'Number of Results to Retrieve (k):',
            min_value=1,
            max_value=20,
            value=8,
            on_change=clear_history
        )

        add_data = st.button('Add Data', on_click=clear_history)

        # Add buttons to clear vector store and conversation history
        if st.button('Clear Vector Store'):
            clear_vector_store()

        if st.button('Clear Conversation History'):
            clear_history()
            st.session_state['conversation_ended'] = False
            st.success('Conversation history cleared.')

    # Missing block: Process files when 'Add Data' is pressed
    if uploaded_files and add_data:
        with st.spinner('Processing files...'):
            all_chunks = []
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                if data is not None:
                    chunks = chunk_data(
                        data,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    all_chunks.extend(chunks)
                else:
                    st.error(f'Failed to load the document {uploaded_file.name}. Please check the file format.')

                # Update progress
                progress_bar.progress((idx + 1) / total_files)

            # After all processing
            progress_bar.empty()

            if all_chunks:
                st.sidebar.write(f'Total Chunks: {len(all_chunks)}')
                tokens, embedding_cost = calculate_embedding_cost(all_chunks)
                st.sidebar.write(f'Embedding Cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(all_chunks)
                st.session_state[f'{PAGE_KEY}_vs'] = vector_store
                st.sidebar.success('Files processed and embeddings created successfully!')
            else:
                st.error('No valid documents were processed.')

    # Initialize conversation history and conversation ended flag
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'conversation_ended' not in st.session_state:
        st.session_state['conversation_ended'] = False

    if not st.session_state['conversation_ended']:
        # Display chat history
        if st.session_state['conversation_history']:
            for entry in st.session_state['conversation_history']:
                if 'role' in entry and 'message' in entry:
                    with st.chat_message(entry['role']):
                        st.markdown(entry['message'])
                else:
                    # Handle entries without 'role' or 'message' keys
                    st.error("Invalid conversation history entry detected. Clearing conversation history.")
                    clear_history()
                    st.stop()

        # Use st.chat_input instead of st.text_input
        question = st.chat_input("Ask a question about your documents:")
        if question:
            # Display the user's message
            with st.chat_message("user"):
                st.markdown(question)

            if "thank you" in question.strip().lower():
                with st.chat_message("assistant"):
                    st.write("You're welcome! If you have any more questions, feel free to ask.")
                st.session_state['conversation_ended'] = True
                st.stop()
            else:
                # Check if vector_store is needed
                if answer_mode == "RAG":
                    if f"{PAGE_KEY}_vs" in st.session_state:
                        vector_store = st.session_state[f'{PAGE_KEY}_vs']
                    else:
                        st.warning('Please upload and process documents before asking a question in RAG mode.')
                        st.stop()
                else:
                    vector_store = None  # Set to None in LLM Only mode

                # Get the answer
                answer = ask_and_get_answer(
                    vector_store=vector_store,
                    question=question,
                    answer_mode=answer_mode,
                    k=k,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                if answer:
                    # Display the assistant's message
                    with st.chat_message("assistant"):
                        st.markdown(answer)

                    # Store history
                    st.session_state['conversation_history'].append({
                        'role': 'user',
                        'message': question
                    })
                    st.session_state['conversation_history'].append({
                        'role': 'assistant',
                        'message': answer
                    })
                else:
                    st.error('Failed to get an answer from the model.')
        else:
            if answer_mode == "RAG" and f'{PAGE_KEY}_vs' not in st.session_state:
                st.info('Please upload documents and click "Add Data" to process them.')
    else:
        st.write("**Conversation has ended.** Refresh the page or click the button below to start a new conversation.")
        if st.button('Start a New Conversation'):
            st.session_state['conversation_ended'] = False
            clear_history()
            st.write("You can start a new conversation now!")

    # Logo or Branding
    #st.sidebar.image('https://aldom.gr/wp-content/uploads/2020/05/alumil.png', use_column_width=True)

# Ensure the main function is called
if __name__ == "__main__":
    main()
