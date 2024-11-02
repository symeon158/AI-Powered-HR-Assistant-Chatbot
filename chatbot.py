import streamlit as st
import os
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables (if any)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Set up page configuration
logo_url = "https://aldom.gr/wp-content/uploads/2020/05/alumil.png"

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
        <h1>🤖 AI ESG Assistant</h1>
        <p>Empowering ESG with AI-Driven Insights & Document Management</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Instructions in an expander
instructions = """
**Getting Started:**

**Access the Application:**
- Open the Streamlit app to begin using the AI ESG Assistant.

**Uploading and Processing Documents:**

**Enter OpenAI API Key:**
- In the sidebar, enter your OpenAI API key in the provided text input field. This key is necessary for generating document embeddings and enabling the AI-powered functionalities of the chatbot.

**Upload a File:**
- Use the **"Upload a file"** button in the sidebar to select and upload one or multiple documents from your computer.
- The app supports the following file formats:
  - **PDF** (.pdf): For reports or static documents.
  - **DOCX** (.docx): Word documents with text and formatting.
  - **TXT** (.txt): Plain text files.
  - **CSV** (.csv): Tabular data; each row is converted into strings for querying.

**Adjust Processing Parameters (Optional):**
- Below the file uploader, you’ll find options to adjust the following settings:
  - **Chunk Size**: Controls how the document is segmented into chunks for processing. A larger chunk size preserves more context (recommended: `750–1000`).
  - **Chunk Overlap**: Ensures continuity across chunks by allowing an overlap of characters. Recommended overlap is `10–20%` of the chunk size.
  - **Number of Results to Retrieve (k)**: Determines how many top chunks are retrieved for answering a query. Start with `k=8`.

**Add Data:**
- Click the **"Add Data"** button after uploading your files. The assistant will read, segment, and embed the documents based on the specified chunk size and overlap.
- A **spinner** will indicate that the file is being processed. Once completed, the number of chunks processed and an estimated **embedding cost** will be displayed.

**Asking Questions:**

**Input Your Question:**
- After document processing, enter your query in the **"Ask a question about your documents"** text input field.
  - **Example Questions**:
    - "What are the ESG initiatives mentioned in the 2022 report?"
    - "Provide profit details for Q1 2021."

**Receive an Answer:**
- Upon submitting your question, the assistant will process it through a sequential chain:
  1. **Preprocesses** the question for better context.
  2. **Retrieves relevant document chunks**.
  3. **Generates an answer** based on the retrieved context.
- The answer will appear in the **"LLM Answer"** text area.

**Reviewing Session History:**

**Chat History:**
- A session history of your questions and the AI's responses is automatically maintained. Scroll through the **"Chat History"** section to review all interactions within the session.

**Managing Sessions and Data:**

**Session Reset:**
- To clear the current question and answer history, refresh the app page. This will reset your session without affecting the processed documents.

**Additional Tips:**

**Multiple Documents:**
- If you wish to query multiple documents in a session, upload and process them together. This allows for queries across all uploaded content.

**Improving Answer Accuracy:**
- **Temperature Setting**: For more accurate answers, set a **lower temperature** (e.g., `0.1–0.3`).
- **Max Tokens**: Increase `max_tokens` for more detailed answers (up to `1500–2048`).

**Security:**
- Keep your OpenAI API key confidential and do not share it publicly to avoid unauthorized usage.
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



# Function to ask question and get answer with conversational context
def ask_and_get_answer(vector_store, question, k=3, model_name='gpt-4', temperature=0.7, max_tokens=1500):
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

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

    # Define the retriever
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )

    # Initialize the Conversational Retrieval Chain using from_llm
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
                if f"{PAGE_KEY}_vs" in st.session_state:
                    vector_store = st.session_state[f'{PAGE_KEY}_vs']

                    # Get the answer
                    answer = ask_and_get_answer(
                        vector_store,
                        question,
                        k,
                        model_name,
                        temperature,
                        max_tokens
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
                    st.warning('Please upload and process documents before asking a question.')
        else:
            if f'{PAGE_KEY}_vs' not in st.session_state:
                st.info('Please upload documents and click "Add Data" to process them.')
    else:
        st.write("**Conversation has ended.** Refresh the page or click the button below to start a new conversation.")
        if st.button('Start a New Conversation'):
            st.session_state['conversation_ended'] = False
            clear_history()
            st.write("You can start a new conversation now!")

    # Logo or Branding
    st.sidebar.image('https://aldom.gr/wp-content/uploads/2020/05/alumil.png', use_column_width=True)

# Ensure the main function is called

main()
