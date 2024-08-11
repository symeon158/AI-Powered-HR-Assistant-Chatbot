import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

import streamlit as st

import streamlit as st

import streamlit as st

import streamlit as st

# URL of the Alumil logo image
#logo_url = "https://aldom.gr/wp-content/uploads/2020/05/alumil.png"  # Replace this with the actual URL of the Alumil logo
logo_url = "data:image/png;base64,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"  # Replace this with the actual URL of the Alumil logo
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
    width: 200px; /* Adjust based on your logo's size */
}
.custom-header h1 {
    color: #333;
    font-size: 40px; /* Adjusted to balance with logo size */
    margin: 0;
}
.custom-header p {
    color: #555;
    font-size: 16px; /* Adjusted to balance with logo size */
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
        <h1>🤖 AI 
        HR Assistant</h1>
        <p>Empowering HR with AI-Driven Insights & Document Management</p>
    </div>
</div>
""", unsafe_allow_html=True)



# Instructions in an expander
instructions = """
**Getting Started:**

**Access the Application:**
- Open the Streamlit app. This can typically be done by navigating to the URL where the app is hosted or running it locally if you're working in a development environment.

**Uploading Documents:**

**Enter OpenAI API Key:**
- In the sidebar, enter your OpenAI API key in the provided text input field. This key is necessary for generating document embeddings and enabling the AI-powered functionalities of the ChatBot.

**Upload a File:**
- Use the "Upload a file" button in the sidebar to select and upload a document from your computer. The app supports PDF, DOCX, and TXT formats, catering to most document types you'll encounter in HR operations.

**Adjust Parameters (Optional):**
- Below the file uploader, you'll find options to adjust the chunk size and the parameter `k`. The chunk size determines how the document is segmented for processing, while `k` controls the number of results returned for queries. Adjust these based on your needs, but the default settings should work well for general use.

**Add Data:**
- Click the "Add Data" button after uploading your file. This initiates the reading, chunking, and embedding process for the uploaded document. A spinner will indicate that the file is being processed.

**Asking Questions:**

**Input Your Question:**
- Once the document is processed, you can ask specific questions about its content. Enter your question in the "Ask a question about the content of your file!" text input field.

**Receive an Answer:**
- After submitting your question, the app will display the answer in the "LLM Answer" text area. This response is retrieved based on the semantic understanding of your query and the document's content.

**Reviewing Session History:**

**Chat History:**
- The application automatically keeps a history of your questions and the AI's answers for the duration of your session. You can review this history in the "Chat History" text area, which helps track information retrieved during your session.

**Additional Tips:**

**Multiple Documents:**
- If you wish to query multiple documents during a single session, repeat the upload and processing steps for each new document. Remember to adjust chunk size and `k` as needed.

**Session Reset:**
- To start a new session or clear the current session history, refresh the application page. This resets the environment and clears any stored data or history.

**Security:**
- Keep your OpenAI API key confidential to prevent unauthorized usage. The application does not store your key permanently, but it's essential to exercise caution.
"""


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.run(q)
    return answer

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    st.image('img.webp')

    st.subheader('📂 Upload your documents and ask away. 💡 Quick insights from your HR data are just a question away! ✨')

    with st.sidebar:
        st.expander("Instructions").markdown(instructions)
        api_key = st.text_input('OpenAI API Key:', type = 'password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf','docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100,max_value=2048, value=512,on_change=clear_history)
        k=st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)
        
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking, and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./',uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk Size: {chunk_size}, Chunks: {len(chunks)}')
                
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding Cost: ${embedding_cost:.4f}')
                
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded, Chunked and Embedded successfully')
                

q = st.text_input('Ask a question about the content of your file!')
if q:
    if "vs" in st.session_state:
        vector_store = st.session_state.vs
        #st.write(f'k: {k}')
        answer = ask_and_get_answer(vector_store, q, k)
        st.text_area('LLM Answer: ', value=answer)
        
        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        h = st.session_state.history
        st.text_area(label='Chat History', value=h, key='history', height=400)
    
    
    
    
