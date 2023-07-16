from langchain.document_loaders import WebBaseLoader
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from PIL import Image


  
st.set_page_config(
    page_title="Read Web Page",
    page_icon="🌐 ",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

image = Image.open("read-web.jpg")
st.image(image, caption='created by MJ')



st.title("🌐 Read your Web Page")

system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key

url = st.text_input("Step 1 : Enter the web page URL", "https://edition.cnn.com/world")
print(f">>>  web page url: {url}")

query = st.text_input("Step 2 : Enter your query ?", "Please summarize the content of this web page in point form, and extract 10 keywords")
if st.button("Submit", type="primary"):
    ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    DB_DIR: str = os.path.join(ABS_PATH, "db")

    # Load data from the specified URL
    loader = WebBaseLoader(url)
    data = loader.load()
    print(f">>> Load data from : {url}")

    # Split the loaded data
    text_splitter = CharacterTextSplitter(separator='\n', 
                                    chunk_size=1000, 
                                    chunk_overlap=200)

    docs = text_splitter.split_documents(data)

    no_chunks = len(docs)

    print(f">>> web page split data into : {no_chunks} chunks ") 


    # Create OpenAI embeddings
    openai_embeddings = OpenAIEmbeddings()

    # Create a local Chroma vector database from the documents
    vectordb = Chroma.from_documents(documents=docs, 
                                    embedding=openai_embeddings,
                                    persist_directory=DB_DIR)

    vectordb.persist()

    print(f">>> Save the docs at  a local Chroma vector database at : {DB_DIR}")

    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use a ChatOpenAI model
    #llm = ChatOpenAI(model_name='text-davinci-003')
    llm = ChatOpenAI()

    # Create a RetrievalQA from the model and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Run the query and return the result
    result = qa.run(query)
    print(f">>> Create RetrievalQA and run query :\n{query}")


    st.write(result)
    print(f">>> Query result:\n{result}")


log = """

→ Data : WebBaseLoader
→ Split documents : CharacterTextSplitter
→ Text embedding: openai
→ vector stores : Chroma
→ Agent : N/A
→ Chain : N/A
→ LLM search: RetrievalQA
→ Prompt template: N/A



>>> Load data from : https://www.sandbox.game/en/create/vox-edit/
>>> split data into : <built-in method count of list object at 0x1354b9700> chunks 
>>> Save the docs at  a local Chroma vector database at : /Users/davidcheung/Desktop/Demo/Langchain-website/db
>>> Create RetrievalQA and run query :
what are product or service in this web page ?
>>> Query result:
Based on the provided context, the products and services on this web page include:

- Mac (computers)
- iPad (tablets)
- iPhone (smartphones)
- Watch (Apple Watches)
- TV & Home (Apple TV and related accessories)
- Accessories for Apple devices
- Support for Apple devices and services
"""        
    
with st.expander("explanation"):
    st.code(log)

