from langchain.document_loaders import WebBaseLoader
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from PIL import Image

# requirements.txt
# streamlit
# langchain
# openai
# chromadb
# bs4
# tiktoken

  
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



st.title("🌐 :blue[Webpage Reader]")

system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key


url = st.text_input("**Step 1 : Enter the web page URL**", "https://edition.cnn.com/world")

with st.expander("Sample URL"):
    image = Image.open("webreader-sample-1.jpg")
    st.image(image, caption='https://www.basiclaw.gov.hk/en/basiclaw/chapter3.html')

    image = Image.open("webreader-sample-2.jpg")
    st.image(image, caption='https://www.td.gov.hk/mini_site/cic/en/laws/cap374.html')

    image = Image.open("webreader-sample-3.jpg")
    st.image(image, caption='https://www.hr.hku.hk/career_opportunities/how_to_apply.html')

print(f">>>  web page url: {url}")

query = st.text_input("**Step 2 : Enter your query ?**", "Please summarize the content of this web page in point form, and extract 10 keywords")
if st.button("Submit", type="primary"):
    with st.spinner('Generating ...'):
        st.markdown('#### Extract Process')
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()
        st.write(f'✔️ Loading website completed :  {url}')

        print(f">>> Load data from : {url}")

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n', 
                                        chunk_size=1000, 
                                        chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        no_chunks = len(docs)
        st.write(f'✔️ webpage content chunking completed :  {str(no_chunks)}')


        print(f">>> web page split data into : {no_chunks} chunks ") 


        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings()
        st.write('✔️ Embedding completed')

        # Create a local Chroma vector database from the documents
        # vectordb = Chroma.from_documents(documents=docs, 
        #                                 embedding=openai_embeddings,
        #                                 persist_directory=DB_DIR)
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=openai_embeddings,
            persist_directory=DB_DIR,
            metadata_field="chunk_id"  # Specify a metadata field to store chunk IDs
        )

        # Add metadata for each chunk
        for i, chunk in enumerate(docs):
            chunk_metadata = {
                "chunk_id": str(i),  # Convert the chunk ID to a string
                # Add other metadata fields if necessary
            }
            vectordb.add_metadata(chunk_metadata)


        vectordb.persist()
        st.write('✔️ Local VectorDB created completed')

        print(f">>> Save the docs at  a local Chroma vector database at : {DB_DIR}")

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatOpenAI model
        #llm = ChatOpenAI(model_name='text-davinci-003')
        llm = ChatOpenAI()

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        st.write('✔️ Langchain created and LLM proessing ...')

        # Run the query and return the result
        result = qa.run(query)
        print(f">>> Create RetrievalQA and run query :\n{query}")
        
        st.write('✔️ LLM query completed  ...')

        st.markdown('#### Query Result"')
        st.info(result)
        print(f">>> Query result:\n{result}")



    

