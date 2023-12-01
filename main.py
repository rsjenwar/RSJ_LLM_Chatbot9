import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chatbot App')
    st.markdown('''
    ## About
    This AI App is an LLM-powered chatbot based on user data source, built using open and free technologies:
    - [LLAMA2](https://platform.openai.com/docs/models) LLM model
    - [LangChain](https://python.langchain.com/)
    - [Streamlit](https://streamlit.io/)


    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by RSJ@MEITY.GOV.IN')

load_dotenv()


def main():
    st.header("LLM Chatbot 💬")
    add_vertical_space(4)
    # select chatbot type

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'Finetune: Embeddings: {store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # LLM
            llm = Ollama(model="llama2",
                         verbose=True,
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            print(f"Loaded LLM model {llm.model}")

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            # print(cb)
            st.write(response)


if __name__ == '__main__':
    main()
