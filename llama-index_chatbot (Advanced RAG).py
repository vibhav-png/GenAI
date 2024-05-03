
#Importing necessary libraries
import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, Settings
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core import QueryBundle
from llama_index.core.retrievers import (BaseRetriever,VectorIndexRetriever,KeywordTableSimpleRetriever,)
from hybrid_search import CustomRetriever
from llama_index.core import ChatPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.postprocessor.cohere_rerank import CohereRerank
from prompts2 import chat_text_qa_msgs

from dotenv import load_dotenv
import nest_asyncio

#setting the environment varibles
load_dotenv()
nest_asyncio.apply()

#setting up LLM and embedding model
llm = OpenAI(model="ft:gpt-3.5-turbo-1106:personal:my-experiment-1:9KJP7r8G", temperature=0.1)
embed_model = OpenAIEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model

#Loading and splitting the document
about_docs = SimpleDirectoryReader(input_dir="Final").load_data()
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(about_docs)

#Hybrid search
vector_store = LanceDBVectorStore('C:\\Users\\Vibhav\\vscode\\lancedb')
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes=nodes, storage_context=storage_context)

#Setting retriver pipelines
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

#Setting up re-ranker with query engine
text_qa_template_test = ChatPromptTemplate(chat_text_qa_msgs)
cohere_rerank = CohereRerank(top_n=2)
custom_query_engine = RetrieverQueryEngine.from_args(
    retriever=custom_retriever,
    node_postprocessors=[cohere_rerank],
    text_qa_template=text_qa_template_test,
)
#custom_query_engine.update_prompts({"response_synthesizer:text_qa_template": new_prompt_tmpl})

#Streamlit app
st.title("Chatbot 💬")
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how may I assist you?"}
    ]

chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=custom_query_engine, verbose=True,)

if prompt := st.chat_input("your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            pprint_response(response, show_source=True)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)

st.cache_data.clear()