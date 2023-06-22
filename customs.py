"""
##################
Customizing LLM's
##################
NOTE: default: GPT-3 text-davinci-003
"""

from llama_index import LLMPredictor, VectorStoreIndex, ServiceContext
from langchain import OpenAI
documents = SimpleDirectoryReader('./data').load_data()
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = VectorStoreIndex.from_documents(
    documents=documents,
    service_context=service_context
)


"""
##############
Custom Prompt
##############
NOTE: [https://gpt-index.readthedocs.io/en/latest/how_to/customization/custom_prompts.html]
"""

from llama_index import Prompt, VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
TEMPLATE_STR = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
QA_TEMPLATE = Prompt(TEMPLATE_STR)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)
response = query_engine.query("What did the author do growing up?")
print(response)



"""
##################
Custom Embeddings
##################
"""

# An example snippet for ListIndex

from llama_index import ListIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
new_index = ListIndex.from_documents(documents)

# query with embed_model specified
query_engine = new_index.as_query_engine(
    retriever_mode="embedding", 
    verbose=True, 
    service_context=service_context
)
response = query_engine.query("<query_text>")
print(response)


# An example snippet for VectorStoreIndex

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# load index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
new_index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context,
)

# query will use the same embed_model
query_engine = new_index.as_query_engine(
    verbose=True, 
)
response = query_engine.query("<query_text>")
print(response)






