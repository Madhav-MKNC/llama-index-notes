# miscellaneous / optional / advanced stuff
from usage_pattern import * 
from customizating import *

"""
Cost Predictor:
Creating an index, inserting to an index, and querying an index may use tokens.
We can track token usage through the outputs of these operations.
When running operations, the token usage will be printed.
"""

index.llm_predictor.last_token_usage


"""
Save the index for future use
"""

# to persistant
import os
os.mkdir('./persist_dir')
index.storage_context.persist(persist_dir="./persist_dir")

# To reload from disk:
from llama_index import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
index = load_index_from_storage(storage_context)

# NOTE: If you had initialized the index with a custom ServiceContext object, you will also need to pass in the same ServiceContext during load_index_from_storage.
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index = load_index_from_storage(service_context=service_context,)


"""
Low-level API
"""

from llama_index import (
    VectorStoreIndex,
    ResponseSynthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index, 
    similarity_top_k=2,
)
response_synthesizer = ResponseSynthesizer.from_args(
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)
response = query_engine.query("What did the author do growing up?")
print(response)


"""
Parsing the response
The object returned is a Response object. The object contains both the response text as well as the “sources” of the response:
"""

response = query_engine.query("<query_str>")
print(response.response)
print(response.source_nodes.get_formatted_sources())


