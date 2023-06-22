#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Env Variables
from dotenv import load_dotenv
load_dotenv()

from llama_index import (
    SimpleDirectoryReader,
    node_parser,
    VectorStoreIndex,
    ServiceContext
)

# configuring LLM
from llama_index.llm_predictor import HuggingFaceLLMPredictor
stable_llm_predictor = HuggingFaceLLMPredictor(
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b"
)
service_context = ServiceContext.from_defaults(
    chunk_size=1024, 
    llm_predictor=stable_llm_predictor
)
HuggingFaceLLMPredictor(
    tokenizer_outputs_to_remove=["token_type_ids"]
) 

# main
documents = SimpleDirectoryReader('../data').load_data()
parser = node_parser.SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents=documents)
index = VectorStoreIndex(
    nodes=nodes,
    service_context=service_context
)
query_engine = index.as_query_engine(verbose=True)
while True:
    query = input("[enter the query] ")
    print("[chatTEXT]",query_engine.query(query))
