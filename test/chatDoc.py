#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from llama_index import (
    VectorStoreIndex,
    node_parser,
    SimpleDirectoryReader
)

from dotenv import load_dotenv
load_dotenv()

documents = SimpleDirectoryReader('../data').load_data()
parser = node_parser.SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents=documents)
index = VectorStoreIndex(nodes=nodes)

query_engine = index.as_query_engine(verbose=True)
while True:
    query = input("[enter the query] ")
    print("[chatTEXT]",query_engine.query(query))
