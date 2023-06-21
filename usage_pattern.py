"""
USAGE PATTERN FOR LLAMA_INDEX FRAMEWORK
"""

"""
#######################################################
STEP 1: Load Documents, manually or using data loaders
#######################################################
"""

# for loading documents using data loaders
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader('./data').load_data()

# for loading documents manually
from llama_index import Document
texts_list = [
"LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).",
"LangChain is amazing."
]
documents = [Document[text] for text in texts_list]

# adding metadata in docs
document = Document(
    text="""
        LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).
        LangChain is amazing.
    """,
    extra_info={
        'filename': './data/sample.txt',
        'category': 'info'
    }
)


"""
#################################
STEP 2: Parse Documents => Nodes
#################################
"""

# using node parser
from llama_index.node_parser import SimpleNodeParser
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents=documents)

# manually
from llama_index.data_structs.node import Node, DocumentRelationship
node1 = Node(text="""

""")
node1 = Node(text="LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).", doc_id="node1")
node2 = Node(text="LangChain is amazing.", doc_id="node2")
node1.relationships[DocumentRelationship.NEXT] = node2.get_doc_id()
node2.relationships[DocumentRelationship.PREVIOUS] = node1.get_doc_id()
nodes = [node1, node2]


"""
#########################################
STEP 3: Contruct Index from Nodes / docs
#########################################
"""

# from docs
from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents=documents)

# from nodes
from llama_index import VectorStoreIndex
index = VectorStoreIndex(nodes=nodes)


# Reusing Nodes across Index Structures
from llama_index import StorageContext, ListIndex
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)
index1 = VectorStoreIndex(nodes, storage_context=storage_context)
index2 = ListIndex(nodes, storage_context=storage_context)

# Inserting docs
from llama_index import VectorStoreIndex
index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)

# Inserting nodes
from llama_index import VectorStoreIndex
index = VectorStoreIndex([])
index.insert_nodes(nodes=nodes)


"""
STEP 4: [optional/advanced] Build index on top of other indices
"""

# code


"""
STEP 5: Query the Index
"""

query_engine = index.as_query_engine(verbose=True)
response = query_engine.query("<query_text>")
print(response)


