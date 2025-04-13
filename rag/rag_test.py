
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from IPython.display import Markdown, display
from transformers import AutoTokenizer
import torch

#==================================================================================================
#         Load the model using Llama Index
#==================================================================================================

model_name = "joshua-atolagbe/wellGPT_llama"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token = "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",
)

#stop tokens
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

#llm
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7},  
    tokenizer_name=model_name,
    model_name=model_name,
    device_map="auto",
    stopping_ids=stopping_ids,
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16,
                  "token": "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",}
)

#==================================================================================================
#         Load the data using Llama Index and create the embeddings
#==================================================================================================

documents = SimpleDirectoryReader("./reports").load_data()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#hhh
Settings.llm = llm
Settings.chunk_size = 500
Settings.embed_model = embed_model

Settings.transformations = [SentenceSplitter(chunk_size=500)]

# Create the index
index = VectorStoreIndex.from_documents(documents)

def get_response(query_str):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_str)
    return response
# query_str = "What were the main objectives for the well 6407/7-2 in the Haltenbanken field?"

# query_engine = index.as_query_engine()
# response = query_engine.query(query_str)
# display(Markdown(f"<b>{response}</b>"))

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)