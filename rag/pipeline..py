import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import transformers
from langchain.document_loaders import UnstructuredPDFLoader,PDFMinerLoader,TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Hugging Face model id
model_id = "joshua-atolagbe/wellGPT_llama"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,token="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",
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)

terminators =  [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]