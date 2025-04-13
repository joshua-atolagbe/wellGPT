import logging
import nest_asyncio
import os
import sys
import json
import pandas as pd
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI
from prompt import DEFAULT_TEXT_QA_PROMPT, QUESTION_GEN_QUERY

def setup_logging():
    """
    Set up logging configuration.
    """
    nest_asyncio.apply()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# os.environ["OPENAI_API_KEY"] = "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"


def create_sft(input_dir, output_dir):

    documents = SimpleDirectoryReader(input_dir).load_data(show_progress=False)

    # Generate Question/Answer pairs
    llm = OpenAI(model="gpt-4o-mini", temperature=1.0)

    dataset_generator = RagDatasetGenerator.from_documents(
        documents,
        llm=llm,
        num_questions_per_chunk=5,
        show_progress=True,
        text_qa_template = DEFAULT_TEXT_QA_PROMPT,
        # text_question_template = DEFAULT_QUESTION_GENERATION_PROMPT,
        question_gen_query=QUESTION_GEN_QUERY,
        workers=10
    )

    rag_dataset = dataset_generator.generate_dataset_from_nodes()

    # rag_dataset.to_pandas()[:5]  #took 13.28 minutes for 234 chunks

    # Save the dataset to a JSON file
    output_dir.mkdir(parents=True, exist_ok=True)

    rag_dataset.save_json(output_dir+'/instruction.json')

# format to openAI jsonl format
def convert2JSONL(json_objects):
    formatted_examples = []
    for example in json_objects['examples']:
        formatted_example = {
            "messages": [
                {"role": "system", "content": "You're an AI assistant that answers questions about oil/gas well reports."},
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["reference_answer"]}
            ]
        }
        formatted_examples.append(formatted_example)
    return formatted_examples

if __name__ == "__main__":
    setup_logging()

    #input and output directories
    input_dir = "processed"
    output_dir = "sft_data"

    # Create the SFT dataset
    create_sft(input_dir, output_dir)

    # Load the dataset
    with open(os.path.join(output_dir, 'instruction.json'), 'r') as f:
        data = json.load(f)

    # Convert to JSONL format
    examples = convert2JSONL(data)
    with open(os.path.join(output_dir, 'instruction.jsonl'), 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')