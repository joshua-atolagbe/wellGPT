from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.base import PromptTemplate


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Well Completion or Drilling or Petrophysical context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. Ensure the answer is technical. The answer must explicitly include the well name and the main area or field name or oil block. \n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

QUESTION_GEN_QUERY = """\
You are a trained Well or Oil Field Engineer. Your task is to setup {num_questions_per_chunk} questions from well completion/drilling/petrophysical reports.
The questions should be diverse in nature across the document. Restrict the questions to the context information provided.
The context and question must explicitly include the well name and the main area or field name.
Do not generate questions from context that do not explicitly include the well name and the main area or field name.
"""