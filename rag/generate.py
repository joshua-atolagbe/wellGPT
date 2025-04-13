from pipeline import pipeline, terminators


class Llama3_8B_gen:
    def __init__(self,pipeline):
        self.pipeline= pipeline

    @staticmethod
    def generate_prompt(query,retrieved_text):
        messages = [
            {"role": "system", "content": "Answer the Question for the given below context and information and not prior knowledge, only give the output result \n\ncontext:\n\n{}".format(retrieved_text) },
            {"role": "user", "content": query},]
        return pipeline.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)

    def generate(self,query,retrieved_context):
        prompt = self.generate_prompt(query ,retrieved_context)
        output =  pipeline(prompt,max_new_tokens=512,eos_token_id=terminators,do_sample=True,
                            temperature=0.7,top_p=0.9,)
        return output[0]["generated_text"][len(prompt):]