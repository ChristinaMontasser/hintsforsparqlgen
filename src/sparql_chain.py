from langchain_core.output_parsers import StrOutputParser
# from .prompts import build_prompt
#from qachain import GraphSparqlQAChain

class SPARQLGenerator:
    def __init__(self, llm, system_message: str, prompt_type: str = "instructional"):
        self.llm = llm
        self.prompt = self.llm.build_prompt(prompt_type, system_message)

       #chain = GraphSparqlQAChain.from_llm(llm, g)
    def generate(self, user_message:str, question: str, schema: str) -> str:
        if hasattr(self.prompt, "format_messages"):
            messages = self.prompt.format_messages(
                user_message=user_message,
                schema=schema,
                question=question
            )
            #print(f"Generated prompt messages:\n{messages}\n")
            return self.llm(messages)

        input_text = self.prompt.format_prompt(
            user_message=user_message,
            schema=schema,
            question=question
        ).to_string()
        #print(f"Generated prompt:\n{input_text}\n")

        #print(f"Generated prompt:\n{input_text}\n")
        return self.llm(input_text)
