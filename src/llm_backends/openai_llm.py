from .base_llm import BaseLLMBackend
from langchain_openai import ChatOpenAI


from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.prompts import PromptTemplate

class OpenAIBackend(BaseLLMBackend):
    def __init__(self, cfg):
        self.model = ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            api_key=cfg.api_key,
        )

    def build_prompt(self, prompt_type: str, system_message: str):
        if prompt_type == "instructional":
            return ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human",
                 """{user_message}

                    Schema:
                    {schema}

                    Question:
                    {question}

                    SPARQL:
                    """)
            ])
        elif prompt_type == "structural":
            return ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{user_message}\n{question}")
            ])
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def generate(self, prompt, **kwargs) -> str:
        #messages = prompt_template.format_messages(**kwargs)
        response = self.model.invoke(prompt)
        return response.content