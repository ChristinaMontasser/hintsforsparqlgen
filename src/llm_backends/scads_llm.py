from .base_llm import BaseLLMBackend
from langchain_openai import ChatOpenAI

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class SCADSBackend(BaseLLMBackend):
    """
    OpenAI-compatible backend for TUD SCADS LLM API.

    Config fields expected:
      - api_key: str
      - model_name: str
      - temperature: float (optional)
      - base_url: str (optional, defaults to SCADS public endpoint)
      - max_tokens: int (optional)
    """

    def __init__(self, cfg):
        self.model = ChatOpenAI(
            model=cfg.model_name,
            temperature=getattr(cfg, "temperature", 0.0),
            api_key=cfg.api_key,
            base_url=getattr(cfg, "base_url", "https://llm.scads.ai/v1"),
            max_tokens=getattr(cfg, "max_tokens", None),
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
        response = self.model.invoke(prompt)
        return response.content
