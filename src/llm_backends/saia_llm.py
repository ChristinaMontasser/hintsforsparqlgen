from .base_llm import BaseLLMBackend
from langchain_openai import ChatOpenAI

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class SAIABackend(BaseLLMBackend):
    """
    OpenAI-compatible backend for SAIA and other compatible endpoints.

    Config fields expected:
      - api_key: str
      - model_name: str
      - temperature: float (optional)
      - base_url: str (optional, defaults to SAIA public endpoint)
      - max_tokens: int (optional)
    """

    def __init__(self, cfg):
        self.model = ChatOpenAI(
            model=cfg.model_name,
            temperature=getattr(cfg, "temperature", 0.0),
            api_key=cfg.api_key,
            base_url=getattr(cfg, "base_url", "https://chat-ai.academiccloud.de/v1"),
            max_tokens=getattr(cfg, "max_tokens", None),
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": bool(
                        getattr(cfg, "enable_thinking", False)
                    )
                }
            },
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
        content = response.content

        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        content = str(content or "").strip()
        if not content:
            response_metadata = getattr(response, "response_metadata", {})
            finish_reason = response_metadata.get("finish_reason", "unknown")
            raise RuntimeError(
                "The configured LLM returned an empty response "
                f"(finish_reason={finish_reason}). "
                "Increase max_tokens or disable model reasoning for this backend."
            )

        return content
