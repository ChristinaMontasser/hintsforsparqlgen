from typing import Optional
import cohere
from .base_llm import BaseLLMBackend
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

class CohereBackend(BaseLLMBackend):
    """
    Backend for Cohere Command R / Command R+ models.
    """

    def __init__(self, cfg):
        """
        cfg should contain at least:
          - api_key: str
          - model_name: str (e.g. "command-r-plus")
        """
        self.cfg = cfg
        self.client = cohere.Client(api_key=cfg.api_key)

        self.model_name = cfg.model_name
        self.temperature = getattr(cfg, "temperature", 0.0)
        self.max_tokens = getattr(cfg, "max_tokens", 512)
        self.stop_sequences = getattr(cfg, "stop_sequences", None)

    def generate(
        self,
        prompt,
    ) -> str:
        """
        Generate a response from Command R / R+.

        Args:
            prompt: Either a plain string prompt or a list of chat messages.
        Returns:
            Generated text (string)
        """

        if isinstance(prompt, list):
            system_text = ""
            user_text = ""

            for message in prompt:
                message_type = getattr(message, "type", None)
                message_content = getattr(message, "content", "")

                if message_type == "system":
                    system_text = message_content
                elif message_type in {"human", "user"}:
                    user_text = message_content

            response = self.client.chat(
                model=self.model_name,
                preamble=system_text or None,
                message=user_text or str(prompt),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_sequences=self.stop_sequences,
            )
        else:
            response = self.client.chat(
                model=self.model_name,
                message=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_sequences=self.stop_sequences,
            )

        return response.text.strip()



    def build_prompt(self, prompt_type: str, system_message: str):
        if prompt_type == "instructional":
            return ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_message),
                    HumanMessagePromptTemplate.from_template(
                        """{user_message}
                        Schema:
                        {schema}
                        Question:
                        {question}
                        SPARQL:"""
                    ),
                ],
            )

        elif prompt_type == "structural":
            return ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        "Schema:\n{schema}\n\nQuestion:\n{question}"
                    ),
                ]
            )

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def __call__(self, prompt) -> str:
        """
        Callable interface for uniform backend usage.
        """
        return self.generate(prompt)
