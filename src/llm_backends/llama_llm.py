import os
import torch
from .base_llm import BaseLLMBackend
import transformers

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.prompts import PromptTemplate

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer, prompt_length):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # Only decode newly generated tokens
        generated_ids = input_ids[0][self.prompt_length:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return any(stop_str in decoded for stop_str in self.stop_strings)



class LLAMABackend(BaseLLMBackend):
    def __init__(self, cfg):
        """
        cfg should contain:
        - model_name (e.g. "")
        - device ("cuda" or "cpu")
        """
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.device = cfg.device if torch.cuda.is_available() else "cpu"


        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            token=os.getenv("HF_TOKEN"),
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device if self.device == "cuda" else None,
            #cache_dir= os.getenv("HF_HOME", None), 
        )


        print(f"LLAMAnstruct initialized with {self.model_name} on {self.device}")

    def build_prompt(self, prompt_type: str, system_message: str):

        if prompt_type == "instructional":
            return ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    ("human",
                     """{user_message}

                    Schema:
                    {schema}

                    Question:
                    {question}

                    SPARQL:
                    """)
                ]
            )

        elif prompt_type == "structural":
            return ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    ("human",
                     """{user_message}
                    {question}

                    """)
                ]
            )

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def generate(self, prompt) -> str:
        if isinstance(prompt, list):
            prompt_text = self.pipeline.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt

        # Generate
        inputs = self.pipeline.tokenizer(prompt_text, return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[1]

        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(
                ["Human:", "Question:", "Query:", "Answer:"],
                self.pipeline.tokenizer,
                prompt_length
            )
        ])

        with torch.no_grad():
            outputs = self.pipeline(
                prompt_text,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                stopping_criteria=stopping_criteria,
                return_full_text=False
            )
        return outputs[0]["generated_text"]
