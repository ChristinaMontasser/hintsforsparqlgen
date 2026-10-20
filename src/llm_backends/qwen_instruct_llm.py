import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_llm import BaseLLMBackend
from langchain_core.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        AIMessagePromptTemplate
    )
class QwenInstructBackend(BaseLLMBackend):
    def __init__(self, cfg):
        """
        cfg should contain:
        - model_name (e.g. "qwen-2.5-14b-instruct")
        - device ("cuda" or "cpu")
        """
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.device = cfg.device if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir= os.getenv("HF_HOME", None))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            cache_dir= os.getenv("HF_HOME", None)
        )

        print(f"QwenInstruct initialized with {self.model_name} on {self.device}")



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

                    """)
                ],
            )

        elif prompt_type == "structural":
            return ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    ("human", """{user_message}
                    {question}

                    """)
                ]
            )

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
    def generate(self, prompt) -> str:
        if isinstance(prompt, list):
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt

        # Tokenize input
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Remove prompt from output
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return generated_text[len(prompt_text):].strip()
