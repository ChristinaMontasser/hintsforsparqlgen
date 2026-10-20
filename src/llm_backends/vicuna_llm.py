import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

from .base_llm import BaseLLMBackend


class VicunaBackend(BaseLLMBackend):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.model_name

        self.device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False, 
            cache_dir= os.getenv("HF_HOME", None)
        )
        print("Tokenizer loaded.")
        print( os.getenv("HF_HOME", None))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            cache_dir= os.getenv("HF_HOME", None)
        )
        self.model.eval()
        print("Model loaded.")

        self.generation_config = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        print(f"VicunaBackend initialized with {self.model_name}")

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

                        Assistant:"""
                    ),
                ],
            )

        elif prompt_type == "structural":
                return ChatPromptTemplate.from_messages(
                    [
                        HumanMessagePromptTemplate.from_template(
                            """Schema:
                            {schema}

                            {user_message}

                            Question:
                            {question}

                            ### ASSISTANT: SPARQL: """
                        )
                    ]
                )

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    def generate(self, prompt: str) -> str:
        print(f"Generating response for prompt of length {len(prompt)} characters.")
        print(f"Prompt preview: {prompt[:200]}...")  # Print the first 200 characters of the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        print(f"Input tokens: {inputs['input_ids'].shape[1]}")
        print("Input keys:", list(inputs.keys()))
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        input_len = inputs["input_ids"].shape[1]
        print(f"Output tokens: {output_ids.shape[1] - input_len}")
        # Return only the generated part of the output
        return self.tokenizer.decode(
            output_ids[0, input_len:],  # Only return the generated part (after input)
            skip_special_tokens=True

        )
