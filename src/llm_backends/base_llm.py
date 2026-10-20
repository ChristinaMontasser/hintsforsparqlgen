from abc import ABC as ABS, abstractmethod

class BaseLLMBackend(ABS):
    @abstractmethod
    def generate(self, prompt, **kwargs) -> str:
        pass
    def __call__(self, prompt) -> str:
        return self.generate(prompt)
    @abstractmethod
    def build_prompt(self, prompt_type: str, system_message: str):
        pass