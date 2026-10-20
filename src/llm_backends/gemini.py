from google import genai



class GeminiBackend(BaseLLMBackend):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = genai.Client()
        self.model_name = cfg.llm.model_name

        print(f"Initialized OpenAIBackend with model: {cfg.model_name}")
        
    def generate(self, prompt: str) -> str:
        response = self.model.models.generate_content(
            model= self.model_name,
            contents= prompt
        )        
        return response.text
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def build_prompt(self, prompt_type: str, system_message: str):
        pass