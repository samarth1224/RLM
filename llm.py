import random


class BaseLLM:
    def __init__(self, provider, api_key):
        self.provider = provider
        self.api_key = api_key

    def call_llm(self,prompt):
        return f"called agent with Prompt = {prompt}"
    
class GeminiLLM(BaseLLM):
    def __init__(self,api_key):
        super().__init__("gemini", api_key)
        self.provider = "Gemini"
    def call_llm(self,prompt):
        if random.randint(1,10) %2 == 0:
            return "code"
        return f"called gemini agent with Prompt = {prompt}"
        
        