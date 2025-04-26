# src/llm_client.py
class MockLLMClient:
    """
    Mock LLM client for testing.
    In production, replace with a real LLM client (e.g., OpenAI, Anthropic).
    """
    def query(self, prompt):
        print(f"Querying LLM with prompt (first 100 chars): {prompt[:100]}...")
        print(f"Total prompt length: {len(prompt)} characters")
        
        # Return a mock response with a JSON array
        return """
        Based on the issue description and file structures, here are the related elements that should be examined:

        [
            {
                "file_path": "dubbo-common/src/main/java/org/apache/dubbo/common/utils/ReflectUtils.java",
                "class_name": "ReflectUtils",
                "method_name": "getReturnTypes",
                "reason": "This method needs to be modified to handle TypeVariable correctly"
            }
        ]
        """

class OpenAIClient:
    """
    OpenAI client for ChatGPT queries.
    """
    def __init__(self, api_key, model="gpt-3.5-turbo"):  # Change default to gpt-3.5-turbo
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def query(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,  # Use the model parameter
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that specializes in Java programming and software engineering."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return f"Error: {str(e)}"

class GeminiClient:
    """
    Gemini client for real LLM queries.
    """
    def __init__(self, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def query(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return f"Error: {str(e)}"