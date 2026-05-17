from google import genai
import os
from dotenv import load_dotenv

load_dotenv()  # loads from root signbridge/.env automatically

api_key = os.getenv("GEMINI_API_KEY")
print(f"Gemini key loaded: {'YES' if api_key else 'NO'}")

client = genai.Client(api_key=api_key)

def correct_grammar(raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    
    prompt = f"""You are helping a deaf person communicate. 
Convert this raw sign language output into grammatically correct natural English.
Keep the meaning exactly the same. Only return the corrected sentence, nothing else.
Input: {raw_text}
Output:"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return raw_text