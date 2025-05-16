import google.generativeai as genai
import sys
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    print(json.dumps({"error": "No signs provided"}))
    sys.exit(1)

sign_string = sys.argv[1]

# GOOGLE_API_KEY = os.environ.get(
#     "GEMINI_API_KEY", "AIzaSyDKo6QCqrRkJY9Llaa8d1gX4pACe6Hl-W8"
# )
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

prompt = (
    f"Translate the following sequence of sign language words into a simple English phrase. "
    f"Use only the given words and the absolute minimum necessary connecting words (like 'on', 'in', 'at'). "
    f"For example if the recognized sign is cow then the sentence should be 'I saw a cow'."
    f"Signs: {sign_string}"
)

# Retry settings
max_retries = 3
retry_delay = 30  # seconds

model = genai.GenerativeModel("gemini-1.5-flash")

for attempt in range(max_retries):
    try:
        response = model.generate_content(prompt)
        print(json.dumps({"sentence": response.text.strip()}))
        break
    except Exception as e:
        error_message = str(e)
        # Check for quota or rate limit violations
        if "quota" in error_message.lower() or "rate limit" in error_message.lower():
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(
                    json.dumps(
                        {"error": "Rate limit exceeded. Please try again later."}
                    )
                )
                sys.exit(1)
        else:
            # Other errors (e.g., bad request, API key issues)
            print(json.dumps({"error": error_message}))
            sys.exit(1)
