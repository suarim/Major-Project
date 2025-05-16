import google.generativeai as genai

genai.configure(api_key="AIzaSyBLY2TrFV25LTkHvw_nScre5NSYlEZnLV0")

models = genai.list_models()

for model in models:
    print(model.name)
