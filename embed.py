import google.generativeai 

client = google.generativeai.Client(api_key="AIzaSyDUECoDmrGVRF1lS9vIlgQWhBDUVSTj2rA")

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the meaning of life?")

print(result.embeddings)