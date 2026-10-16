
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = "text-embedding-3-small"


def get_embedding(text):
    if not API_KEY or not ENDPOINT or not API_VERSION:
        raise ValueError(
            "Please configure AZURE_OPENAI_KEY, "
            "AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_VERSION in .env"
        )

    client = AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION
    )

    response = client.embeddings.create(
        input=text,
        model=DEPLOYMENT
    )

    embedding = response.data[0].embedding

    print("Embedding dimensions:", len(embedding))
    print("Token usage:", response.usage.total_tokens)

    return embedding


if __name__ == "__main__":
    text = "What are the leave policies?"
    embedding = get_embedding(text)
    print("First 5 embedding values:", embedding[:5])
