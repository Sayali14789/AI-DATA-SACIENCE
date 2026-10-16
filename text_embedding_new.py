import os
from dotenv import load_dotenv # type: ignore
from openai import AzureOpenAI # type: ignore
API_KEY = os.getenv("AZURE_OPENAI_KEY")
def get_embedding(text):
    import os
    from openai import AzureOpenAI # type: ignore

    endpoint = "https://umesh-mmnk7d7h-eastus2.cognitiveservices.azure.com/"
    model_name = "text-embedding-3-small"
    deployment = "text-embedding-3-small"

    api_version =os.getenv("AZURE_OPENAI_API_VERSION")

    client = AzureOpenAI(
        api_version=api_version,
        # endpoint=endpoint,
        api_key=API_KEY
    )

    response = client.embeddings.create(
        input=["first phrase","second phrase","third phrase"],
        model=deployment
    )

    for item in response.data:
        length = len(item.embedding)
        print(
            f"data[{item.index}]: length={length}, "
            f"[{item.embedding[0]}, {item.embedding[1]}, "
            f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
        )
    print(response.usage)


if __name__ == "__main__":
    text = "What are the leave policies?"
    embedding = get_embedding(text)
    print(embedding)