
import os
from dotenv import load_dotenv
from chatbot import get_embedding # type: ignore
from rag_utils.faiss_utils import index_existing_text_files

load_dotenv()

# Change this to the actual folder containing your HR .txt files
DOCS_FOLDER = os.getenv("HR_DOCS_FOLDER", "data")


if not os.path.isdir(DOCS_FOLDER):
    raise FileNotFoundError(
        f"HR documents folder not found: {os.path.abspath(DOCS_FOLDER)}"
    )

index, docs = index_existing_text_files(
    get_embedding,
    DOCS_FOLDER
)

print("FAISS index created successfully.")
print("Documents are ready for chatbot search.")
