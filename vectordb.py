import chromadb


chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="document")


def store_data(data):

    for page in data:
        data = f"Page_Number: PAGE {page['page_number']}\n\n Page_Image_Description: {page['images']}\n\n TABLE: Table {page['tables']}\n\n Page_Text: {page['text']}\n\n"

        collection.add(
            documents=[data],
            metadatas={"page_number": f"page {page['page_number']}"},
            ids=[f"doc_{page['page_number']}"]
        )

    return collection