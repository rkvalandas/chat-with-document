import chromadb

class Vectordb:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient('vectordb')

    def load_data(self, data):
        self.collection = self.chroma_client.get_or_create_collection(name="document")
        for page in data:
            data = f"Page_Number: PAGE {page['page_number']}\n\n Page_Image_Description: {page['images']}\n\n TABLE: Table {page['tables']}\n\n Page_Text: {page['text']}\n\n"
            self.collection.add(
                documents=[data],
                metadatas={"page_number": f"page {page['page_number']}"},
                ids=[f"doc_{page['page_number']}"]
            )
        return self.collection

    def delete_collection(self):
        self.chroma_client.delete_collection("document")

    def delete_all_collections(self):
        """Deletes all collections in ChromaDB."""
        try:
            collections = self.chroma_client.list_collections()
            if not collections:
                print("No collections to delete.")
                return

            for collection in collections:
                collection_name = collection.name
                self.chroma_client.delete_collection(name=collection_name)
                print(f"Deleted collection: {collection_name}")
            print("All collections have been deleted successfully.")
        except Exception as e:
            print(f"Error deleting collections: {e}")
