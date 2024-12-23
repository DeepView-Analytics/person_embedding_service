import asyncio
import os
import sys
from PIL import Image
import uuid
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,has_collection 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from person_embedding_service.model.Osnet import OSNetEmbedder

class EmbeddingStore:
    def __init__(self, host='localhost', port='19530', collection_name='person_embeddings', dim=512):
        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        
        # Define schema and create collection if it doesn't exist
        self.collection_name = collection_name
        self.collection = self.create_or_get_collection(dim)

    def create_or_get_collection(self, dim):
        # Check if the collection exists
        if not has_collection(self.collection_name):
            field1 = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=255)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            
            schema = CollectionSchema(fields=[field1, field2])
            collection = Collection(name=self.collection_name, schema=schema)
        else:
            collection = Collection(name=self.collection_name)
        return collection
    def create_index(self): 
        # Create an index on the 'embedding' field 
        index_params = { 
          "index_type": "IVF_FLAT",
          "metric_type": "L2",
          "params": {"nlist": 128} }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    def save_to_milvus(self, embeddings):
        """Saves embeddings to Milvus and returns their keys."""
        ids = [f"{str(uuid.uuid4())}" for _ in range(len(embeddings))]
        data = [ { "id": ids[i], "embedding": embeddings[i] } for i , id in enumerate(ids) ]
        
        self.collection.insert(data)
        return ids

    def get_embedding_by_key(self, key):
        """Retrieves an embedding from Milvus using the key."""
        self.collection.load()
        expr = f"id == '{key}'"
        results = self.collection.query(expr, output_fields=["embedding"])
        return results

# Example usage
async def save_to_milvus_example(embedder, images):
    vectors = await embedder.get_embeddings(images)
    embedding_store = EmbeddingStore()
    embedding_store.create_index()
    keys = embedding_store.save_to_milvus(vectors)
    return keys

async def main():
    image_paths = [
        r'D:\DeepView\MicroServices\person_embedding\saru.jpg',
        r'D:\DeepView\MicroServices\person_embedding\1.jpg'
    ]
    
    image_byte_arrays = [Image.open(image_path) for image_path in image_paths]

    timestamp = '2024-12-02T14:00:00Z'
    camera_id = "camera_1"
    embedder = OSNetEmbedder()
    # Assume the embedder instance is available and loaded
    keys = await save_to_milvus_example(embedder, image_byte_arrays, timestamp, camera_id)
    print("Saved Embeddings Keys:", keys)

    # Retrieve an embedding by key for verification
    embedding_store = EmbeddingStore()
    key = keys[1]  # Replace with an actual key from keys list
    embedding_data = embedding_store.get_embedding_by_key(key)
    print("Retrieved Embedding Data:", embedding_data)

# Run the asynchronous main function
if __name__ == '__main__':
    asyncio.run(main())
