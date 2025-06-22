import logging
from typing import Any, Dict, List
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import your search system
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BanglaSemanticSearch:
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        model_path: str = "output/bangla-sbert-finetuned",
        batch_size: int = 100,  # Reduced batch size for stability
    ):
        """
        Initialize the Bangla Semantic Search system with Weaviate

        Args:
            weaviate_url: Weaviate instance URL
            model_path: Path to your fine-tuned sentence transformer model
            batch_size: Number of documents to process in each batch
        """
        # Initialize Weaviate client (v4)
        self.client = weaviate.connect_to_local(host="localhost", port=8080)
        self.batch_size = batch_size

        # Load your fine-tuned model
        try:
            self.model = SentenceTransformer(model_path)
            logger.info(f"Successfully loaded fine-tuned model from {model_path}")
        except Exception as e:
            logger.warning(
                f"Could not load fine-tuned model: {e}. Using base model instead."
            )
            self.model = SentenceTransformer("sagorsarker/bangla-bert-base")

        self.class_name = "BanglaDocument"

    def create_schema(self):
        """Create Weaviate schema for Bangla documents"""
        # Delete existing collection if it exists
        try:
            self.client.collections.delete(self.class_name)
            logger.info(f"Deleted existing collection {self.class_name}")
        except:
            pass

        # Create new collection with v4 API
        self.client.collections.create(
            name=self.class_name,
            description="Bangla religious documents and questions",
            vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own vectors
            properties=[
                Property(
                    name="document_id",
                    data_type=DataType.TEXT,
                    description="Unique document identifier",
                ),
                Property(
                    name="title", data_type=DataType.TEXT, description="Main Question"
                ),
                Property(
                    name="sub_title",
                    data_type=DataType.TEXT,
                    description="Short form of the main question",
                ),
                Property(
                    name="answer",
                    data_type=DataType.TEXT,
                    description="Answer of the question",
                ),
            ],
        )
        logger.info(f"Created collection {self.class_name}")

    def encode_text(self, text: str) -> List[float]:
        """Encode text using fine-tuned model"""
        if not text or text.strip() == "":
            # Return zero vector for empty text
            return [0.0] * self.model.get_sentence_embedding_dimension()

        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return [0.0] * self.model.get_sentence_embedding_dimension()

    def encode_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts in batch for efficiency"""
        try:
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)

            if not valid_texts:
                # Return zero vectors for all texts
                dim = self.model.get_sentence_embedding_dimension()
                return [[0.0] * dim for _ in texts]

            # Encode valid texts
            embeddings = self.model.encode(
                valid_texts, batch_size=32, show_progress_bar=False
            )

            # Create result array with zero vectors for empty texts
            dim = embeddings.shape[1]
            result = [[0.0] * dim for _ in texts]

            # Fill in valid embeddings
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = embeddings[i].tolist()

            return result

        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            dim = self.model.get_sentence_embedding_dimension()
            return [[0.0] * dim for _ in texts]

    def load_and_index_documents(
        self,
        csv_path: str = "questions_202505201444.csv",
        max_records: int = None,
        start_from: int = 0,
    ):
        """
        Load documents from CSV and index them in Weaviate with optimized batch processing

        Args:
            csv_path: Path to CSV file
            max_records: Maximum number of records to process (None for all)
            start_from: Index to start processing from (for resuming interrupted loads)
        """
        try:
            # Read CSV in chunks to handle large files
            chunk_size = 1000
            df_chunks = []

            logger.info(f"Reading CSV file: {csv_path}")
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                df_chunks.append(chunk)
                if max_records and len(df_chunks) * chunk_size >= max_records:
                    break

            df = pd.concat(df_chunks, ignore_index=True)

            if max_records:
                df = df.head(max_records)

            if start_from > 0:
                df = df.iloc[start_from:].reset_index(drop=True)

            logger.info(f"Loaded {len(df)} documents from {csv_path}")

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return

        # Get the collection
        collection = self.client.collections.get(self.class_name)

        # Process in batches
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        successful_inserts = 0
        failed_inserts = 0

        logger.info(
            f"Processing {len(df)} documents in {total_batches} batches of size {self.batch_size}"
        )

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            try:
                objects_to_insert = []

                # Prepare texts for batch encoding
                texts_to_encode = []
                for _, row in batch_df.iterrows():
                    text_to_embed = str(row.get("sub_title", "")) or str(
                        row.get("title", "")
                    )
                    texts_to_encode.append(text_to_embed)

                # Batch encode all texts
                vectors = self.encode_texts_batch(texts_to_encode)

                # Create document objects
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    doc_obj = {
                        "document_id": f"doc_{start_idx + idx + start_from}",
                        "title": str(row.get("title", "")),
                        "sub_title": str(row.get("sub_title", "")),
                        "answer": str(row.get("answer", "")),
                    }

                    objects_to_insert.append(
                        wvc.data.DataObject(properties=doc_obj, vector=vectors[idx])
                    )

                # Insert batch with error handling
                try:
                    result = collection.data.insert_many(objects_to_insert)

                    # Check for errors in batch insert
                    if hasattr(result, "errors") and result.errors:
                        failed_count = len(
                            [e for e in result.errors.values() if e is not None]
                        )
                        successful_count = len(objects_to_insert) - failed_count
                        failed_inserts += failed_count
                        successful_inserts += successful_count

                        if failed_count > 0:
                            logger.warning(
                                f"Batch {batch_idx + 1}: {failed_count} failed insertions"
                            )
                    else:
                        successful_inserts += len(objects_to_insert)

                except Exception as batch_error:
                    logger.error(
                        f"Error inserting batch {batch_idx + 1}: {batch_error}"
                    )
                    failed_inserts += len(objects_to_insert)

                    # Try individual insertions as fallback
                    logger.info(
                        f"Attempting individual insertions for batch {batch_idx + 1}"
                    )
                    for obj in objects_to_insert:
                        try:
                            collection.data.insert(obj)
                            successful_inserts += 1
                        except Exception as individual_error:
                            logger.error(
                                f"Individual insert failed: {individual_error}"
                            )
                            failed_inserts += 1

                # Small delay to prevent overwhelming the system
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                failed_inserts += len(batch_df)
                continue

        logger.info(
            f"Indexing completed: {successful_inserts} successful, {failed_inserts} failed"
        )
        return {"successful": successful_inserts, "failed": failed_inserts}

    def semantic_search(
        self, query: str, limit: int = 10, min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using your fine-tuned model

        Args:
            query: Search query in Bangla
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of search results with similarity scores
        """
        # Encode query using fine-tuned model
        query_vector = self.encode_text(query)

        # Get collection and perform vector search
        collection = self.client.collections.get(self.class_name)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            distance=1 - min_score,  # Convert certainty to distance
            return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True),
        )

        # Process results
        search_results = []
        for item in response.objects:
            search_results.append(
                {
                    "document_id": item.properties.get("document_id", ""),
                    "title": item.properties.get("title", ""),
                    "sub_title": item.properties.get("sub_title", ""),
                    "answer": item.properties.get("answer", ""),
                    "similarity_score": item.metadata.certainty
                    if item.metadata.certainty
                    else 0,
                    "distance": item.metadata.distance if item.metadata.distance else 1,
                }
            )

        return search_results

    def hybrid_search(
        self, query: str, limit: int = 10, alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (semantic + keyword) for better results

        Args:
            query: Search query
            limit: Maximum results
            alpha: Weight for semantic vs keyword search (0.0 = keyword only, 1.0 = semantic only)
        """
        query_vector = self.encode_text(query)

        collection = self.client.collections.get(self.class_name)

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        search_results = []
        for item in response.objects:
            search_results.append(
                {
                    "document_id": item.properties.get("document_id", ""),
                    "title": item.properties.get("title", ""),
                    "sub_title": item.properties.get("sub_title", ""),
                    "answer": item.properties.get("answer", ""),
                    "hybrid_score": item.metadata.score if item.metadata.score else 0,
                }
            )

        return search_results

    def find_similar_documents(
        self, document_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        collection = self.client.collections.get(self.class_name)

        # First get the document
        response = collection.query.fetch_objects(
            where=Filter.by_property("document_id").equal(document_id), limit=1
        )

        if not response.objects:
            return []

        doc_text = response.objects[0].properties["sub_title"]

        # Use the document text as query to find similar ones
        similar_results = self.semantic_search(doc_text, limit + 1, min_score=0.3)

        # Filter out the original document
        return [
            result for result in similar_results if result["document_id"] != document_id
        ][:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        collection = self.client.collections.get(self.class_name)

        # Get total count
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count

        return {
            "total_documents": count,
            "class_name": self.class_name,
            "model_info": "Fine-tuned Bangla BERT",
        }

    def close(self):
        """Close the Weaviate connection"""
        self.client.close()