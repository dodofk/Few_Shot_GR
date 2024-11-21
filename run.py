from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import torch  # Import torch to use torch.bfloat16


class FewShotGR:
    def __init__(
        self, model_name: str = "meta-llama/Llama-2-8b-chat", device: str = "cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16  # Set model to use bf16 precision
        )
        self.model.to(device)
        self.device = device
        self.docid_bank = {}  # Maps docid to document
        self.doc_to_docids = {}  # Maps document to list of its docids

    def generate_prompt(self, query: str) -> str:
        """
        Generate a formatted prompt with examples for identifier generation.
        """
        return f"""Generate a hyphen-separated identifier based on these examples:

[Example 1]
Q: Provide list of the olympic games?
→ olympic-game-list

[Example 2]
Q: What is minority interest in accounting?
→ subsidiary-corporation-parent

[Example 3]
Q: How does photosynthesis work in plants?
→ photosynthesis-plant-process

[Your Query]
Q: {query}
→"""

    def generate_pseudo_queries(self, document: str, n_queries: int = 10) -> List[str]:
        # Prompt for generating diverse questions about the document
        prompt = f"Generate {n_queries} different questions seperated by newlines that could be answered by this document:\n{document}\n\nQuestions:"

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        output_ids = self.model.generate(
            input_ids,
            max_length=512,
            num_return_sequences=n_queries,
            num_beams=n_queries,
            temperature=0.7,
            do_sample=True,
        )

        queries = []
        for output in output_ids:
            query = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract just the generated question, removing the prompt
            query = query.split("Questions:")[-1].strip().split("\n")[0]
            queries.append(query)

        return queries[:n_queries]  # Ensure we return exactly n_queries

    def _validate_identifier_format(self, identifier: str) -> bool:
        """
        Validate if the identifier follows the correct hyphen-separated format.

        Args:
            identifier (str): The identifier to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check if identifier contains at least one hyphen
        if "-" not in identifier:
            return False

        # Check that all parts have content (no empty segments)
        parts = identifier.split("-")
        if not all(part.strip() for part in parts):
            return False

        # Check no other separators are used
        cleaned_parts = [part.strip() for part in parts]
        reconstructed = "-".join(cleaned_parts)
        return reconstructed == identifier.strip()

    def generate_docid(self, query: str, existing_docids: set = None) -> str:
        prompt = self.generate_prompt(query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        max_attempts = 3
        for _ in range(max_attempts):
            output_ids = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 15,
                min_length=len(input_ids[0]) + 3,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
            )
            docid = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            docid = docid.split("Identifier:")[-1].strip()

            if self._validate_identifier_format(docid):
                if existing_docids and docid not in existing_docids:
                    closest_docid = self.find_closest_docid(docid, existing_docids)
                    return closest_docid
                return docid

        # If all attempts fail, raise an error
        raise ValueError("Failed to generate valid hyphen-separated identifier")

    def index_documents(self, documents: List[str], n_docids_per_doc: int = 10):
        for doc in tqdm(documents, desc="Indexing documents"):
            # Generate multiple pseudo queries for the document
            pseudo_queries = self.generate_pseudo_queries(doc, n_docids_per_doc)

            # Generate docids for each pseudo query
            doc_docids = []
            for query in pseudo_queries:
                docid = self.generate_docid(query)
                if docid not in self.docid_bank:  # Avoid duplicates
                    self.docid_bank[docid] = doc
                    doc_docids.append(docid)

            self.doc_to_docids[doc] = doc_docids

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        # Generate docid for query
        generated_docid = self.generate_docid(query, set(self.docid_bank.keys()))

        # Get document and calculate confidence score
        if generated_docid in self.docid_bank:
            document = self.docid_bank[generated_docid]
            confidence = 1.0  # Perfect match
            return [(document, confidence)]

        # If no exact match, return empty list
        return []

    def retrieve_with_partial_matching(
        self, query: str, similarity_threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents based on partial matching using word-based similarity comparison.

        Args:
            query (str): The search query
            similarity_threshold (float): Minimum similarity score (default: 0.8)

        Returns:
            List[Tuple[str, float]]: List of (document, similarity_score) pairs
        """
        query_words = set(query.lower().split("-"))
        results = []

        for existing_docid, document in self.docid_bank.items():
            existing_words = set(existing_docid.lower().split("-"))
            similarity = self.calculate_similarity(query_words, existing_words)
            if similarity >= similarity_threshold:
                results.append((document, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def calculate_similarity(self, words1: set, words2: set) -> float:
        """
        Calculate similarity between two sets of words based on common words.

        Args:
            words1 (set): First set of words
            words2 (set): Second set of words

        Returns:
            float: Similarity score between 0 and 1
        """
        common_words = words1.intersection(words2)
        total_unique_words = words1.union(words2)

        if not total_unique_words:
            return 0.0

        return len(common_words) / len(total_unique_words)


# Example usage
def main():
    # Initialize Few-Shot GR
    # model = "meta-llama/Llama-3.1-8B-Instruct"
    model = "mistralai/Mistral-7B-Instruct-v0.1"
    retriever = FewShotGR(model)

    # Example documents
    documents = [
        "In accounting, minority interest (or non-controlling interest) is the portion of a subsidiary corporation's stock that is not owned by the parent corporation.",
        "The Olympic Games are a major international multi-sport event.",
    ]

    # Index documents
    # retriever.index_documents(documents)
    
    # test the generate_query function
    print(retriever.generate_pseudo_queries(documents[0], 10))

    # # Example query
    # query = "What is minority interest in accounting?"
    # results = retriever.retrieve(query)

    # # Print results
    # for doc, score in results:
    #     print(f"Retrieved document (score: {score}):\n{doc}\n")


if __name__ == "__main__":
    main()
