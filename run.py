"""
Few-Shot GR Model Implementation

This module implements a Few-Shot Generation and Retrieval model for document indexing
and retrieval using hyphen-separated identifiers.

Features:
    - Document indexing with multiple identifiers per document
    - Query-based document retrieval
    - Pseudo-query generation for better document representation
    - Partial matching support for flexible retrieval

Usage:
    >>> retriever = FewShotGR(model_name="meta-llama/Llama-2-8b-chat")
    >>> retriever.index_documents(documents)
    >>> results = retriever.retrieve("What is minority interest?")

Author: Dodofk
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from jinja2 import Environment, FileSystemLoader
import hydra
from omegaconf import DictConfig


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

class HuggingFaceBackend(LLMBackend):
    """Hugging Face model backend"""
    
    def __init__(self, cfg: DictConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=torch.bfloat16
        )
        self.model.to(cfg.model.device)
        self.cfg = cfg
    
    def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.cfg.model.device)
        output_ids = self.model.generate(
            input_ids,
            **kwargs
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

class OpenAIBackend(LLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Initialize OpenAI client here
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement OpenAI API call here
        pass

class PseudoQueryGenerator:
    """Generates pseudo queries for documents"""
    
    def __init__(self, llm_backend: LLMBackend, cfg: DictConfig):
        self.llm = llm_backend
        self.cfg = cfg
        env = Environment(loader=FileSystemLoader(cfg.prompts.directory))
        self.pq_template = env.get_template(cfg.prompts.pq_template)
    
    def generate(self, document: str, n_queries: int) -> List[str]:
        prompt = self.pq_template.render(
            n_queries=n_queries,
            document=document
        )
        
        response = self.llm.generate(
            prompt,
            max_length=self.cfg.generation.max_length,
            num_return_sequences=n_queries,
            num_beams=self.cfg.generation.num_beams,
            temperature=self.cfg.generation.temperature,
            do_sample=self.cfg.generation.do_sample,
        )
        
        queries = response.split("Questions:")[-1].strip().split("\n")
        return queries[:n_queries]

class DocumentIndexer:
    """Handles document indexing and identifier generation"""
    
    def __init__(self, llm_backend: LLMBackend, cfg: DictConfig):
        self.llm = llm_backend
        self.cfg = cfg
        
        env = Environment(loader=FileSystemLoader(cfg.prompts.directory))
        self.identifier_template = env.get_template(cfg.prompts.identifier_template)
        
        self.pq_to_identifier = {}
        self.identifier_to_doc = {}
    
    def generate_identifier(self, query: str) -> str:
        prompt = self.identifier_template.render(query=query)
        
        max_attempts = 3
        for _ in range(max_attempts):
            response = self.llm.generate(
                prompt,
                max_length=self.cfg.generation.max_length,
                temperature=0.7,
                do_sample=True,
            )
            docid = response.split("Identifier:")[-1].strip()

            if self._validate_identifier_format(docid):
                return docid

        raise ValueError("Failed to generate valid identifier")
    
    def _validate_identifier_format(self, identifier: str) -> bool:
        """Validate identifier format (implement your validation logic)"""
        if not identifier:
            return False
        parts = identifier.split("-")
        return len(parts) >= 2 and all(part.isalnum() for part in parts)
    
    def index_pqs(self, doc_pqs: Dict[str, List[str]]) -> None:
        """Index documents based on their pseudo queries"""
        for doc, pqs in tqdm(doc_pqs.items(), desc="Indexing documents"):
            for pq in pqs:
                identifier = self.generate_identifier(pq)
                self.pq_to_identifier[pq] = identifier
                self.identifier_to_doc[identifier] = doc

class FewShotGR:
    """Main class for document retrieval"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Initialize LLM backend based on config
        if cfg.model.backend == "huggingface":
            llm_backend = HuggingFaceBackend(cfg)
        elif cfg.model.backend == "openai":
            llm_backend = OpenAIBackend(cfg)
        else:
            raise ValueError(f"Unsupported backend: {cfg.model.backend}")
        
        self.pq_generator = PseudoQueryGenerator(llm_backend, cfg)
        self.indexer = DocumentIndexer(llm_backend, cfg)
        self.doc_to_pqs = {}
    
    def generate_pqs(self, documents: List[str]) -> Dict[str, List[str]]:
        """Generate pseudo queries for documents"""
        doc_to_pqs = {}
        for doc in tqdm(documents, desc="Generating pseudo queries"):
            pqs = self.pq_generator.generate(
                doc, 
                self.cfg.generation.n_queries
            )
            doc_to_pqs[doc] = pqs
        return doc_to_pqs
    
    def index_documents(self, doc_to_pqs: Dict[str, List[str]]) -> None:
        """Index documents using their pseudo queries"""
        self.doc_to_pqs = doc_to_pqs
        self.indexer.index_pqs(doc_to_pqs)
    
    def process_documents(self, documents: List[str]) -> None:
        """Complete pipeline: generate PQs and index"""
        doc_to_pqs = self.generate_pqs(documents)
        self.index_documents(doc_to_pqs)
    
    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        query_identifier = self.indexer.generate_identifier(query)
        results = []
        
        # Exact match
        if query_identifier in self.indexer.identifier_to_doc:
            doc = self.indexer.identifier_to_doc[query_identifier]
            results.append((doc, 1.0))
        
        # Partial matching if needed
        if not results or len(results) < top_k:
            partial_matches = self._get_partial_matches(
                query_identifier,
                self.cfg.retrieval.similarity_threshold
            )
            results.extend(partial_matches)
        
        return results[:top_k]
    
    def _get_partial_matches(
        self, 
        query_identifier: str, 
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """Get partial matches based on identifier similarity"""
        query_parts = set(query_identifier.lower().split("-"))
        results = []

        for stored_identifier, doc in self.indexer.identifier_to_doc.items():
            stored_parts = set(stored_identifier.lower().split("-"))
            
            # Calculate Jaccard similarity
            intersection = len(query_parts & stored_parts)
            union = len(query_parts | stored_parts)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= similarity_threshold:
                results.append((doc, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)


# Example usage
@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig):
    # Initialize Few-Shot GR
    model = "meta-llama/Llama-3.1-8B-Instruct"
    # model = "mistralai/Mistral-7B-Instruct-v0.1"
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
