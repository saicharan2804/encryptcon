"""Module for a Retrieval-Augmented Generation pipeline for ranking projects based on user preferences."""
from typing import List
from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import (
    PromptNode,
    PromptTemplate,
    BM25Retriever,
    SentenceTransformersRanker,
)
from haystack import Pipeline
import torch


class RAGPipeline:
    def __init__(
        self,
        documents: List[str],
        retriever_top_k: int = 100,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        prompt_model: str = "MBZUAI/LaMini-Flan-T5-783M",
        ranker_top_k: int = 10,
    ):
        """
        Initializes a Retrieval-Augmented Generation pipeline.

        This pipeline integrates with a Variational Autoencoder (VAE) and Language Models (LLMs) to assess
        and rank various projects. It evaluates user preferences, ranks projects according to relevance, and
        provides risk assessments and scoring for each project using the VAE and LLMs.

        Args:
            documents (List[str]): A list of project descriptions or relevant documents.
            retriever_top_k (int, optional): The number of top documents to retrieve for initial filtering. Defaults to 100.
            reranker_model (str, optional): Model path for Sentence Transformer Ranker, used for fine-grained ranking. Defaults to "cross-encoder/ms-marco-MiniLM-L-12-v2".
            prompt_model (str, optional): Model path for the prompt node, integrated with LLMs for detailed analysis. Defaults to "MBZUAI/LaMini-Flan-T5-783M".
            ranker_top_k (str, optional): Number of top documents to rank. Defaults to 5.
        """
        self.document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
        self._write_documents_to_store(documents)
        self.retriever = BM25Retriever(
            document_store=self.document_store, top_k=retriever_top_k
        )
        self.reranker = SentenceTransformersRanker(
            model_name_or_path=reranker_model, top_k=1
        )
        self.prompt = self._create_prompt_node(prompt_model)
        self.pipeline = self._create_pipeline()

    def _write_documents_to_store(self, documents: List[str]):
        """
        Writes the provided documents to the in-memory document store.

        Args:
            documents (List[str]): List of documents to be written.
        """
        doc_objects = [Document(content=doc, id=i) for i, doc in enumerate(documents)]
        self.document_store.write_documents(doc_objects)

    def _create_prompt_node(self, model_name_or_path: str) -> PromptNode:
        """
        Creates a PromptNode with the given model for processing and analyzing the documents.

        Args:
            model_name_or_path (str): Model path for the prompt node.

        Returns:
            PromptNode: The configured prompt node.
        """
        lfqa_prompt = PromptTemplate(
            prompt="Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words. \n\n Context: {join(documents)} \n\n Question: {query} \n\n Answer:",
            output_parser={"type": "AnswerParser"},
        )
        return PromptNode(
            model_name_or_path=model_name_or_path,
            default_prompt_template=lfqa_prompt,
            model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16},
        )

    def _create_pipeline(self) -> Pipeline:
        """
        Creates the RAG pipeline integrating retriever, reranker, and prompt nodes.

        Returns:
            Pipeline: The configured RAG pipeline.
        """
        p = Pipeline()
        p.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        p.add_node(component=self.reranker, name="Reranker", inputs=["Retriever"])
        p.add_node(component=self.prompt, name="PromptNode", inputs=["Reranker"])
        return p

    def run(self, query: str):
        """
        Runs the RAG pipeline with a user query to rank projects based on preferences.

        The query is processed to retrieve, rank, and analyze projects, returning a list of top projects
        with their respective scores and risk assessments.

        Args:
            query (str): User's preferences or inquiry about projects.

        Returns:
            A ranked list of projects with scores and risk evaluations.
        """
        return self.pipeline.run(query)


# Example usage:
documents = [
    "Document 1 content",
    "Document 2 content",
    ...,
]  # Replace with actual document strings
rag_pipeline = RAGPipeline(documents)
result = rag_pipeline.run("What is the content of Document 1?")
print(result)
