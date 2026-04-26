"""Hand-labeled evaluation set for the RAG bot.

Each example is a (question, expected_answer_summary, expected_sources) tuple.
The expected_answer_summary is what a correct answer should *contain* — not
verbatim, but the key facts. The judge LLM scores partial credit.

Curated from the actual knowledge base (data/*.md) so all answers are
grounded in retrievable content. Add more examples to make the eval more
robust; ~10 is enough to spot regressions.
"""

EVAL_SET = [
    {
        "id": "office_hours_time",
        "question": "When are office hours scheduled?",
        "expected_answer": "Office hours are on Sundays from 10:00 AM to 11:00 AM EST.",
        "expected_sources": ["intern_faq.md"],
    },
    {
        "id": "rag_definition",
        "question": "What is RAG?",
        "expected_answer": "RAG stands for Retrieval-Augmented Generation. It is a technique where relevant documents are retrieved from a knowledge base and provided as context to an LLM to generate grounded answers.",
        "expected_sources": ["training.md", "intern_faq.md", "project_assignment.md"],
    },
    {
        "id": "submission_deadline",
        "question": "When is the weeks 1-3 assignment due?",
        "expected_answer": "The weeks 1-3 assignment is due at 11pm EST on Sunday April 26th.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "backend_role",
        "question": "What does a Backend Engineer do in this project?",
        "expected_answer": "The Backend Engineer builds APIs, handles deployment with Docker, sets up logging and observability, and defines API contracts. If there is no Frontend partner, they also implement the Discord bot logic.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "data_scientist_role",
        "question": "What are the responsibilities of a Data Scientist on this project?",
        "expected_answer": "The Data Scientist builds the RAG pipeline including chunking, embeddings, vector search, retrieval logic, prompt construction, and the LLM call. They choose models and vector stores.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "submission_parts",
        "question": "What do I need to submit for the weeks 1-3 project?",
        "expected_answer": "The submission has three parts: a design document, a GitHub repository shared with community@pmaccelerator.io, and a 2-4 minute video walkthrough.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "api_key_security",
        "question": "How should API keys be handled in the project?",
        "expected_answer": "API keys must be stored in environment files (.env) and must never be hard-coded or pushed to GitHub. The .env file must be added to .gitignore.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "recommended_llm",
        "question": "What LLM does the assignment recommend?",
        "expected_answer": "The assignment recommends DeepSeek R1 via Microsoft Azure AI Foundry, which is free for new Azure customers with $200 in credit.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "vector_store_options",
        "question": "What vector store options are recommended?",
        "expected_answer": "MongoDB Atlas Vector Search is recommended. Alternatives include Faiss, Pinecone, Weaviate, Milvus, or in-memory NumPy for prototyping.",
        "expected_sources": ["project_assignment.md"],
    },
    {
        "id": "out_of_scope",
        "question": "What is the most famous observatory in Los Angeles?",
        "expected_answer": "REFUSAL_EXPECTED",  # Special marker — the bot SHOULD refuse this.
        "expected_sources": [],
    },
]
