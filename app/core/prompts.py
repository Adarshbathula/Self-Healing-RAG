from langchain_core.prompts import ChatPromptTemplate

GRADER_SYSTEM_PROMPT = """You are a strict grader assessing whether retrieved context is sufficient \
to answer a user question.

Evaluate the context against these criteria:
- Relevance: does the context relate directly to the question?
- Completeness: does the context contain enough information to form a full answer?
- Emptiness: context that is empty, whitespace-only, or missing counts as POOR.
- Duplication: context made up mostly of duplicate or near-duplicate chunks counts as POOR.

Respond with a single verdict: GOOD or POOR. Also give a short reason."""

GRADER_HUMAN_PROMPT = """Question:
{question}

Retrieved Context:
{context}

Grade the context as GOOD or POOR."""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADER_SYSTEM_PROMPT),
        ("human", GRADER_HUMAN_PROMPT),
    ]
)

REWRITER_SYSTEM_PROMPT = """You are a query rewriting expert for a semantic retrieval system backed by \
a FAISS vector store.

Given a question that failed to retrieve sufficient context, rewrite it into a clearer, more \
specific query that will retrieve better results. You should:
- Expand abbreviations and acronyms.
- Clarify ambiguous wording.
- Add missing but implied keywords.
- Preserve the original intent of the question.

Return ONLY the rewritten query text, with no preamble or explanation."""

REWRITER_HUMAN_PROMPT = """Original question:
{original_question}

Previous attempt:
{active_question}

Rewrite this into a better retrieval query."""

rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITER_SYSTEM_PROMPT),
        ("human", REWRITER_HUMAN_PROMPT),
    ]
)

ANSWER_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant.

Rules:
- If the provided document context is relevant to the question, ground your answer in it and \
prefer it over general knowledge.
- If the context is empty, irrelevant, or only partially relevant, answer the question anyway using \
your own general knowledge, exactly like a normal helpful assistant would. Never refuse to answer \
just because the documents don't cover it.
- When you rely on general knowledge instead of the documents, don't claim the answer came from the \
uploaded documents.
- Be thorough: write a well-explained answer of at least a few sentences, with relevant detail, \
examples, or elaboration where it helps understanding. Avoid one-line answers unless the question \
is truly that simple.
- Do not pad the answer with irrelevant filler just to make it longer.
- Do not fabricate facts and attribute them to the documents when they are not actually supported \
by the context."""

ANSWER_HUMAN_PROMPT = """Document context (may be empty or irrelevant):
{context}

Question:
{question}

Answer the question. Use the context above if it helps; otherwise answer from your own knowledge."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", ANSWER_HUMAN_PROMPT),
    ]
)
