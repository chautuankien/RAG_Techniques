"""
RAG Evaluation Script

This script evaluates the performance of a RAG model using various metrics from deepeval library.
"""

from typing import Any
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# We define four evaluation metrics:
# 1. Correctness: Response vs reference answer -> measure how well the response aligns with the reference answer.
# 2. Faithfulness: Response vs retrieved docs -> measure how well the response aligns with the retrieved documents.
# 3. Retrieval Relevance: Retrieved docs vs query -> measure how well the retrieved documents align with the query.
# 4. Answer Relevance: Response vs query -> measure how well the response aligns with the query.
correctness_metric = GEval(
    name="correctness",
    model="gpt-4o-mini",
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output.",
    ],
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
)

faithfullness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)

retrieval_relevance_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)

answer_relevance_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)

def rag_evaluation(retriever, num_questions: int = 5, eval_topic: str = "climate change") -> dict[str, Any]|None:
    # Initalize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create evaluation prompt
    eval_prompt = """
    Evaluate the following retrieval results for the question.
    Question: {question}
    Retrieved Documents: {context}
    
    Rate on a scale of 1-5 (5 being the best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """
    eval_prompt_template: PromptTemplate = PromptTemplate.from_template(eval_prompt)

    # Create evaluation chain
    eval_chain = (eval_prompt_template | llm | StrOutputParser())

    # Generate test questions
    question_gen_template: PromptTemplate = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about about {topic}."
    )
    question_gen_chain = (question_gen_template | llm | StrOutputParser())
    questions: list[str] = question_gen_chain.invoke({
        "num_questions": num_questions,
        "topic": eval_topic
    }).split("\n")

    # Evaluate each question
    results = []
    for i, question in enumerate(questions):
        # Get retrieval results
        context = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context])

        # Evaluate retrieval results
        eval_result: str = eval_chain.invoke({
            "question": question,
            "context": context_text
            })
        # results.append(eval_result)

        print(f"Evaluated question {i+1}/{num_questions}:")
        print(f"Question: {question}")
        print(f"Context: {context_text}")
        print(f"Evaluation: {eval_result}")
        print("-------------------\n")
    
    # return {
    #     "questions": questions,
    #     "results": results
    # }