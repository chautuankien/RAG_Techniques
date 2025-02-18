from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI


# We define four evaluation metrics:
# 1. Correctness: Response vs reference answer -> measure how well the response aligns with the reference answer.
# 2. Faithfulness: Response vs retrieved docs -> measure how well the response aligns with the retrieved documents.
# 3. Retrieval Relevance: Retrieved docs vs query -> measure how well the retrieved documents align with the query.
# 4. Answer Relevance: Response vs query -> measure how well the response aligns with the query.

class CorrectnessGrade(TypedDict):
    explaination: Annotated[str, ..., "Exaplain the reasoning behind the grade"]
    correct: Annotated[bool, ..., "True if the response is correct, False otherwise"]
    
def correctness(inputs: dict, outputs: dict, ref_outputs: dict) -> bool:
    correctness_prompt = """
    You will be given a QUESTION, a GROUND TRUTH (correct) ANSWER, and a RESPONSE.
    Here is the evaluation criteria to follow:
    (1) Grade the RESPONSE based ONLY on their factual accuracy relative to the GROUND TRUTH ANSWER.
    (2) Ensure that the RESPONSE does not contain any conflicting statements.
    (3) It is OK if the RESPONSE contains more information than the GROUND TRUTH ANSWER, as long as it is factually relative to the GROUND TRUTH ANSWER.

    Correctness:
    A correctness of True means that the RESPONSE meets all the criteria.
    A correctness of False means that the RESPONSE does not meet all the criteria.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
    Avoid simply stating the correct answer at the outset.
    """

    grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

    context: str = f"""
    QUESTION: {inputs['question']}
    GROUND TRUTH ANSWER: {ref_outputs['answer']}
    RESPONSE: {outputs['answer']}
    """

    # Run evaluator
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_prompt},
        {"role": "user", "content": context}
    ])

    return grade["correct"]

class QA_generator(TypedDict):
    question: Annotated[str, ..., "The question generated from the context."]
    answer: Annotated[str, ..., "The answer to the question."]

def qa_generator(context: str) -> dict[str, str]:
    """Generate a question and answer pair from a given context."""
    instruct_prompt = """
    You are an expert evaluator specialized in creating high-quality question-answer pairs for information retrieval and reading comprehension tasks. 
    Your goal is to generate diverse, natural, and challenging questions that test understanding of the given context.

    Follow these principles:
    1. Generate questions that require precise understanding of the context
    2. Vary question types (what, why, how, when, etc.)
    3. Target different cognitive levels (recall, understanding, analysis)
    4. Ensure questions are unambiguous and have clear, verifiable answers
    """

    generation_prompt = """"
    Create a natural and challenging question-answer pair from the given context that would be valuable for evaluating language models.

    Question Requirements:
    1. Natural Language: Frame the question as a real user would ask on a search engine or conversational AI
    2. Specificity: Question must be answerable with specific information from the context
    3. Clarity: Question should be unambiguous with only one correct interpretation
    4. Depth: Prefer questions that require understanding rather than simple keyword matching
    5. Independence: Question must stand alone without referencing the context or source
    6. Conciseness: Both question and answer should be concise and to the point

    Answer Requirements:
    1. Accuracy: Must be supported by the context
    2. Completeness: Include all necessary information to fully answer the question
    3. Conciseness: Exclude irrelevant details
    4. Self-contained: Answer should make sense without needing additional context

    AVOID:
    - Questions that can be answered without the context
    - Questions with multiple possible correct answers
    - Questions that require information beyond the context
    - Meta-references like "according to the passage" or "in the context"
    - Yes/no questions unless they test critical understanding
    - Overly simple questions that only require word matching

    Output Format:
    Question: <your question>
    Answer: <your answer>

    Context:
    {context}
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, top_p=0.99).with_structured_output(QA_generator)
    outputs = llm.invoke([
        {"role": "system", "content": instruct_prompt},
        {"role": "user", "content": generation_prompt.format(context=context)}
    ])

    return outputs