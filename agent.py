import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools import (
    detect_filler_words,
    check_star_structure,
    score_relevance,
)


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "detect_filler_words",
            "description": "Count filler words in an interview answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_star_structure",
            "description": "Check Situation, Task, Action, and Result coverage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_relevance",
            "description": "Compare an answer with expected interview concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "expected_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["answer", "expected_keywords"],
            },
        },
    },
]

SYSTEM_PROMPT = """
You are InterviewIQ, an encouraging mock-interview coach.

For every candidate answer, you must call all three tools:
1. detect_filler_words
2. check_star_structure
3. score_relevance

After receiving the tool results:
- Report the exact relevance score out of 100.
- Mention covered and missed concepts.
- Mention filler-word findings.
- Mention missing STAR stages when applicable.
- Keep the feedback short, specific, and encouraging.
"""



TOOL_FUNCTIONS = {
    "detect_filler_words": lambda arguments: detect_filler_words(
        arguments["answer"]
    ),
    "check_star_structure": lambda arguments: check_star_structure(
        arguments["answer"]
    ),
    "score_relevance": lambda arguments: score_relevance(
        arguments["answer"],
        arguments["expected_keywords"],
    ),
}


MAX_TOOL_ROUNDS = 5
session_log = []


def run_turn(question: str, answer: str, expected_keywords: list) -> str:
    results = {}

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Candidate answer: {answer}\n"
                f"Expected keywords: {expected_keywords}\n"
                "You must call all three tools: detect filler words, "
                "check STAR structure, and score relevance. "
                "Then provide short feedback including the exact relevance score."
            ),
        },
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            feedback = message.content
            session_log.append({
                "question": question,
                "answer": answer,
                "results": results,
            })
            return feedback

        messages.append(message)

        for tool_call in message.tool_calls:
            arguments = (
                json.loads(tool_call.function.arguments)
                if tool_call.function.arguments
                else {}
            )

            function_name = tool_call.function.name
            result = TOOL_FUNCTIONS[function_name](arguments)

            results[function_name] = result

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    feedback = "I could not complete the evaluation."

    session_log.append({
        "question": question,
        "answer": answer,
        "results": results,
    })

    return feedback