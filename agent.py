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
    {
    "type": "function",
    "function": {
        "name": "generate_final_report",
        "description": (
            "Aggregate every answer in the current session, including "
            "average relevance and the weakest question."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
},
]

SYSTEM_PROMPT = """
You are InterviewIQ, an encouraging mock-interview coach.

For every candidate answer:
1. Call detect_filler_words.
2. Call check_star_structure.
3. Call score_relevance.

If the candidate asks how they are doing, asks for their weakest area,
or requests an overall report, call generate_final_report and answer
using the complete session data.

Keep feedback short, specific, and encouraging.
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
    "generate_final_report": lambda arguments: generate_final_report(),
}


MAX_TOOL_ROUNDS = 5
session_log = []


def generate_final_report():
    if not session_log:
        return "No answers have been scored yet."

    relevance_scores = []
    total_fillers = 0
    lines = [f"Interview Report - {len(session_log)} question(s) answered"]

    for index, entry in enumerate(session_log, 1):
        results = entry["results"]

        lines.append(f"\nQ{index}: {entry['question']}")

        filler = results.get("detect_filler_words")

        if filler:
            total_fillers += filler["total"]
            lines.append(
                f"- Fillers: {filler['total']} ({filler['verdict']})"
            )

        star = results.get("check_star_structure")

        if star:
            if star["all_present"]:
                lines.append("- STAR: all stages present")
            else:
                lines.append(
                    "- STAR: missing "
                    + ", ".join(star["missing"])
                )

        relevance = results.get("score_relevance")

        if relevance:
            score = relevance["score"]
            missed = relevance["missed"] or ["none"]

            relevance_scores.append((score, entry["question"]))

            lines.append(f"- Relevance: {score}/100")
            lines.append(f"- Missed concepts: {', '.join(missed)}")

    if relevance_scores:
        average = round(
            sum(score for score, _ in relevance_scores)
            / len(relevance_scores)
        )

        weakest_score, weakest_question = min(
            relevance_scores,
            key=lambda item: item[0],
        )

        lines.append(f"\nAverage relevance: {average}/100")
        lines.append(
            f"Weakest area: \"{weakest_question}\" "
            f"(relevance {weakest_score}/100)"
        )

    lines.append(
        f"Total filler words across the session: {total_fillers}"
    )

    return "\n".join(lines)


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

def ask_agent(message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        model_message = response.choices[0].message

        if not model_message.tool_calls:
            return model_message.content

        messages.append(model_message)

        for tool_call in model_message.tool_calls:
            arguments = (
                json.loads(tool_call.function.arguments)
                if tool_call.function.arguments
                else {}
            )

            function_name = tool_call.function.name
            result = TOOL_FUNCTIONS[function_name](arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    result
                    if isinstance(result, str)
                    else json.dumps(result)
                ),
            })

    return "I could not complete the session summary."


def ask_for_final_report():
    return generate_final_report()