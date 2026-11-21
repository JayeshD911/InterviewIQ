# Previous import retained for reference:
# from agent import run_evaluation, run_turn
from agent import run_turn


question = (
    "Tell me about a time you had to debug a difficult "
    "production issue."
)

answer = (
    "When our checkout service failed at 2am, I led an investigation "
    "and checked the logs. I found the root cause was a bad deployment "
    "and performed a rollback with my team. As a result, service was "
    "restored within 20 minutes."
)

expected_keywords = [
    "logs",
    "root cause",
    "monitoring",
    "rollback",
    "team",
]


from agent import session_log


def main():
    feedback = run_turn(question, answer, expected_keywords)

    print("Coach feedback:")
    print(feedback)

    print("\nStored session memory:")
    print(session_log)


if __name__ == "__main__":
    main()