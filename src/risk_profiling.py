"""
risk_profiling.py
------------------
A simplified risk-profiling questionnaire, in the spirit of the
know-your-client (KYC) assessments used in wealth management to map a
client onto a risk profile before constructing a portfolio.

This is intentionally simple (3 questions) to keep the logic transparent
and fully explainable — a real KYC questionnaire used by a bank is longer
and regulated (e.g. MiFID II suitability requirements in the EU), but the
scoring principle is the same: horizon, loss tolerance, and objective
combine into a single risk score.
"""

QUESTIONS = [
    {
        "text": "What is your investment horizon?",
        "options": [
            ("Less than 3 years", 1),
            ("Between 3 and 7 years", 2),
            ("More than 7 years", 3),
        ],
    },
    {
        "text": "If your portfolio lost 20% of its value in a few months, you would:",
        "options": [
            ("Sell everything to avoid further losses", 1),
            ("Sell part of the portfolio", 2),
            ("Hold, or consider buying more", 3),
        ],
    },
    {
        "text": "What is your primary objective?",
        "options": [
            ("Preserve capital", 1),
            ("Balanced growth with moderate risk", 2),
            ("Maximize long-term growth", 3),
        ],
    },
]

PROFILE_MAP = {
    range(3, 5): "Conservative",
    range(5, 8): "Balanced",
    range(8, 10): "Aggressive",
}


def score_to_profile(score):
    for score_range, profile in PROFILE_MAP.items():
        if score in score_range:
            return profile
    return "Balanced"  # default fallback, should not trigger given 3-9 range


def run_questionnaire(interactive=True, answers=None):
    """
    If interactive=True, prompts the user via input().
    If interactive=False, `answers` must be a list of 3 integers (1-3),
    one per question, in order — useful for scripted/reproducible runs.
    Returns (profile, total_score, answers_used).
    """
    if interactive:
        collected = []
        print("=== Risk Profiling Questionnaire ===\n")
        for q in QUESTIONS:
            print(q["text"])
            for i, (label, _) in enumerate(q["options"], start=1):
                print(f"  {i}. {label}")
            while True:
                try:
                    choice = int(input("Your choice (1-3): "))
                    if choice in (1, 2, 3):
                        break
                except ValueError:
                    pass
                print("Please enter 1, 2, or 3.")
            collected.append(q["options"][choice - 1][1])
            print()
        answers_used = collected
    else:
        if answers is None or len(answers) != len(QUESTIONS):
            raise ValueError("Provide exactly 3 answers (1-3) when interactive=False.")
        answers_used = answers

    total_score = sum(answers_used)
    profile = score_to_profile(total_score)
    return profile, total_score, answers_used


if __name__ == "__main__":
    # Non-interactive example run
    profile, score, answers = run_questionnaire(interactive=False, answers=[3, 2, 2])
    print(f"Score: {score}/9 -> Profile: {profile}")
