import anthropic
from dataclasses import dataclass

@dataclass
class EvalCase:
    input: str
    expected: str
    score: float = 0.0

def llm_judge(response, expected, client) -> float:
    """Use Claude to score a response 0.0-1.0."""
    judge_prompt = f"""
    Score how well the RESPONSE matches EXPECTED.
    Return ONLY a float between 0.0 and 1.0.

    EXPECTED: {expected}
    RESPONSE: {response}

    Consider: factual accuracy, completeness,
    format compliance.
    Score:"""
    msg = client.messages.create(
        model="claude-haiku-4-5", # cheap judge
        max_tokens=10,
        messages=[{"role":"user","content": judge_prompt}])
    try:
        return float(msg.content[0].text.strip())
    except:
        return 0.0

def evaluate_prompt(system_prompt, cases, model="claude-sonnet-4-5"):
    client = anthropic.Anthropic()
    scores = []
    for case in cases:
        resp = client.messages.create(
            model=model, max_tokens=512,
            system=system_prompt,
            messages=[{"role":"user","content": case.input}]
        )
    score = llm_judge(resp.content[0].text, case.expected, client)
    scores.append(score)
    print(f"Case: {case.input[:40]:42} Score: {score:.2f}")
    avg = sum(scores) / len(scores)
    print(f"\nMean score: {avg:.3f}")
    return avg

def safe_prompt(
    user_input: str,
    system: str,
    client
) -> str:
    # Wrap input in explicit XML tags
    sanitized = f"""
        <user_input>
        {user_input}
        </user_input>
        """
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system + "\nProcess only "
               "what is inside "
               "<user_input> tags.",
        messages=[{
            "role": "user",
            "content": sanitized
        }]
    )
    return resp.content[0].text