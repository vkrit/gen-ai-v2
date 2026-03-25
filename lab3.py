import warnings
from typing import Literal
warnings.filterwarnings("ignore",category=FutureWarning,module=r"instructor\.providers\.gemini\.client",)
import instructor
import anthropic
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to your environment or .env file.")

client = anthropic.Anthropic(api_key=api_key)

class TriageResult(BaseModel):
    level: Literal[1, 2, 3]
    reason: str
    vital_flags: list[str]
    escalate: bool

# text-only call, then parse as JSON manually; avoids dependency on instructor's response_model
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    messages=[
        {
            "role": "system",
            "content": (
                "คุณเป็นพยาบาลคัดแยกในโรงพยาบาลเอกชน\n"
                "ตอบออกมาเป็น JSON เท่านั้นโดยมีฟิลด์: level, reason, vital_flags, escalate\n"
                "level: 1 ถึง 3, escalate: true หรือ false\n"
                "ตัวอย่าง: {\"level\":1,\"reason\":\"...\",\"vital_flags\":[\"...\"],\"escalate\":true}"
            ),
        },
        {
            "role": "user",
            "content": "ผู้ป่วยชาย 60 ปี เจ็บหน้าอก แน่น เหงื่อออก มือชา 30 นาที"
        },
    ],
)

raw_text = message.content[0].text.strip()

import json, re
json_match = re.search(r"\{.*\}", raw_text, re.S)
if not json_match:
    raise RuntimeError(f"Failed to extract JSON from model output: {raw_text!r}")

parsed = json.loads(json_match.group())
result = TriageResult.parse_obj(parsed)
print(result.level, result.escalate)
