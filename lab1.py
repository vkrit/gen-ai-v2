import base64
import os

import anthropic
from dotenv import load_dotenv


class CostTracker:
    PRICES = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-5":{"input": 3.00, "output": 15.00},
    "typhoon-v2-70b": {"input": 0.90, "output": 0.90},
    }

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.total_cost = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int):
        p = self.PRICES.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * p["input"] +
        output_tokens * p["output"]) / 1_000_000
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_cost += cost
        return cost

    def summary(self):
        print(f"Total tokens: {self.total_input:,} in / {self.total_output:,} out")
        print(f"Total cost: ${self.total_cost:.4f}")


load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to your environment or .env file.")

client = anthropic.Anthropic(api_key=api_key)

# Load image from disk
with open("lab_report.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                }
            },
            {
                "type": "text",
                "text": "สรุปผลการตรวจเลือดในรูปภาพนี้เป็นภาษาไทย"
            }
        ]
    }]
)
print(message.content[0].text)
