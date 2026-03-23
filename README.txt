# ============================================================
# Generative AI using Python — 2026
# Workshop Source Code
# ============================================================
#
# Directory structure:
#   day1_lab/   cli_assistant.py        — streaming CLI with cost tracking
#               day1_bonus.py           — tokenisation benchmark + embeddings
#   day2_lab/   lab2a_prompt_duel.py    — prompt evaluation & injection defense
#               lab2b_thai_rag.py       — Thai Legal RAG (hybrid search + RAGAS)
#   day3_lab/   lab3_mini_llm.py        — MiniLLM from scratch + Thai training
#   day4_lab/   lab4_finetune.py        — QLoRA fine-tuning + DPO alignment
#   day5_lab/   lab5_agent.py           — Hospital agent (tools + LangGraph + FastAPI)
#   day6_lab/   lab6_production.py      — LLMOps, guardrails, PDPA, benchmarks
#
# ============================================================
# QUICK START
# ============================================================
#
# 1. Install uv (fast Python package manager):
#    curl -LsSf https://astral.sh/uv/install.sh | sh
#
# 2. Create project and install:
#    uv init genai-course && cd genai-course
#    uv add -r requirements.txt
#
# 3. Set API keys in .env:
#    ANTHROPIC_API_KEY=sk-ant-...
#    OPENAI_API_KEY=sk-...           (optional)
#    TYPHOON_API_KEY=...             (optional)
#
# 4. Run labs:
#    python day1_lab/cli_assistant.py --question "What is an LLM?"
#    python day2_lab/lab2b_thai_rag.py
#    python day3_lab/lab3_mini_llm.py
#    python day5_lab/lab5_agent.py --mode test
#    python day6_lab/lab6_production.py --part all
#
# ============================================================
# SERVICES NEEDED
# ============================================================
#
# Day 2 Lab B (RAG):
#   docker run -p 6333:6333 qdrant/qdrant
#
# Day 5 Lab (Agent API mode):
#   python day5_lab/lab5_agent.py --mode api
#   streamlit run day5_lab/dashboard.py   (see Day 6 notes)
#
# Day 6 Lab (Langfuse tracing):
#   docker compose up -d   (uses docker-compose.yml from Day 6 notes)
#
# ============================================================
