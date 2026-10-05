# AgentOps — Lab 1: Full tracing setup (Arize Phoenix)

Агент з теми 12 (LangGraph ReAct + MCP stdio, Ollama `qwen2.5:7b`) із трейсингом у Phoenix.

```bash
docker compose up -d                       # Phoenix UI: http://localhost:6006
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_scenarios.py  # 6 сценаріїв -> 6 traces у проєкті retention-agent
.venv/bin/python scripts/cost_report.py    # docs/lab1/cost_dashboard.png + CSV
```

- Tracing: [agent/tracing.py](agent/tracing.py) (`phoenix.otel.register`, автоінструментація LangChain/LangGraph і MCP); кореневий span `agent.run` з атрибутом `script.name`.
- Проєкт: http://localhost:6006/projects (назва `retention-agent`).
- Вартість: Ollama безкоштовна, тому `$/run` рахується за **умовним тарифом Claude Haiku 4.5** ($1 / 1M вхідних, $5 / 1M вихідних токенів) з токенів LLM-спанів.
- Дашборд: [docs/lab1/cost_dashboard.png](docs/lab1/cost_dashboard.png) — середній `$/run` ≈ $0.0014, latency по спанах і tool-ах.
# notes
