# DeepSeek-V3.2 Thinking Context Manager 🧠🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of the **Thinking Context Management** system introduced in the [DeepSeek-V3.2 technical report](https://arxiv.org/abs/2512.02556v1). This tool optimizes multi-turn agentic workflows by managing reasoning persistence, drastically reducing token waste and preventing "redundant re-reasoning".

---

## 🚀 The Problem: Token Inefficiency in Agents
In standard agentic frameworks, reasoning traces (like DeepSeek's `<think>` blocks) are often discarded as soon as a new message—even a tool output—is added to the context[. This forces the model to re-reason through the entire problem for every subsequent tool call, leading to:
* **High Latency:** Models spend time regenerating logic they already "thought" through.
* **Context Bloat:** Redundant traces fill up the context window.
* **Inconsistent Behavior:** Repeated reasoning increases the chance of logical drift.

## 🛠️ The Solution: DeepSeek-V3.2 Logic
This repository implements the **DeepSeek-V3.2 Thinking Context Management** rules:
1.  **Thinking Persistence:** Reasoning content is **retained** throughout the interaction if only tool-related messages (e.g., tool outputs) are appended.
2.  **Selective Pruning:** Historical reasoning traces are **discarded only** when a new **user message** is introduced to the conversation.
3.  **Context Integrity:** When reasoning is removed, the history of tool calls and their results remains preserved, ensuring the agent stays grounded.

## 💻 Implementation Highlights
The core of this project is a state-managed context handler that intercepts conversation turns and applies the DeepSeek-V3.2 pruning logic.

```python
# Rule: Prune historical <think> blocks only when a new user turn starts
def add_message(self, role, content):
    if role == "user":
        self._prune_historical_thinking()
    
    self.messages.append({"role": role, "content": content})
```

### 📊 Key Research Insights (DeepSeek-V3.2)

* **DeepSeek Sparse Attention (DSA):** This implementation is designed to work alongside DSA, which reduces core attention complexity from $O(L^2)$ to $O(Lk)$, enabling efficient 128K token context processing.
* **Agentic Performance:** By maintaining reasoning persistence, DeepSeek-V3.2 achieves parity with Gemini-3.0-Pro on agentic benchmarks like **Terminal Bench 2.0** and **Tool Decathlon**.
* **Scalable RL:** The model uses **Group Relative Policy Optimization (GRPO)** to optimize reward signals across diverse tasks including search, code engineering, and general planning.

### 🎯 Use Case: Smart Trip Planning
As highlighted in the paper, this system is ideal for complex, multi-constraint tasks like **Trip Planning**.

* **Scenario:** Managing a 3-day itinerary with cascading budget constraints (e.g., if a luxury hotel > 800 CNY is booked, dining must be < 350 CNY).
* **Advantage:** The agent can reason through hotel price impacts and retain that plan while calling weather and booking APIs, rather than re-calculating the budget after every tool result.

### 📄 License
This project is licensed under the MIT License.

---
*Based on the research paper: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models" (DeepSeek-AI, 2025).*
