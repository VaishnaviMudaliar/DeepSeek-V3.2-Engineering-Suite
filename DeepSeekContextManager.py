import uuid

class DeepSeekContextManager:
    """
    Implements DeepSeek-V3.2 Thinking Context Management for tool-calling.
    Goal: Retain reasoning history during tool calls to prevent redundant 
    re-reasoning, while pruning it when a new user turn starts.
    """
    
    def __init__(self):
        self.conversation_history = []
        # Marks if the current 'turn' is still part of the same user inquiry
        self.is_within_tool_loop = False

    def add_message(self, role, content, has_thinking=False):
        """
        Adds a message to the history while applying pruning logic.
        """
        if role == "user":
            # New user message arrived: Prune historical thinking from the context
            # as per DeepSeek-V3.2 rules to maintain efficiency.
            self._prune_thinking_traces()
            self.is_within_tool_loop = False
        
        elif role == "tool":
            # Tool results are appended: Ensure the next thinking turn is retained.
            self.is_within_tool_loop = True

        message = {
            "role": role, 
            "content": content, 
            "has_thinking": has_thinking,
            "id": str(uuid.uuid4())
        }
        self.conversation_history.append(message)

    def _prune_thinking_traces(self):
        """
        Removes <think> blocks from past assistant messages. 
        DeepSeek-V3.2 keeps the tool results but discards the reasoning 
        steps when the next round of user messages arrive.
        """
        for msg in self.conversation_history:
            if msg["role"] == "assistant" and msg["has_thinking"]:
                # Logic: Keep the final answer/tool call but remove the 
                # <think>...</think> content to save tokens.
                msg["content"] = self._strip_think_tags(msg["content"])
                msg["has_thinking"] = False

    def _strip_think_tags(self, content):
        import re
        # Pattern to remove DeepSeek style thinking tags
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    def get_context_for_llm(self):
        """Returns the formatted history ready for the API call."""
        return [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]

# --- Example Usage for your Portfolio ---

manager = DeepSeekContextManager()

# 1. User asks a question
manager.add_message("user", "Find a flight to杭州 (Hangzhou) and book the cheapest one.")

# 2. Assistant reasons and calls a tool
assistant_output = "<think>The user wants a flight to Hangzhou. I need to list flights first.</think> [Tool Call: get_flights(city='Hangzhou')]"
manager.add_message("assistant", assistant_output, has_thinking=True)

# 3. Tool returns result (Reasoning is RETAINED here)
manager.add_message("tool", "Flight A: 500 CNY, Flight B: 800 CNY")

# 4. Assistant reasons again (Context still has the previous <think> block)
# This prevents 'redundant re-reasoning' as highlighted in the paper.
next_output = "<think>Flight A is cheaper at 500 CNY. Proceeding to book.</think> [Tool Call: book_flight(id='A')]"
manager.add_message("assistant", next_output, has_thinking=True)

# 5. NEW User message arrives (Now we PRUNE old thinking to save tokens)
manager.add_message("user", "Also find a hotel nearby.")

print(f"Messages in context: {len(manager.get_context_for_llm())}")
print(f"Last Assistant Thought Sample: {manager.conversation_history[1]['content']}") # Should be stripped
