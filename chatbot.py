"""
TechBot - Rule-Based Chatbot
TechGuide AI Internship | Project 1

Run in terminal :  python chatbot.py
Run with a window:  python chatbot.py --gui

How it works
1. The user's message is cleaned (lowercase, punctuation removed).
2. Each rule's regex patterns are checked in order; the first match wins.
3. The rule either picks a random canned reply or calls a handler function
   (for dynamic answers such as time, math, or remembering the user's name).
4. If nothing matches, a fallback reply is returned.
"""

from __future__ import annotations

import argparse
import ast
import operator
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

BOT_NAME = "TechBot"
LOG_FILE = "chat_log.txt"


# --------------------------------------------------------------------------
# Conversation memory
# --------------------------------------------------------------------------
@dataclass
class Context:
    """Stores things the bot should remember during the conversation."""
    user_name: Optional[str] = None
    last_intent: Optional[str] = None
    finished: bool = False
    history: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Safe calculator (never use plain eval on user input)
# --------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    return _eval(ast.parse(expression, mode="eval"))


# --------------------------------------------------------------------------
# Handlers for dynamic replies
# --------------------------------------------------------------------------
def handle_greeting(m: re.Match, ctx: Context) -> str:
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    name = f", {ctx.user_name}" if ctx.user_name else ""
    return random.choice([
        f"Hello{name}! Good {part}. How can I help you today?",
        f"Hey{name}! Nice to see you. What can I do for you?",
        f"Hi{name}! I'm {BOT_NAME}. Ask me anything or type 'help'.",
    ])


def handle_set_name(m: re.Match, ctx: Context) -> str:
    ctx.user_name = m.group(1).capitalize()
    return f"Nice to meet you, {ctx.user_name}! I'll remember your name."


def handle_ask_user_name(m: re.Match, ctx: Context) -> str:
    if ctx.user_name:
        return f"Your name is {ctx.user_name}, of course!"
    return "I don't know your name yet. Tell me by saying 'My name is ...'."


def handle_bot_name(m: re.Match, ctx: Context) -> str:
    return f"I'm {BOT_NAME}, a rule-based chatbot built with Python."


def handle_time(m: re.Match, ctx: Context) -> str:
    return f"The current time is {datetime.now().strftime('%I:%M %p')}."


def handle_date(m: re.Match, ctx: Context) -> str:
    return f"Today is {datetime.now().strftime('%A, %d %B %Y')}."


def handle_calc(m: re.Match, ctx: Context) -> str:
    expr = m.group(1).replace("^", "**").strip()
    if not re.search(r"\d\s*[\+\-\*/%]|\*\*", expr):
        return "Give me an expression, for example: calculate 12 * (3 + 4)"
    try:
        result = safe_eval(expr)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 6)
        return f"{expr} = {result}"
    except ZeroDivisionError:
        return "I can't divide by zero."
    except Exception:
        return "Sorry, I couldn't evaluate that expression."


def handle_feeling(m: re.Match, ctx: Context) -> str:
    feeling = m.group(1)
    positive = {"happy", "great", "good", "fine", "awesome", "excited"}
    if feeling in positive:
        return f"That's wonderful to hear! Glad you're feeling {feeling}."
    return (f"I'm sorry you're feeling {feeling}. "
            "Maybe a short break or a chat with a friend will help.")


def handle_goodbye(m: re.Match, ctx: Context) -> str:
    ctx.finished = True
    name = f", {ctx.user_name}" if ctx.user_name else ""
    return random.choice([
        f"Goodbye{name}! Have a great day.",
        f"See you later{name}! Take care.",
    ])


def handle_help(m: re.Match, ctx: Context) -> str:
    return (
        "Here is what I can do:\n"
        "  - Greet you and remember your name  (try: 'My name is Aman')\n"
        "  - Tell the time and date            (try: 'What time is it?')\n"
        "  - Do maths                          (try: 'calculate 25 * 4 + 10')\n"
        "  - Tell jokes and fun facts          (try: 'Tell me a joke')\n"
        "  - Answer questions about AI         (try: 'What is AI?')\n"
        "  - Say goodbye                       (try: 'bye')"
    )


# --------------------------------------------------------------------------
# Rule table (ORDER MATTERS: first match wins)
# --------------------------------------------------------------------------
@dataclass
class Rule:
    name: str
    patterns: list
    responses: Optional[list] = None
    handler: Optional[Callable[[re.Match, Context], str]] = None

    def __post_init__(self):
        self.compiled = [re.compile(p) for p in self.patterns]

    def match(self, text: str) -> Optional[re.Match]:
        for pattern in self.compiled:
            found = pattern.search(text)
            if found:
                return found
        return None


RULES = [
    Rule("goodbye",
         [r"\b(bye|goodbye|see you|quit|exit|good night)\b"],
         handler=handle_goodbye),

    Rule("set_name",
         [r"\b(?:my name is|call me|i am called)\s+([a-z]+)"],
         handler=handle_set_name),

    Rule("ask_user_name",
         [r"\bwhat(?:s| is) my name\b", r"\bdo you know my name\b"],
         handler=handle_ask_user_name),

    Rule("bot_name",
         [r"\byour name\b", r"\bwho are you\b", r"\bwhat are you\b"],
         handler=handle_bot_name),

    Rule("how_are_you",
         [r"\bhow are you\b", r"\bhow do you do\b", r"\bhow is it going\b", r"\bwhats up\b"],
         responses=[
             "I'm doing great, thanks for asking! How about you?",
             "All systems running smoothly! How are you?",
         ]),

    Rule("calc",
         [r"^(?:calculate|calc|compute|solve|what is|whats)?\s*"
          r"([\d\s\+\-\*/%\.\(\)\^]+)$"],
         handler=handle_calc),

    Rule("time",
         [r"\bwhat(?:s| is)? the time\b", r"\bcurrent time\b", r"\btime is it\b", r"\btell me the time\b"],
         handler=handle_time),

    Rule("date",
         [r"\bwhat(?:s| is)? the date\b", r"\btodays date\b", r"\bwhat day is it\b", r"\bcurrent date\b"],
         handler=handle_date),

    Rule("joke",
         [r"\bjoke\b", r"\bmake me laugh\b", r"\bfunny\b"],
         responses=[
             "Why do programmers prefer dark mode? Because light attracts bugs!",
             "Why did the developer go broke? He used up all his cache.",
             "There are 10 types of people: those who understand binary and those who don't.",
             "A SQL query walks into a bar, sees two tables and asks: 'Can I join you?'",
         ]),

    Rule("fun_fact",
         [r"\bfun fact\b", r"\btell me something\b", r"\binteresting fact\b"],
         responses=[
             "The term 'Artificial Intelligence' was coined in 1956 at the Dartmouth Conference.",
             "ELIZA, one of the first chatbots, was created at MIT in 1966.",
             "The first computer bug was a real moth found inside a Harvard Mark II computer in 1947.",
         ]),

    Rule("what_is_ai",
         [r"\bwhat is (?:ai|artificial intelligence)\b", r"\bdefine (?:ai|artificial intelligence)\b"],
         responses=[
             "Artificial Intelligence is the field of building machines that can perform tasks "
             "that normally need human intelligence, like learning, reasoning, and understanding language.",
         ]),

    Rule("what_is_ml",
         [r"\bwhat is (?:ml|machine learning)\b"],
         responses=[
             "Machine Learning is a branch of AI where computers learn patterns from data "
             "instead of being explicitly programmed with rules.",
         ]),

    Rule("what_is_nlp",
         [r"\bwhat is (?:nlp|natural language processing)\b"],
         responses=[
             "Natural Language Processing (NLP) helps computers understand, interpret and generate "
             "human language. Chatbots like me are a simple example of it.",
         ]),

    Rule("what_is_chatbot",
         [r"\bwhat is a chatbot\b", r"\bhow do you work\b", r"\bhow do chatbots work\b"],
         responses=[
             "A rule-based chatbot matches your message against predefined patterns and replies "
             "with a matching response. I use regular expressions to do exactly that.",
         ]),

    Rule("creator",
         [r"\bwho (?:made|created|built|developed) you\b"],
         responses=[
             "I was built in Python as part of the TechGuide AI Internship, Project 1.",
         ]),

    Rule("capabilities",
         [r"\bwhat can you do\b", r"\bhelp\b", r"\bcommands\b", r"\bfeatures\b"],
         handler=handle_help),

    Rule("feeling",
         [r"\bi(?:\s+am|m|\s+feel)\s+(?:feeling\s+)?(sad|tired|bored|stressed|angry|happy|great|good|fine|excited|awesome)\b"],
         handler=handle_feeling),

    Rule("thanks",
         [r"\b(thanks|thank you|thx|appreciate it)\b"],
         responses=["You're welcome!", "Happy to help!", "Anytime!"]),

    Rule("greeting",
         [r"\b(hi|hello|hey|hola|namaste|howdy|good morning|good afternoon|good evening)\b"],
         handler=handle_greeting),

    Rule("weather",
         [r"\bweather\b"],
         responses=[
             "I can't check live weather because I'm an offline rule-based bot, "
             "but a weather website or app will give you accurate data.",
         ]),
]

FALLBACKS = [
    "Sorry, I didn't understand that. Type 'help' to see what I can do.",
    "Hmm, I don't have a rule for that yet. Could you rephrase it?",
    "I'm still learning! Try asking about AI, the time, a joke, or some maths.",
]


# --------------------------------------------------------------------------
# Core engine
# --------------------------------------------------------------------------
def clean(text: str) -> str:
    """Lowercase and strip punctuation (keeps digits and maths symbols)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\+\-\*/%\.\(\)\^]", "", text)
    text = re.sub(r"(?<![\d\)])\.(?![\d])", " ", text)  # drop sentence dots, keep decimals
    return re.sub(r"\s+", " ", text).strip()


def get_response(user_input: str, ctx: Context) -> str:
    text = clean(user_input)
    if not text:
        return "Please type something and I'll respond."

    reply = None
    for rule in RULES:
        found = rule.match(text)
        if found:
            ctx.last_intent = rule.name
            reply = rule.handler(found, ctx) if rule.handler else random.choice(rule.responses)
            break

    if reply is None:
        ctx.last_intent = "fallback"
        reply = random.choice(FALLBACKS)

    ctx.history.append((user_input, reply))
    return reply


def log_turn(user_input: str, reply: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{stamp}] You: {user_input}\n[{stamp}] {BOT_NAME}: {reply}\n")
    except OSError:
        pass  # logging must never crash the bot


# --------------------------------------------------------------------------
# Terminal interface
# --------------------------------------------------------------------------
def run_cli() -> None:
    ctx = Context()
    print("=" * 60)
    print(f"  {BOT_NAME} - Rule-Based Chatbot")
    print("  Type 'help' for options, 'bye' to exit.")
    print("=" * 60)
    while not ctx.finished:
        try:
            user_input = input("You   : ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: Goodbye!")
            break
        reply = get_response(user_input, ctx)
        print(f"{BOT_NAME}: {reply}\n")
        log_turn(user_input, reply)


# --------------------------------------------------------------------------
# Optional GUI (tkinter ships with Python, nothing to install)
# --------------------------------------------------------------------------
def run_gui() -> None:
    import tkinter as tk
    from tkinter import scrolledtext

    ctx = Context()
    root = tk.Tk()
    root.title(f"{BOT_NAME} - Rule-Based Chatbot")
    root.geometry("560x620")
    root.configure(bg="#111827")

    chat = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, state="disabled", bg="#1f2937", fg="#f9fafb",
        font=("Segoe UI", 11), padx=10, pady=10, bd=0)
    chat.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 6))
    chat.tag_config("user", foreground="#93c5fd")
    chat.tag_config("bot", foreground="#6ee7b7")

    def add(sender: str, message: str, tag: str) -> None:
        chat.configure(state="normal")
        chat.insert(tk.END, f"{sender}: ", tag)
        chat.insert(tk.END, message + "\n\n")
        chat.configure(state="disabled")
        chat.see(tk.END)

    bottom = tk.Frame(root, bg="#111827")
    bottom.pack(fill=tk.X, padx=12, pady=(0, 12))
    entry = tk.Entry(bottom, font=("Segoe UI", 12), bg="#374151", fg="white",
                     insertbackground="white", bd=0)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
    entry.focus()

    def send(event=None) -> None:
        text = entry.get().strip()
        if not text:
            return
        entry.delete(0, tk.END)
        add("You", text, "user")
        reply = get_response(text, ctx)
        add(BOT_NAME, reply, "bot")
        log_turn(text, reply)
        if ctx.finished:
            root.after(1200, root.destroy)

    tk.Button(bottom, text="Send", command=send, bg="#2563eb", fg="white",
              font=("Segoe UI", 11, "bold"), bd=0, padx=16, pady=6).pack(side=tk.RIGHT)
    entry.bind("<Return>", send)

    add(BOT_NAME, "Hi! I'm TechBot. Type 'help' to see what I can do.", "bot")
    root.mainloop()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based chatbot")
    parser.add_argument("--gui", action="store_true", help="open the chat window")
    args = parser.parse_args()
    run_gui() if args.gui else run_cli()