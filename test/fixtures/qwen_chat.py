"""Qwen3.8's chat template rendered for gen.llm.chat-test's fixed messages.

Run: python3 -I test/fixtures/qwen_chat.py <chat_template.jinja> test/fixtures/qwen38-chat.edn

The template is the one Qwen3.8's model files carry (chat_template.jinja);
it is not in this repository, only its renderings. jinja2 renders it as
Hugging Face's apply_chat_template does: tojson as json.dumps without
ASCII escapes, raise_exception raising.
"""
import json
import sys

import jinja2


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


TOOL = {"type": "function",
        "function": {"name": "eval", "description": "Eval \"it\".",
                     "parameters": {"type": "object",
                                    "properties": {"code": {"type": "string", "description": "Forms."}},
                                    "required": ["code"]}}}

CONVERSATION = [{"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1", "reasoning_content": "r1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2", "reasoning_content": "r2"},
                {"role": "tool", "content": "T1"},
                {"role": "tool", "content": "T2"}]

CASES = {
    "user-thinking": ([{"role": "user", "content": "U"}], {"enable_thinking": True}),
    "user-default-effort": ([{"role": "user", "content": "U"}], {}),
    "system-low": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                   {"enable_thinking": True, "reasoning_effort": "low"}),
    "system-medium": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                      {"enable_thinking": True, "reasoning_effort": "medium"}),
    "system-xhigh": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                     {"enable_thinking": True}),
    "empty-system-thinking": ([{"role": "system", "content": "  "}, {"role": "user", "content": "U"}],
                              {"enable_thinking": True}),
    "user-off": ([{"role": "user", "content": "U"}], {"enable_thinking": False}),
    "system-off": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U\n"}], {"enable_thinking": False}),
    "tools-thinking": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                       {"enable_thinking": True, "tools": [TOOL]}),
    "tools-off": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                  {"enable_thinking": False, "tools": [TOOL]}),
    "conversation-thinking": (CONVERSATION, {"enable_thinking": True}),
    "conversation-off": (CONVERSATION, {"enable_thinking": False}),
    "conversation-not-preserved": (CONVERSATION, {"enable_thinking": True, "preserve_thinking": False}),
    "no-generation-prompt": ([{"role": "user", "content": "U"}], {"enable_thinking": True, "add_generation_prompt": False}),
}


def edn_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def main(template_path, out_path):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    with open(template_path) as f:
        template = env.from_string(f.read())
    lines = [";; Qwen3.8's chat template rendered by test/fixtures/qwen_chat.py (jinja2), by case"]
    lines.append("{")
    for name, (messages, kwargs) in CASES.items():
        kw = {"add_generation_prompt": True, **kwargs}
        lines.append(" :" + name + " " + edn_string(template.render(messages=messages, **kw)))
    lines.append("}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
