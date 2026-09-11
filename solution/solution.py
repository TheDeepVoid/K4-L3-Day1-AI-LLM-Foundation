"""
K4 — Ngày 1: Khám Phá LLM API (4 tiếng)
AICB-P1: AI Practical Competency Program, Phase 1

Bài làm hoàn chỉnh — solution/solution.py.
Dùng endpoint Gemini (tương thích chuẩn OpenAI) cấu hình trong .env.
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

OPENAI_MODEL = os.getenv("MODEL1", "gemini-3.5-flash-lite")
OPENAI_MINI_MODEL = os.getenv("MODEL2", "gemini-3.1-flash-lite")
OPENAI_API_KEY = os.getenv("GEMINI_API") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")


def _client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: phút 60–100)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi model lớn
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content, latency


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi model nhỏ (mini)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model nhỏ hơn — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh model lớn vs model nhỏ
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (USD ước tính cho phản hồi)
    """
    gpt4o_text, gpt4o_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)

    cost = (
        (len(gpt4o_text.split()) / 0.75)
        / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )

    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: phút 100–140)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với messages gồm 2 phần: system prompt và user prompt.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content, latency


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Returns:
        Số token (int).
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí tách bạch input / output
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token thật.

    Returns:
        Dict với các key:
            - "input_tokens":  int
            - "output_tokens": int
            - "input_cost":    float  (USD)
            - "output_cost":   float  (USD)
            - "total_cost":    float  (USD)
    """
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)

    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: phút 150–190)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.

    Hành vi:
        - Stream token từ API ngay khi chúng được sinh ra (in từng chunk).
        - Duy trì 3 lượt hội thoại gần nhất trong history.
        - Gõ 'quit' hoặc 'exit' để thoát.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
    history = []
    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = history + [{"role": "user", "content": user_msg}]
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).

    Returns:
        Giá trị trả về của fn() khi thành công.

    Raises:
        Exception cuối cùng của fn() sau khi hết số lần thử.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: phút 190–230)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ đã xây trong Part 1–3.

    Returns:
        Dict thống kê phiên chat:
            - "num_turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "total_tokens": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 6 message)
    """
    if get_input is None:
        get_input = input
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
    history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]

        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    results = []
    for prompt in prompts:
        item = compare_models(prompt)
        item["prompt"] = prompt
        results.append(item)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    """
    header = (
        f"{'Prompt':<40} | {'GPT-4o Response':<40} | {'Mini Response':<40} "
        f"| {'GPT-4o Latency':>14} | {'Mini Latency':>12}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        prompt = r["prompt"][:40]
        gpt = r["gpt4o_response"][:40]
        mini = r["mini_response"][:40]
        lines.append(
            f"{prompt:<40} | {gpt:<40} | {mini:<40} "
            f"| {r['gpt4o_latency']:>14.3f} | {r['mini_latency']:>12.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần API key)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")