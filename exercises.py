#!/usr/bin/env python3
"""
exercises.py — kiểm tra toàn bộ yêu cầu trong exercises.md
K4 — Ngày 1: Khám Phá LLM API

Chạy:  python exercises.py

Script này tự động chạy và kiểm tra từng yêu cầu của phiếu thực hành:

    BLOCK 1 — API Cơ Bản
        Câu 1.1: call_openai với temperature 0.0 / 0.5 / 1.0 / 1.5
                 (prompt: "Hãy kể cho tôi một sự thật thú vị về Việt Nam.")
        Câu 1.3: ước lượng chênh lệch chi phí GPT-4o vs GPT-4o-mini
                 cho workload 10.000 người dùng × 3 lần × ~350 token đầu ra

    BLOCK 2 — System Prompt & Token
        Câu 2.1: chat_with_system_prompt với 2 persona khác nhau,
                 cùng câu hỏi "Giải thích blockchain là gì?"
        Câu 2.2: đoạn tiếng Việt ~100 từ — so sánh count_tokens
                 với ước lượng từ/0.75 và tính % chênh lệch

    BLOCK 3 — Streaming & Độ Bền
        Câu 3.1: chatbot streaming (stream=True) + thoát bằng 'quit'
        Câu 3.2: retry_with_backoff (thành công lần đầu / retry / hết lượt)

    BLOCK 4 — Mini-Project
        Câu 4.1/4.2: run_assistant với persona — kiểm tra thống kê,
                     history cắt ≤ 6 message, stream=True, max_turns

    DANH SÁCH KIỂM TRA NỘP BÀI
        - 9/9 câu trong exercises.md đã trả lời
        - 4 checkpoint pytest pass
        - python grade.py đạt ≥ 75/100

Bài làm được chấm là: solution/solution.py (nếu có) — fallback template.py,
giống hệt tests/_loader.py và grade.py.

Yêu cầu API key để chạy các phép đo live (Câu 1.1, 2.1, demo 4.1). Nếu
thiếu key hoặc gặp lỗi mạng, script vẫn kiểm tra được bằng mock và báo SKIP.

Trả về exit code 0 khi mọi check bắt buộc pass, khác 0 nếu có check FAIL.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

DAY_DIR = Path(__file__).resolve().parent

# ===========================================================================
# Nạp bài làm — giữ ưu tiên giống grade.py / tests/_loader.py
# ===========================================================================
SOLUTION = DAY_DIR / "solution" / "solution.py"


def _load_module(path: Path) -> object:
    unique_name = "k4_day01_exercises_check"
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD_PATH = SOLUTION if SOLUTION.exists() else DAY_DIR / "template.py"
MOD = _load_module(MOD_PATH)

# ===========================================================================
# Bộ đếm/ghi nhận kết quả
# ===========================================================================
RESULTS: list[tuple[str, str, str, str]] = []  # (block, mô tả, status, chi tiết)


def check(block: str, name: str, status: str, detail: str = "") -> None:
    RESULTS.append((block, name, status, detail))
    mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP", "INFO": "INFO"}[status]
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def note(text: str) -> None:
    print(f"      ↳ {text}")


def _safe(fn: Callable[[], object]) -> tuple[object, Exception | None]:
    """Gọi fn(); trả về (kết quả, None) hoặc (None, exception)."""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 — muốn bắt mọi lỗi chạy thật
        return None, exc


def _has_api_key() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API")
        or getattr(MOD, "OPENAI_API_KEY", "")
    )


def _make_stream(text: str):
    """Giả lập stream chunk giống OpenAI streaming (giống file test)."""
    chunks = []
    for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = piece
        chunks.append(chunk)
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta.content = None
    chunks.append(final)
    return chunks


# ===========================================================================
# BLOCK 1 — API CƠ BẢN
# ===========================================================================
def block1() -> None:
    print("\n" + "=" * 70)
    print("BLOCK 1 — API CƠ BẢN (Câu 1.1, 1.2, 1.3)")
    print("=" * 70)

    # Câu 1.1 — độ nhạy của temperature --------------------------------
    prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
    print(f"\n>>> Câu 1.1 — call_openai với temperature 0.0/0.5/1.0/1.5")
    note(f"prompt: {prompt!r}")
    live_ok = True
    for temp in (0.0, 0.5, 1.0, 1.5):
        res, err = _safe(lambda t=temp: MOD.call_openai(prompt, temperature=t))
        if err is not None:
            live_ok = False
            check(
                "Block 1",
                f"call_openai temperature={temp}",
                "SKIP",
                f"live lỗi: {err} — sẽ kiểm tra lại bằng mock",
            )
            continue
        text, latency = res
        ok = (
            isinstance(text, str)
            and len(text) > 0
            and isinstance(latency, float)
            and latency > 0.0
        )
        detail = f"latency={latency:.2f}s, {len(text)} ký tự"
        check("Block 1", f"call_openai temperature={temp}", "PASS" if ok else "FAIL", detail)
        note(f"phản hồi: {text!r}")

    if not live_ok:
        # Fallback mock — vẫn xác nhận hàm hoạt động đúng chữ ký
        for temp in (0.0, 0.5, 1.0, 1.5):
            with patch("openai.OpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                mock_openai.return_value.chat.completions.create.return_value = (
                    _ok_response(f"Trả lời mock cho temperature={temp}")
                )
                res, err = _safe(lambda t=temp: MOD.call_openai(prompt, temperature=t))
            if err is None:
                check(
                    "Block 1",
                    f"call_openai temperature={temp} (mock)",
                    "PASS",
                    "chạy được với OpenAI() được mock",
                )
            else:
                check("Block 1", f"call_openai temperature={temp} (mock)", "FAIL", str(err))

    # Câu 1.2 — chọn temperature cho chatbot hỗ trợ khách hàng -----------
    print("\n>>> Câu 1.2 — chọn temperature cho chatbot hỗ trợ khách hàng")
    check(
        "Block 1",
        "temperature thấp (0.0–0.2) cho support ổn định, đúng chuẩn",
        "INFO",
        "đây là câu phản ánh — hãy trả lời trong exercises.md",
    )

    # Câu 1.3 — đánh đổi chi phí ---------------------------------------
    print("\n>>> Câu 1.3 — 10.000 người × 3 lần × ~350 token đầu ra")
    users, calls, out_tokens = 10_000, 3, 350
    total_output = users * calls * out_tokens
    try:
        p4o = MOD.PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
        pmini = MOD.PRICING_PER_1K_TOKENS["gpt-4o-mini"]["output"]
    except Exception as exc:
        check("Block 1", "tính chi phí GPT-4o vs mini", "FAIL", f"thiếu bảng giá: {exc}")
        return
    cost_4o = total_output / 1000 * p4o
    cost_mini = total_output / 1000 * pmini
    ratio = cost_4o / cost_mini
    ok = ratio > 1 and abs(ratio - p4o / pmini) < 1e-9
    check(
        "Block 1",
        "ước tính chênh lệch chi phí GPT-4o vs mini",
        "PASS" if ok else "FAIL",
        f"GPT-4o ${cost_4o:,.2f}/ngày, mini ${cost_mini:,.2f}/ngày "
        f"— đắt hơn {ratio:.1f}×",
    )
    note(
        "Gợi ý trả lời: workload này output-only ~"
        f"{total_output/1000:.0f}K token/ngày."
    )


def _ok_response(text: str):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ===========================================================================
# BLOCK 2 — SYSTEM PROMPT & TOKEN
# ===========================================================================
def block2() -> None:
    print("\n" + "=" * 70)
    print("BLOCK 2 — SYSTEM PROMPT & TOKEN (Câu 2.1, 2.2)")
    print("=" * 70)

    # Câu 2.1 — sức mạnh của persona ------------------------------------
    print('\n>>> Câu 2.1 — chat_with_system_prompt: "Giải thích blockchain là gì?"')
    question = "Giải thích blockchain là gì?"
    personas = {
        "giáo viên tiểu học": "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi.",
        "chuyên gia tài chính": "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật.",
    }
    responses = {}
    live_ok = True
    for label, sp in personas.items():
        res, err = _safe(
            lambda s=sp: MOD.chat_with_system_prompt(s, question, max_tokens=400)
        )
        if err is not None:
            live_ok = False
            check("Block 2", f"persona {label}", "SKIP", f"live lỗi: {err}")
            continue
        text, latency = res
        ok = isinstance(text, str) and len(text) > 0
        responses[label] = (text, latency)
        check(
            "Block 2",
            f"persona {label}",
            "PASS" if ok else "FAIL",
            f"latency={latency:.2f}s, {len(text)} ký tự, {len(text.split())} từ",
        )
        note(f"phản hồi: {text!r}")

    if not live_ok:
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            mock_openai.return_value.chat.completions.create.return_value = _ok_response(
                "Trả lời mock theo persona"
            )
            for label, sp in personas.items():
                res, err = _safe(lambda s=sp: MOD.chat_with_system_prompt(s, question))
                if err is None:
                    check("Block 2", f"persona {label} (mock)", "PASS", "")
                else:
                    check("Block 2", f"persona {label} (mock)", "FAIL", str(err))
    elif len(responses) == len(personas):
        lens = {k: len(v[0].split()) for k, v in responses.items()}
        note(f"độ dài từ: {lens} — persona làm phản hồi khác nhau rõ rệt")

    # Câu 2.2 — tiktoken vs đếm từ ---------------------------------------
    print("\n>>> Câu 2.2 — token tiếng Việt: tiktoken vs ước lượng từ/0.75")
    paragraph = (
        "Việt Nam là một đất nước nằm ở khu vực Đông Nam Á, nổi bật với bờ biển "
        "dài hơn ba nghìn cây số và đồng bằng sông Cửu Long trù phú ở phía Nam. "
        "Thủ đô Hà Nội gìn giữ nét cổ kính qua những con phố rêu phong, trong khi "
        "thành phố Hồ Chí Minh nhộn nhịp như một trung tâm kinh tế hiện đại. "
        "Người dân Việt Nam nổi tiếng hiếu khách, luôn sẵn lòng chào đón du khách "
        "bằng nụ cười thân thiện. Bên cạnh đó, đất nước này còn có nhiều di sản "
        "văn hóa được UNESCO ghi danh, góp phần làm nên hình ảnh Việt Nam vừa "
        "truyền thống vừa hiện đại, phong phú và đáng tự hào."
    )
    words = len(paragraph.split())
    tokens = MOD.count_tokens(paragraph)
    estimate = words / 0.75
    pct = (tokens - estimate) / estimate * 100 if estimate else 0.0
    ok = tokens > 0 and estimate > 0
    check(
        "Block 2",
        "count_tokens vs ước lượng từ/0.75",
        "PASS" if ok else "FAIL",
        f"{words} từ → tiktoken={tokens}, ước lượng={estimate:.0f}, "
        f"chênh {pct:+.1f}%",
    )
    note("Gợi ý trả lời: tiếng Việt nhiều dấu/phụ âm ghép → token dày hơn tiếng Anh.")


# ===========================================================================
# BLOCK 3 — STREAMING & ĐỘ BỀN
# ===========================================================================
def block3() -> None:
    print("\n" + "=" * 70)
    print("BLOCK 3 — STREAMING & ĐỘ BỀN (Câu 3.1, 3.2)")
    print("=" * 70)

    # Câu 3.1 — chatbot streaming ---------------------------------------
    print("\n>>> Câu 3.1 — streaming_chatbot (stream=True, thoát bằng 'quit')")
    try:
        with patch("openai.OpenAI") as mock_openai, patch(
            "builtins.input", side_effect=["Xin chào", "quit"]
        ), redirect_stdout(io.StringIO()):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_stream("Chào bạn!")
            try:
                MOD.streaming_chatbot()
            except StopIteration:
                pass  # cạn side_effect của input() — chấp nhận được
        called = mock_client.chat.completions.create.called
        _, kwargs = mock_client.chat.completions.create.call_args
        ok = called and kwargs.get("stream", False)
        check(
            "Block 3",
            "streaming_chatbot gọi API với stream=True",
            "PASS" if ok else "FAIL",
            "mock client · stream=True" if ok else "thiếu stream=True hoặc không gọi API",
        )
    except Exception as exc:
        check("Block 3", "streaming_chatbot chạy được", "FAIL", str(exc))

    # Câu 3.2 — retry với exponential backoff ---------------------------
    print("\n>>> Câu 3.2 — retry_with_backoff")
    ok = MOD.retry_with_backoff(lambda: 42) == 42
    check("Block 3", "thành công ngay lần đầu", "PASS" if ok else "FAIL", "fn() trả 42")

    calls = [0]

    def flaky():
        calls[0] += 1
        if calls[0] < 2:
            raise ValueError("transient")
        return "ok"

    res = MOD.retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
    ok = res == "ok" and calls[0] == 2
    check(
        "Block 3",
        "retry khi gặp lỗi tạm thời",
        "PASS" if ok else "FAIL",
        f"gọi {calls[0]} lần (lần 2 thành công) — delay cấp số nhân base_delay×2^attempt",
    )

    def always_fail():
        raise ValueError("permanent")

    try:
        MOD.retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)
        check("Block 3", "raise sau hết số lần thử", "FAIL", "không raise")
    except ValueError:
        check("Block 3", "raise sau hết số lần thử", "PASS", "ValueError sau 2 retry")


# ===========================================================================
# BLOCK 4 — MINI-PROJECT
# ===========================================================================
PERSONA = (
    "Bạn là chuyên gia tư vấn thân thiện. Trả lời ngắn gọn, đúng trọng tâm "
    "bằng tiếng Việt; nếu không chắc chắn thì nói rõ thay vì bịa đặt."
)
REQUIRED_KEYS = {"num_turns", "total_tokens", "total_cost", "history"}


def _run_mock_assistant(
    user_messages: list[str],
    replies: list[str],
    max_turns: int | None = None,
):
    buf = io.StringIO()
    with patch("openai.OpenAI") as mock_openai, redirect_stdout(buf):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_stream(r) for r in replies
        ]
        get_input = MagicMock(side_effect=list(user_messages) + ["quit"])
        result = MOD.run_assistant(
            PERSONA, get_input=get_input, max_turns=max_turns
        )
    return result, mock_client


def block4() -> None:
    print("\n" + "=" * 70)
    print("BLOCK 4 — MINI-PROJECT: run_assistant (Câu 4.1, 4.2)")
    print("=" * 70)
    print(f"\n>>> Câu 4.1 — persona mẫu")
    note(f"persona: {PERSONA!r}")

    print("\n>>> Câu 4.2 — kiểm tra hành vi run_assistant (mock)")

    # quit ngay lập tức
    with patch("openai.OpenAI"):
        result = MOD.run_assistant(PERSONA, get_input=MagicMock(side_effect=["quit"]))
    ok = all(k in result for k in REQUIRED_KEYS) and result["num_turns"] == 0
    check(
        "Block 4",
        "thoát ngay khi gõ 'quit' → trả dict đủ 4 key",
        "PASS" if ok else "FAIL",
        f"num_turns={result.get('num_turns')}, keys={sorted(result)}",
    )

    # 2 lượt — stats dương
    result, mock_client = _run_mock_assistant(
        ["Xin chào", "Kể một sự thật thú vị"],
        ["Chào bạn, mình giúp gì được?", "Việt Nam có hơn 3000 km bờ biển."],
    )
    ok = (
        result["num_turns"] == 2
        and result["total_tokens"] > 0
        and result["total_cost"] > 0.0
    )
    check(
        "Block 4",
        "2 lượt chat → num_turns=2, total_tokens/total_cost > 0",
        "PASS" if ok else "FAIL",
        f"num_turns={result['num_turns']}, tokens={result['total_tokens']}, "
        f"cost=${result['total_cost']:.6f}",
    )

    ok = mock_client.chat.completions.create.called
    if ok:
        _, kwargs = mock_client.chat.completions.create.call_args
        ok = kwargs.get("stream", False)
        system_contents = [
            m["content"] for m in kwargs.get("messages", []) if m["role"] == "system"
        ]
        ok = ok and any(PERSONA in c for c in system_contents)
    check(
        "Block 4",
        "API gọi với stream=True + persona làm system prompt",
        "PASS" if ok else "FAIL",
        "",
    )

    # history giữ 3 lượt cuối (≤ 6 message)
    result, _ = _run_mock_assistant(
        [f"Câu hỏi số {i}" for i in range(1, 6)],
        [f"Trả lời số {i}." for i in range(1, 6)],
    )
    ok = result["num_turns"] == 5 and len(result["history"]) <= 6
    check(
        "Block 4",
        "history cắt còn tối đa 3 lượt (≤ 6 message)",
        "PASS" if ok else "FAIL",
        f"num_turns={result['num_turns']}, len(history)={len(result['history'])}",
    )

    # max_turns giới hạn phiên
    result, _ = _run_mock_assistant(
        [f"Câu {i}" for i in range(10)],
        [f"Trả lời {i}" for i in range(10)],
        max_turns=2,
    )
    ok = result["num_turns"] == 2
    check(
        "Block 4",
        "max_turns giới hạn phiên chat",
        "PASS" if ok else "FAIL",
        f"num_turns={result['num_turns']}",
    )

    # Demo live (tùy chọn, cần API key)
    if _has_api_key():
        print("\n>>> Demo live run_assistant (cần API key)")
        get_input = MagicMock(side_effect=["Chào bạn", "Bạn có thể làm gì?", "quit"])
        res, err = _safe(lambda: MOD.run_assistant(PERSONA, get_input=get_input))
        if err is None:
            check(
                "Block 4",
                "demo live run_assistant",
                "PASS",
                f"num_turns={res['num_turns']}, tokens={res['total_tokens']}, "
                f"cost=${res['total_cost']:.6f}",
            )
        else:
            check("Block 4", "demo live run_assistant", "SKIP", f"lỗi: {err}")
    else:
        check("Block 4", "demo live run_assistant", "SKIP", "chưa có API key trong .env")


# ===========================================================================
# Bonus — batch_compare & format_comparison_table
# ===========================================================================
def bonus() -> None:
    if not (hasattr(MOD, "batch_compare") and hasattr(MOD, "format_comparison_table")):
        return
    print("\n" + "=" * 70)
    print("BONUS — batch_compare / format_comparison_table")
    print("=" * 70)
    fake = {
        "gpt4o_response": "trả lời lớn",
        "mini_response": "trả lời nhỏ",
        "gpt4o_latency": 0.5,
        "mini_latency": 0.3,
        "gpt4o_cost_estimate": 0.001,
    }
    prompts = ["hỏi 1", "hỏi 2"]
    with patch.object(MOD, "compare_models", return_value=fake):
        results = MOD.batch_compare(prompts)
    ok = (
        isinstance(results, list)
        and len(results) == len(prompts)
        and all(r.get("prompt") in prompts for r in results)
    )
    check(
        "Bonus",
        "batch_compare chạy đủ prompt & thêm key 'prompt'",
        "PASS" if ok else "FAIL",
        f"{len(results)} mục",
    )
    table = MOD.format_comparison_table(
        [
            {
                "prompt": "Bờ biển Việt Nam dài bao nhiêu km?",
                "gpt4o_response": "Trên 3000 km.",
                "mini_response": "3260 km.",
                "gpt4o_latency": 0.51,
                "mini_latency": 0.32,
            }
        ]
    )
    check(
        "Bonus",
        "format_comparison_table tạo bảng",
        "PASS" if isinstance(table, str) and "GPT-4o" in table else "FAIL",
    )


# ===========================================================================
# Danh sách kiểm tra nộp bài
# ===========================================================================
def _exercise_file() -> Path | None:
    sol = DAY_DIR / "solution" / "exercises.md"
    return sol if sol.exists() else DAY_DIR / "exercises.md"


def _run_pytest(args: list[str]) -> tuple[int, int]:
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no", "--no-header",
           "-p", "no:cacheprovider"] + list(args)
    proc = subprocess.run(cmd, cwd=DAY_DIR, capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).replace("\r", "")
    m_pass = re.search(r"(\d+)\s+passed", out)
    m_fail = re.search(r"(\d+)\s+failed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    failed = int(m_fail.group(1)) if m_fail else 0
    return passed, failed


def checklist() -> None:
    print("\n" + "=" * 70)
    print("DANH SÁCH KIỂM TRA NỘP BÀI")
    print("=" * 70)

    # 1) exercises.md — 9/9 câu trả lời
    ex_file = _exercise_file()
    if ex_file is None:
        check("Chốt bài", "exercises.md có 9/9 câu trả lời", "FAIL", "không tìm thấy file")
        answered = 0
    else:
        remaining = sum(
            1
            for line in ex_file.read_text(encoding="utf-8").splitlines()
            if line.strip() == "> *Câu trả lời của bạn*"
        )
        answered = max(0, 9 - remaining)
        check(
            "Chốt bài",
            f"exercises.md — 9/9 câu trả lời ({ex_file.relative_to(DAY_DIR)})",
            "PASS" if answered == 9 else "FAIL",
            f"{answered}/9 câu đã trả lời",
        )

    # 2) 4 checkpoint pytest
    groups = [
        ("CP1 — Part 1: API cơ bản", ["tests/test_part1.py"]),
        ("CP2 — Part 2: System prompt & token", ["tests/test_part2.py"]),
        ("CP3 — Part 3: Streaming & retry", ["tests/test_part3.py"]),
        ("CP4 — Basic", ["tests/test_part4.py", "-k", "Basic"]),
        ("CP4 — Scenario", ["tests/test_part4.py", "-k", "Scenario"]),
    ]
    for name, args in groups:
        passed, failed = _run_pytest(args)
        check(
            "Chốt bài",
            name,
            "PASS" if failed == 0 and passed > 0 else "FAIL",
            f"{passed} pass, {failed} fail",
        )

    # 3) python grade.py
    proc = subprocess.run(
        [sys.executable, "grade.py"], cwd=DAY_DIR, capture_output=True, text=True
    )
    grade_out = (proc.stdout + proc.stderr).replace("\r", "")
    m_total = re.search(r"TỔNG\s+([\d.]+)/100", grade_out)
    total = float(m_total.group(1)) if m_total else None
    detail = f"tổng {total}/100 (mục tiêu ≥ 75)" if total is not None else "không đọc được điểm"
    check(
        "Chốt bài",
        "python grade.py ≥ 75/100",
        "PASS" if (total is not None and total >= 75) else "FAIL",
        detail,
    )


# ===========================================================================
# Tổng hợp & exit code
# ===========================================================================
def main() -> int:
    print(f"Đang kiểm tra bài làm: {MOD_PATH.relative_to(DAY_DIR)}")
    print(f"API key: {'có' if _has_api_key() else 'KHÔNG — các bài live sẽ SKIP'}")

    block1()
    block2()
    block3()
    block4()
    bonus()
    checklist()

    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    counts = Counter(entry[2] for entry in RESULTS)
    for status in ("PASS", "FAIL", "SKIP", "INFO"):
        if counts.get(status):
            print(f"  {status:<5} {counts[status]}")
    failed = [
        (block, name, detail)
        for block, name, status, detail in RESULTS
        if status == "FAIL"
    ]
    if failed:
        print("-" * 70)
        print("Các mục cần xử lý:")
        for block, name, detail in failed:
            print(f"  {block} | {name} — {detail}")
    print("=" * 70)
    if failed:
        print("KẾT LUẬN: vẫn còn yêu cầu chưa đạt (kiểm tra các mục FAIL bên trên).")
        return 1
    print("KẾT LUẬN: tất cả yêu cầu kiểm chứng được đều đạt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())