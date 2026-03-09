"""
LLM 润色层 benchmark 测试
测试维度：准确性（错别字、断句、标点）、多要点分解、延迟
"""

import time
import asyncio
import json
import httpx
from openai import AsyncOpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

GITHUB_TOKEN_FILE = "github-token"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

GITHUB_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

OLLAMA_MODELS = [
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
]

SYSTEM_PROMPT = """你是一个专业的语音转文字后处理助手。请对以下 ASR 转录文本进行润色：
1. 修正中英文错别字（保持原意，不要改写）
2. 添加合适的标点符号（逗号、句号、问号、感叹号等）
3. 合理断句
只输出润色后的文本，不要添加任何解释或额外内容。"""

# ── 测试用例 ──────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── 纯中文 ──
    {
        "id": "zh-typo",
        "lang": "中文",
        "desc": "错别字修正",
        "input": "今天天气真好我们一起去哪里玩吧我想去公圆里面跑步然后去超市买些东西回家做饭",
        "checks": ["公园", "。", "，"],
    },
    {
        "id": "zh-punct",
        "lang": "中文",
        "desc": "标点符号与断句",
        "input": "你好请问你今天有空吗我想和你一起去看电影如果你有空的话我们可以先吃个饭然后再去",
        "checks": ["？", "。", "，"],
    },
    {
        "id": "zh-multi",
        "lang": "中文",
        "desc": "多要点分解",
        "input": "明天会议有三个议题第一个是讨论Q3销售数据第二个是审查新产品方案第三个是确认下季度预算",
        "checks": ["第一", "第二", "第三", "。"],
    },
    {
        "id": "zh-question",
        "lang": "中文",
        "desc": "疑问句标点",
        "input": "你知道这件事吗他们是什么时候来的为什么没有提前通知我",
        "checks": ["？"],
    },
    # ── 纯英文 ──
    {
        "id": "en-typo",
        "lang": "英文",
        "desc": "英文错字修正",
        "input": "i went to the store yesteday to bye some grocerys but the cashier maked a mistake with my chagne",
        "checks": ["yesterday", "buy", "groceries", "made", "change", "."],
    },
    {
        "id": "en-punct",
        "lang": "英文",
        "desc": "英文标点断句",
        "input": "the meeting starts at nine oclock we need to prepare the slides review the budget and discuss the new hire process",
        "checks": [",", "."],
    },
    {
        "id": "en-multi",
        "lang": "英文",
        "desc": "英文多要点",
        "input": "the project has three main goals first we need to increase user retention second we should reduce server costs third we want to launch the mobile app by q4",
        "checks": ["first", "second", "third", "."],
    },
    # ── 中英混合 ──
    {
        "id": "mixed-typo",
        "lang": "中英混合",
        "desc": "中英混合错字",
        "input": "今天的metting开始了我们讨论了servr的perfonmance问题还有user的feedbak",
        "checks": ["meeting", "server", "performance", "feedback", "。"],
    },
    {
        "id": "mixed-punct",
        "lang": "中英混合",
        "desc": "中英混合标点",
        "input": "我们需要优化API的response time目前平均是500ms这对用户体验很不好我们的target是100ms以内",
        "checks": ["。", "，"],
    },
    {
        "id": "mixed-multi",
        "lang": "中英混合",
        "desc": "中英混合多要点",
        "input": "这次sprint我们要完成三件事第一修复login的bug第二优化dashboard的loading速度第三写完unit test",
        "checks": ["第一", "第二", "第三", "。"],
    },
    {
        "id": "mixed-terms",
        "lang": "中英混合",
        "desc": "技术术语保留",
        "input": "我们的kubernetes集群昨天出了点问题pod一直在crash loop里面restart了好几次最后发现是configmap配置写错了",
        "checks": ["Kubernetes", "crash", "ConfigMap", "。"],
    },
]

# ── 客户端构建 ────────────────────────────────────────────────────────────────

def build_github_client():
    with open(GITHUB_TOKEN_FILE) as f:
        token = f.read().strip()
    return AsyncOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

def build_ollama_client():
    return AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )

# ── 润色调用 ──────────────────────────────────────────────────────────────────

async def polish_ollama(model: str, text: str) -> tuple[str, float]:
    """Ollama 原生 API，think=false"""
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post("http://localhost:11434/api/chat", json={
                "model": model,
                "think": False,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            })
        elapsed = time.perf_counter() - t0
        data = resp.json()
        result = data.get("message", {}).get("content", "")
        return result, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return f"[ERROR] {e}", elapsed


async def polish(client: AsyncOpenAI, model: str, text: str, is_gpt5: bool) -> tuple[str, float]:
    """返回 (润色结果, 耗时秒)"""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    # GPT-5 系列用 max_completion_tokens
    if is_gpt5:
        kwargs["max_completion_tokens"] = 1000
    else:
        kwargs["max_tokens"] = 1000

    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        result = resp.choices[0].message.content or ""
        # qwen3 思考模式会输出 <think>...</think>，过滤掉
        if "<think>" in result:
            result = result.split("</think>")[-1].strip()
        return result, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return f"[ERROR] {e}", elapsed

# ── 评分 ─────────────────────────────────────────────────────────────────────

def score(output: str, checks: list[str]) -> tuple[int, int]:
    """返回 (通过数, 总数)"""
    passed = sum(1 for c in checks if c in output)
    return passed, len(checks)

# ── 主测试逻辑 ────────────────────────────────────────────────────────────────

async def run_model(client: AsyncOpenAI, model: str, is_gpt5: bool, is_ollama: bool = False):
    results = []
    for tc in TEST_CASES:
        if is_ollama:
            output, latency = await polish_ollama(model, tc["input"])
        else:
            output, latency = await polish(client, model, tc["input"], is_gpt5)
        passed, total = score(output, tc["checks"])
        results.append({
            "id": tc["id"],
            "lang": tc["lang"],
            "desc": tc["desc"],
            "input": tc["input"],
            "output": output,
            "latency": latency,
            "score": f"{passed}/{total}",
            "passed": passed,
            "total": total,
        })
        status = "✓" if passed == total else f"~{passed}/{total}"
        print(f"  [{status}] {tc['id']:20s} {latency:.2f}s  {output[:60].replace(chr(10),' ')}")
    return results

async def main():
    github_client = build_github_client()
    ollama_client = build_ollama_client()

    all_results = {}

    # GitHub 模型
    for model in GITHUB_MODELS:
        is_gpt5 = model.startswith("gpt-5")
        print(f"\n{'='*60}")
        print(f"  {model}")
        print(f"{'='*60}")
        results = await run_model(github_client, model, is_gpt5)
        all_results[model] = results

    # Ollama 模型（已测，跳过）
    # for model in OLLAMA_MODELS:
    #     print(f"\n{'='*60}")
    #     print(f"  {model}")
    #     print(f"{'='*60}")
    #     results = await run_model(ollama_client, model, False, is_ollama=True)
    #     all_results[model] = results

    # 汇总报告
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Score':>8} {'Avg Latency':>14} {'Min':>8} {'Max':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*14} {'-'*8} {'-'*8}")

    for model, results in all_results.items():
        total_passed = sum(r["passed"] for r in results)
        total_checks = sum(r["total"] for r in results)
        latencies = [r["latency"] for r in results if not r["output"].startswith("[ERROR]")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        min_lat = min(latencies) if latencies else 0
        max_lat = max(latencies) if latencies else 0
        print(f"{model:<20} {total_passed:>3}/{total_checks:<4} {avg_lat:>12.2f}s {min_lat:>7.2f}s {max_lat:>7.2f}s")

    # 保存详细结果
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())
