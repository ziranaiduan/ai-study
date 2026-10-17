#!/usr/bin/env python3
"""Task 2 · Basic 档：沿用主模板的文件后端，完成「资料 / 报告」的写入 → 读回 → 修改。

主模板（task1/main.py）只有一个动作：让 Agent 把分析报告写到 /reports/a_stock_report.md。
本作业把它拆成四个可独立验证的阶段，每一步都由宿主机（本脚本）直接核对磁盘上的真实文件：

    Phase 1  写资料  ：Agent 取数后，用 write_file 把行情 / 资讯写入 /materials/*.md
    Phase 2  读回+修改：Agent 用 read_file 读回资料，再用 edit_file 给资料补一行读回校验
    Phase 3  写报告  ：Agent 读资料，用 write_file 写 /reports/a_stock_report.md
    Phase 4  读回+修改：Agent 读回报告，用 edit_file 加「（修订版）」与修订记录，再读回确认

每个阶段结束后：
  * 记录 Agent 的完整工具调用轨迹（证明它真的调用了 write_file / read_file / edit_file）；
  * 由宿主机直接 open() 文件、算 sha256、取修改时间（证明产物真实落在项目目录里）；
  * 把阶段快照、校验结果写入 task2/evidence/main_run.json，供打卡与报告引用。

运行方式（在项目根目录）：
    .venv/bin/python task2/main.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# ============================================================================
# 0. 路径与常量
# ============================================================================
TASK_DIR = Path(__file__).resolve().parent
WS = TASK_DIR.parent  # 项目根目录 /Users/anyu/anyu/ai-study

# FilesystemBackend 的 root_dir：虚拟路径 / 在磁盘上对应的真实根目录。
# 注意：这里用绝对路径（task1 用的 "../agent_workspace" 依赖启动时的工作目录，很脆弱）。
WORKSPACE = WS / "agent_workspace" / "task2_basic"
EVIDENCE_DIR = TASK_DIR / "evidence"

MATERIAL_MD = "/materials/market_overview.md"
NEWS_MD = "/materials/hot_news.md"
REPORT_MD = "/reports/a_stock_report.md"

NOW = datetime.now()
TODAY = NOW.strftime("%Y-%m-%d")
NOW_STR = NOW.strftime("%Y-%m-%d %H:%M:%S")

load_dotenv(WS / ".env")


# ============================================================================
# 1. 模型
# ============================================================================
def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash"),
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
    )


# ============================================================================
# 2. 财经数据工具（主模板工具 + 腾讯行情兜底）
# ============================================================================
INDEX_NAMES = {"000001": "上证指数", "399001": "深证成指", "399006": "创业板指", "000300": "沪深300"}
TENCENT_URL = "https://qt.gtimg.cn/q=sh000001,sz399001,sz399006,sh000300"


def _market_from_akshare() -> str:
    """主模板的数据源：AKShare / 东方财富。"""
    import akshare as ak

    frames = [
        ak.stock_zh_index_spot_em(symbol="上证系列指数"),
        ak.stock_zh_index_spot_em(symbol="深证系列指数"),
    ]
    lines = ["## A 股主要指数行情（AKShare / 东方财富）", "", "| 指数 | 代码 | 最新价 | 涨跌幅 | 成交额 |", "| --- | --- | --- | --- | --- |"]
    found = 0
    for code, name in INDEX_NAMES.items():
        for df in frames:
            if df is None or df.empty:
                continue
            match = df[df["代码"] == code]
            if match.empty:
                continue
            row = match.iloc[0]
            amount_yi = float(row["成交额"]) / 1e8
            lines.append(f"| {name} | {code} | {row['最新价']} | {row['涨跌幅']}% | {amount_yi:.2f} 亿 |")
            found += 1
            break
    if found == 0:
        raise RuntimeError("AKShare 未返回目标指数数据")
    lines += ["", f"- 数据来源：AKShare（东方财富实时行情接口）", f"- 抓取时间：{NOW_STR}"]
    return "\n".join(lines)


def _market_from_tencent() -> str:
    """兜底数据源：腾讯行情（qt.gtimg.cn），返回最近一个交易日的收盘快照。"""
    resp = requests.get(TENCENT_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    resp.encoding = "gbk"

    lines = ["## A 股主要指数行情（腾讯行情，最近交易日收盘快照）", "", "| 指数 | 代码 | 收盘 | 涨跌幅 | 成交额 |", "| --- | --- | --- | --- | --- |"]
    data_date = ""
    for segment in resp.text.split(";"):
        segment = segment.strip()
        if not segment.startswith(("v_sh", "v_sz")) or "=" not in segment:
            continue
        payload = segment.split("=", 1)[1].strip().strip('"')
        f = payload.split("~")
        if len(f) < 38:
            continue
        name, code = f[1], f[2]
        price, pct, amount_wan = f[3], f[32], f[37]
        stamp = f[30]  # 形如 20261016161500
        if len(stamp) >= 8 and not data_date:
            data_date = f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
        amount_yi = float(amount_wan) / 1e4
        lines.append(f"| {name} | {code} | {price} | {pct}% | {amount_yi:.2f} 亿 |")

    if data_date:
        lines += ["", f"- 数据来源：腾讯行情接口 qt.gtimg.cn", f"- 数据日期：{data_date}（该日收盘快照）", f"- 抓取时间：{NOW_STR}"]
    return "\n".join(lines)


@tool
def get_market_overview() -> str:
    """获取 A 股主要指数（上证 / 深证 / 创业板 / 沪深300）的最新行情数据。

    优先走 AKShare（主模板数据源），失败时自动降级到腾讯行情接口。
    返回 Markdown 文本，包含数据日期、数据来源与指数表格。
    """
    errors = []
    for source, loader in (("akshare", _market_from_akshare), ("tencent", _market_from_tencent)):
        try:
            return loader()
        except Exception as exc:  # noqa: BLE001 - 数据源兜底
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    return "获取行情数据失败：\n- " + "\n- ".join(errors)


@tool
def get_hot_sector_news() -> str:
    """获取今日热点财经资讯（AKShare / 东方财富财经导读，取前 8 条）。"""
    try:
        import akshare as ak

        df = ak.stock_news_em(symbol="财经导读")
        if df is None or df.empty:
            return "未能获取新闻数据。"
        lines = [f"## 今日热点财经资讯（东方财富·财经导读，抓取时间 {NOW_STR}）"]
        for i, (_, row) in enumerate(df.head(8).iterrows(), 1):
            title = row.get("新闻标题", "无标题")
            source = row.get("文章来源", "未知来源")
            published = row.get("发布时间", "")
            lines.append(f"{i}. **{title}**（{source}，{published}）")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - 工具不应把异常抛给模型
        return f"获取新闻数据失败：{type(exc).__name__}: {exc}"


# ============================================================================
# 3. 创建 Agent（与主模板相同的 FilesystemBackend，仅把 root_dir 换成本作业目录）
# ============================================================================
SYSTEM_PROMPT = """你是一个专业的 A 股市场分析师，同时要用「虚拟文件系统」管理你的工作资料。

文件系统规则（必须遵守）：
- 你看到的所有文件路径都是虚拟路径（以 / 开头），例如 /materials/market_overview.md。
- 你只能通过 ls / read_file / write_file / edit_file / glob / grep 操作文件，看不到也不要去猜真实磁盘路径。
- write_file 会整体覆盖同名文件；edit_file 做精确字符串替换，old_string 必须在文件中唯一出现。
- 严格按用户当前步骤执行：只做用户明确要求的动作，不要提前写报告、不要跳步、不要并行调用写文件工具。
"""


def build_agent():
    return create_deep_agent(
        model=build_model(),
        tools=[get_market_overview, get_hot_sector_news],
        system_prompt=SYSTEM_PROMPT,
        # 与主模板同一类后端：FilesystemBackend + virtual_mode=True；
        # 区别仅在于 root_dir 指向本作业的独立目录，避免覆盖 task1 的产物。
        backend=FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True),
    )


# ============================================================================
# 4. 宿主机侧校验工具（不经 Agent，直接看磁盘）
# ============================================================================
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(virtual_path: str) -> dict:
    """直接读磁盘，记录虚拟路径对应的真实文件状态。"""
    physical = WORKSPACE / virtual_path.lstrip("/")
    if not physical.exists():
        return {"exists": False, "virtual_path": virtual_path, "physical_path": str(physical)}
    raw = physical.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    stat = physical.stat()
    return {
        "exists": True,
        "virtual_path": virtual_path,
        "physical_path": str(physical),
        "size_bytes": len(raw),
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "sha256": sha256_of(physical),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "first_line": first_line,
    }


def extract_trace(messages: list) -> dict:
    """从消息流中提取本阶段真实发生的工具调用与结果。"""
    calls, results = [], []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                calls.append({"tool": call["name"], "args": {k: str(v)[:200] for k, v in call["args"].items()}})
        elif isinstance(msg, ToolMessage):
            results.append(
                {
                    "tool": msg.name,
                    "status": getattr(msg, "status", None),
                    "content": str(msg.content)[:600],
                }
            )
    return {"tool_calls": calls, "tool_results": results}


def last_ai_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            return str(msg.content)[:1200]
    return ""


# ============================================================================
# 5. 四阶段主流程
# ============================================================================
def run() -> dict:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    agent = build_agent()
    messages: list = []
    phases: list[dict] = []

    def run_phase(label: str, prompt: str, checks) -> tuple[dict, list]:
        nonlocal messages
        print("\n" + "=" * 78)
        print(f"▶ {label}")
        print("=" * 78)
        print(prompt)
        messages.append(HumanMessage(content=prompt))
        start = len(messages) - 1
        started = time.time()
        result = agent.invoke({"messages": messages})
        elapsed = round(time.time() - started, 1)
        all_messages = result["messages"]
        new_messages = all_messages[start:]
        messages[:] = all_messages

        trace = extract_trace(new_messages)
        print(f"\n[工具调用 {len(trace['tool_calls'])} 次，耗时 {elapsed}s]")
        for call in trace["tool_calls"]:
            print(f"  → {call['tool']}({', '.join(f'{k}={v[:60]}' for k, v in call['args'].items())})")
        for res in trace["tool_results"]:
            print(f"  ← {res['tool']}: {res['content'][:200].replace(chr(10), ' / ')}")

        check_results = checks()
        for check in check_results:
            mark = "✓" if check["ok"] else "✗"
            print(f"  [{mark}] 宿主校验 · {check['name']}：{check['detail']}")

        record = {
            "phase": label,
            "prompt": prompt,
            "elapsed_sec": elapsed,
            "tool_calls": trace["tool_calls"],
            "tool_results": trace["tool_results"],
            "host_checks": check_results,
            "assistant_reply": last_ai_text(new_messages),
        }
        phases.append(record)
        return record, new_messages

    # ---------------------------------------------------------------- Phase 1
    phase1_prompt = f"""现在是 {NOW_STR}。

第一步：取数并把「资料」落盘（只做这 4 步，不要写分析报告）。
1. 调用 get_market_overview 获取 A 股主要指数最新行情（若今天休市，工具会返回最近一个交易日的收盘快照，请记住它的数据日期）。
2. 调用 get_hot_sector_news 获取今日热点财经资讯。
3. 用 write_file 把行情数据整理成 Markdown 写入 {MATERIAL_MD}，必须包含：数据日期、数据来源、以及「指数 / 点位 / 涨跌幅 / 成交额」表格。
4. 用 write_file 把热点资讯整理成 Markdown 写入 {NEWS_MD}，包含资讯条目、发布时间、来源。
完成后用一句话汇报两个文件各写了多少条数据。"""

    def check1():
        out = []
        for path, name in ((MATERIAL_MD, "行情资料"), (NEWS_MD, "资讯资料")):
            snap = snapshot(path)
            ok = snap["exists"] and snap.get("size_bytes", 0) > 60
            out.append(
                {
                    "name": f"{name}已落盘且非空",
                    "ok": bool(ok),
                    "detail": f"{snap.get('physical_path')}（{snap.get('size_bytes', 0)} B, sha256={str(snap.get('sha256', ''))[:12]}…）" if snap["exists"] else "文件不存在",
                    "snapshot": snap,
                }
            )
        return out

    run_phase("Phase 1 · 写资料（write_file）", phase1_prompt, check1)

    # ---------------------------------------------------------------- Phase 2
    phase2_prompt = f"""第二步：读回并修订「资料」（只做这 4 步）。
1. 用 read_file 读回 {MATERIAL_MD}。
2. 用 read_file 读回 {NEWS_MD}。
3. 用 edit_file 修订 {MATERIAL_MD}：在第 1 行标题之后插入一行「> 读回校验：{NOW_STR}，已将本文件读回核对，内容完整。」
   （old_string 使用第 1 行标题原文，new_string = 该标题 + 换行 + 校验行；old_string 必须与文件内容完全一致。）
4. 用 read_file 再次读回 {MATERIAL_MD}，确认校验行已写入。
最后汇报：两份资料各自的行数、资料中出现的数据日期。"""

    def check2():
        out = []
        material = WORKSPACE / MATERIAL_MD.lstrip("/")
        text = material.read_text(encoding="utf-8") if material.exists() else ""
        ok_marker = "读回校验" in text
        out.append(
            {
                "name": "资料经 edit_file 修订（出现「读回校验」行）",
                "ok": bool(ok_marker),
                "detail": "已找到校验行" if ok_marker else "未找到校验行",
                "snapshot": snapshot(MATERIAL_MD),
            }
        )
        news = snapshot(NEWS_MD)
        out.append(
            {
                "name": "资讯资料仍可读回",
                "ok": bool(news["exists"] and news.get("size_bytes", 0) > 60),
                "detail": news.get("physical_path", "文件不存在"),
                "snapshot": news,
            }
        )
        return out

    run_phase("Phase 2 · 读回并修改资料（read_file + edit_file）", phase2_prompt, check2)
    material_snapshot = snapshot(MATERIAL_MD)

    # ---------------------------------------------------------------- Phase 3
    phase3_prompt = f"""第三步：基于资料写「分析报告」（只做这 2 步）。
1. 用 read_file 读回 {MATERIAL_MD} 和 {NEWS_MD}，以文件里的最新内容为准。
2. 用 write_file 把完整报告写入 {REPORT_MD}，要求：
   - Markdown 格式，第一行标题为「# A 股市场分析报告」；
   - 必须包含数据日期，并且有「一、今日行情概览」「二、热点解读」「三、投资建议」三节；
   - 行情概览用表格给出上证指数、深证成指、创业板指的点位、涨跌幅、成交额，数字必须来自资料，不得编造；
   - 投资建议必须包含风险提示。
写完后用一句话汇报报告的行数与包含的章节。"""

    def check3():
        snap = snapshot(REPORT_MD)
        text = (WORKSPACE / REPORT_MD.lstrip("/")).read_text(encoding="utf-8") if snap["exists"] else ""
        sections = [s for s in ("今日行情概览", "热点解读", "投资建议", "风险") if s in text]
        out = [
            {
                "name": "报告已落盘且结构完整",
                "ok": bool(snap["exists"] and snap.get("size_bytes", 0) > 500 and len(sections) >= 3),
                "detail": f"{snap.get('physical_path', '文件不存在')}（{snap.get('size_bytes', 0)} B, {snap.get('lines', 0)} 行；命中章节 {sections}）",
                "snapshot": snap,
            }
        ]
        return out

    run_phase("Phase 3 · 写报告（write_file）", phase3_prompt, check3)
    report_v1 = snapshot(REPORT_MD)
    if report_v1["exists"]:
        (EVIDENCE_DIR / "report_v1.md").write_text(
            (WORKSPACE / REPORT_MD.lstrip("/")).read_text(encoding="utf-8"), encoding="utf-8"
        )

    # ---------------------------------------------------------------- Phase 4
    report_text = (WORKSPACE / REPORT_MD.lstrip("/")).read_text(encoding="utf-8")
    title_line = report_v1.get("first_line") or report_text.splitlines()[0]
    phase4_prompt = f"""第四步：读回并修订「报告」（只做这 3 步）。
1. 用 read_file 读回 {REPORT_MD}，确认报告第一行标题。
2. 用 edit_file 做一次修订：把报告第一行「{title_line}」替换为下面两行（保持原样缩进，不要改动其它内容）：
{title_line}（修订版）
> 修订记录：{NOW_STR} 宿主读回校验后修订，新增本条修订标记。
3. 用 read_file 再次读回 {REPORT_MD}，确认「（修订版）」和「修订记录」都在文件里。
最后汇报：修订前后的关键差异。"""

    def check4():
        snap = snapshot(REPORT_MD)
        text = (WORKSPACE / REPORT_MD.lstrip("/")).read_text(encoding="utf-8") if snap["exists"] else ""
        ok = bool("（修订版）" in text and "修订记录" in text and snap.get("sha256") != report_v1.get("sha256"))
        return [
            {
                "name": "报告经 edit_file 修订（哈希变化 + 修订标记出现）",
                "ok": ok,
                "detail": f"sha256 {str(report_v1.get('sha256', ''))[:12]}… → {str(snap.get('sha256', ''))[:12]}…；"
                f"大小 {report_v1.get('size_bytes', 0)} B → {snap.get('size_bytes', 0)} B",
                "snapshot": snap,
            }
        ]

    record4, _ = run_phase("Phase 4 · 读回并修改报告（read_file + edit_file）", phase4_prompt, check4)

    # 修订失败时补一次修复回合（模型偶发不按指令 edit 的兜底）
    report_v2 = snapshot(REPORT_MD)
    if not ("（修订版）" in (WORKSPACE / REPORT_MD.lstrip("/")).read_text(encoding="utf-8")):
        print("\n⚠️ 第一次修订未生效，追加一次修复回合……")
        repair_prompt = f"""上次修订没有生效。请严格按下面步骤再试一次：
1. read_file 读回 {REPORT_MD}；
2. 用 edit_file 把第一行「{title_line}」替换为「{title_line}（修订版）」；
3. read_file 再次读回确认。"""
        record4, _ = run_phase("Phase 4-repair · 修订修复", repair_prompt, check4)
        report_v2 = snapshot(REPORT_MD)

    if report_v2["exists"]:
        (EVIDENCE_DIR / "report_v2.md").write_text(
            (WORKSPACE / REPORT_MD.lstrip("/")).read_text(encoding="utf-8"), encoding="utf-8"
        )

    # ---------------------------------------------------------------- 汇总
    summary = {
        "run_at": NOW_STR,
        "model": os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash"),
        "backend": {
            "class": "FilesystemBackend",
            "virtual_mode": True,
            "root_dir": str(WORKSPACE),
            "mapping": "虚拟路径 /x/y  →  root_dir/x/y",
        },
        "virtual_paths": {"materials": [MATERIAL_MD, NEWS_MD], "report": REPORT_MD},
        "phases": phases,
        "artifacts": {
            "market_material": material_snapshot,
            "report_v1": report_v1,
            "report_v2": report_v2,
        },
        "all_checks_passed": all(check["ok"] for p in phases for check in p["host_checks"]),
    }
    (EVIDENCE_DIR / "main_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- 打印产物
    print("\n" + "=" * 78)
    print("产物定位（宿主直接读磁盘，用户可亲手在项目目录里找到）")
    print("=" * 78)
    for key, virtual in (("行情资料", MATERIAL_MD), ("资讯资料", NEWS_MD), ("分析报告", REPORT_MD)):
        snap = snapshot(virtual)
        print(f"\n{key} · 虚拟路径 {virtual}")
        print(f"  物理路径：{snap.get('physical_path')}")
        if snap["exists"]:
            print(f"  大小：{snap['size_bytes']} B ｜ 行数：{snap['lines']} ｜ sha256：{snap['sha256']}")
            print(f"  第一行：{snap['first_line']}")
    print(f"\n阶段全通过：{summary['all_checks_passed']}")
    print(f"证据文件：{EVIDENCE_DIR / 'main_run.json'}")
    print("\n查看命令：")
    print(f"  ls -l {WORKSPACE / 'materials'} {WORKSPACE / 'reports'}")
    print(f"  shasum -a 256 {WORKSPACE / 'reports' / 'a_stock_report.md'}")
    return summary


if __name__ == "__main__":
    run()
