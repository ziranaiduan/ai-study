import akshare as ak
import pandas as pd
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv


load_dotenv()

# ========== 1. 模型配置 ==========
model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
)


# ========== 2. 财经数据工具 ==========
@tool
def get_market_overview() -> str:
    """获取今日 A 股主要指数行情数据（上证、深证、创业板）。"""
    try:
        # 获取实时指数行情
        df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
        sz_df = ak.stock_zh_index_spot_em(symbol="深证系列指数")

        result_lines = ["## A 股主要指数行情\n"]

        # 筛选主要指数
        main_indices = ["000001", "399001", "399006"]
        for idx_code in main_indices:
            for source_df in [df, sz_df]:
                if source_df is not None and not source_df.empty:
                    match = source_df[source_df["代码"] == idx_code]
                    if not match.empty:
                        row = match.iloc[0]
                        result_lines.append(
                            f"- **{row['名称']}**：{row['最新价']}，"
                            f"涨跌幅 {row['涨跌幅']}%，"
                            f"成交额 {row['成交额'] / 1e8:.2f}亿"
                        )

        if len(result_lines) == 1:
            return "未能获取指数数据，请检查 AKShare 版本。"

        return "\n".join(result_lines)
    except Exception as e:
        return f"获取行情数据失败：{e}"


@tool
def get_hot_sector_news() -> str:
    """获取今日热点板块和财经新闻资讯。"""
    try:
        # 获取财经新闻（东方财富热点）
        news_df = ak.stock_news_em(symbol="财经导读")

        if news_df is None or news_df.empty:
            return "未能获取新闻数据。"

        # 取前 8 条新闻
        top_news = news_df.head(8)
        result_lines = ["## 今日热点财经资讯\n"]

        for i, (_, row) in enumerate(top_news.iterrows(), 1):
            title = row.get("新闻标题", row.get("标题", "无标题"))
            source = row.get("文章来源", row.get("来源", "未知来源"))
            result_lines.append(f"{i}. **{title}**（{source}）")

        return "\n".join(result_lines)
    except Exception as e:
        return f"获取新闻数据失败：{e}"


@tool
def analyze_and_recommend(market_data: str, news_data: str) -> str:
    """基于行情数据和新闻资讯，生成市场分析和投资建议。

    Args:
        market_data: 指数行情数据文本
        news_data: 财经新闻资讯文本
    """
    # 这个工具本身不执行逻辑，由主 Agent 的 LLM 驱动分析
    # 返回提示模板，引导模型进行结构化分析
    return f"""请基于以下数据进行分析：

{market_data}

{news_data}

请输出：
1. 今日市场整体走势判断（一句话总结）
2. 热点板块分析（2-3 个值得关注的板块）
3. 投资建议（风险提示 + 操作建议）
"""


# ========== 3. 创建 Agent（启用 FilesystemBackend） ==========
agent = create_deep_agent(
    model=model,
    tools=[get_market_overview, get_hot_sector_news, analyze_and_recommend],
    system_prompt="""你是一个专业的 A 股市场分析师。

工作流程：
1. 先用 get_market_overview 获取指数行情
2. 用 get_hot_sector_news 获取热点资讯
3. 综合以上信息，给出你的分析判断和投资建议
4. 用 write_file 工具将完整报告写入 /reports/a_stock_report.md

报告格式要求：
- 使用 Markdown
- 包含：行情概览、热点解读、投资建议
- 投资建议需包含风险提示
- 文件路径使用 /reports/ 前缀""",
    backend=FilesystemBackend(root_dir="../agent_workspace", virtual_mode=True),
)

# ========== 4. 运行 ==========
print("正在获取 A 股行情并生成分析报告...")
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请获取今日 A 股行情资讯，分析市场走势，给出投资建议，并将报告写入 markdown 文件,要求必须有今天的大盘情况。"
    }]
})

# ========== 5. 验证文件是否落盘 ==========
import os

report_path = "../agent_workspace/reports/a_stock_report.md"
if os.path.exists(report_path):
    print(f"\n✅ 报告已写入：{report_path}")
    print("\n" + "=" * 60)
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print("❌ 文件未找到，请检查 backend 配置")
    print("最终消息：", result["messages"][-1].content)