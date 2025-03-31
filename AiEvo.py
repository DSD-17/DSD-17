import requests
import json
import re

# DeepSeek API Key
API_KEY = "你的API KEY"
API_ENDPOINT = "你的API端口"

# 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def evaluate_sentences(sentence1, sentence2):
    """评估两句话的语义相似度"""

    # 构造 AI 提示词
    prompt = f"""
    请比较以下两句话的语义相似度，并说明你的推理过程：
    句子 1: {sentence1}
    句子 2: {sentence2}

    你是一名 NLP 领域的 AI 评估专家。请对以下两句话进行语义相似度评估，并严格按照以下要求执行：

    1. **计算方法**：
       - 解析两句话的 **关键词**、**语法结构**、**词向量** 之间的关系。
       - 计算词级别的匹配程度，并考虑句法依赖关系。
       - 如果可能，估算基于 TF-IDF / 词向量（如 Word2Vec / BERT）等技术的相似度评分。

    2. **相似度评分**（范围 0 ~ 100）：
       - 100 表示两句话完全相同。
       - 80 ~ 99 说明两句话在意义上高度相似，只有轻微的词汇或语法变动。
       - 60 ~ 79 说明两句话部分相似，但可能有一定程度的信息缺失或变更。
       - 30 ~ 59 说明两句话只有部分相关信息，表达方式有较大不同。
       - 0 ~ 29 说明两句话的意思完全不同。

    3. **请给出详细的计算过程**，包括：
       - 关键词分析（找出核心关键词）
       - 句法分析（句子结构比较）
       - 词向量分析（如果适用）
       - 计算得到的相似度评分

    4. **最终输出格式（严格按照 JSON 返回）**：
    {{
        "相似度评分": 90,
        "关键词匹配": ["天气", "公园", "散步/走走"],
        "句法分析": "句子 1 是陈述句，句子 2 是疑问句，但核心主谓宾结构一致。",
        "词向量分析": "基于 BERT 计算的词向量余弦相似度为 0.88。",
        "计算推导过程": "..."
    }}
    """

    # 构造请求数据
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    # 发送请求
    response = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps(payload))

    # 解析响应
    if response.status_code == 200:
        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]

        # 处理 AI 返回内容，去掉 Markdown 代码块
        ai_reply = re.sub(r"```json\n(.*?)\n```", r"\1", ai_reply, flags=re.DOTALL).strip()

        try:
            ai_json = json.loads(ai_reply)

            # 让 "计算推导过程" 里的 `\n` 变成真正的换行
            if "计算推导过程" in ai_json:
                ai_json["计算推导过程"] = ai_json["计算推导过程"].replace("\\n", "\n")

            # 美化输出
            print("=" * 40)
            print(f"✅ **输入的句子**:")
            print(f"📌 句子 1: {sentence1}")
            print(f"📌 句子 2: {sentence2}")
            print("=" * 40)

            print(f"🎯 **相似度评分**: {ai_json.get('相似度评分', '未知')}")
            print("=" * 40)

            print(f"🔍 **关键词匹配**: {ai_json.get('关键词匹配', '未知')}")
            print("=" * 40)

            print(f"📖 **句法分析**:\n{ai_json.get('句法分析', '未知')}")
            print("=" * 40)

            print(f"🧠 **词向量分析**:\n{ai_json.get('词向量分析', '未知')}")
            print("=" * 40)

            print(f"📌 **计算推导过程**:\n{ai_json.get('计算推导过程', '未知')}")
            print("=" * 40)

        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败：{e}")
            print("可能的原因：AI 返回的内容不是标准 JSON，请检查 `ai_reply`。")

    else:
        print(f"请求失败，错误代码：{response.status_code}")
        print(response.text)


# **🌟 1. 先进行一次示例评估**
print("\n🔍 **示例评估**（程序启动时自动执行）")
example_sentence1 = "今天天气很好，我们去公园散步吧。"
example_sentence2 = "天气不错，咱们一起去公园走走？"
evaluate_sentences(example_sentence1, example_sentence2)

# **🌟 2. 让用户输入新的句子**
while True:
    print("\n📢 请输入两句话进行评估（按回车确认，每次输入一行）")
    user_sentence1 = input("✏️ 句子 1: ").strip()
    user_sentence2 = input("✏️ 句子 2: ").strip()

    if user_sentence1 and user_sentence2:
        print("\n🚀 **开始评估**...\n")
        evaluate_sentences(user_sentence1, user_sentence2)
    else:
        print("\n❌ 请输入有效的句子！")
#今天是三月的最后一天
#明天是四月的第一天