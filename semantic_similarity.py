# semantic_similarity.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import re

# 环境变量设置
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 向量模型加载
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 大模型 API 配置
API_KEY = "sk-f95378059e7e4ad5b37176f09cc4b897"
API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def interpret_similarity(score):
    """将余弦相似度映射为人类可读的相似度描述"""
    if score > 0.9:
        return "两句话几乎完全相同。"
    elif score > 0.75:
        return "两句话在语义上高度相似。"
    elif score > 0.5:
        return "两句话部分相似，有一定差异。"
    elif score > 0.3:
        return "两句话语义相关性较弱。"
    else:
        return "两句话几乎没有相似性。"

# 向量模型计算相似度
def compute_similarity(text1, text2):
    vector1 = model.encode(text1, convert_to_numpy=True)
    vector2 = model.encode(text2, convert_to_numpy=True)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm1 * norm2)

    return {
        "向量1样本": vector1[:5].tolist(),
        "向量2样本": vector2[:5].tolist(),
        "点积": float(dot_product),
        "模长": [float(norm1), float(norm2)],
        "相似度分数": float(cosine_similarity),
        "相似度判断": interpret_similarity(cosine_similarity)
    }


# 大模型调用
def evaluate_sentences(sentence1, sentence2):
    print("🤖 正在调用大模型进行语义评估，请稍候...\n")
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
       - 80 ~ 99 表示高度相似。
       - 60 ~ 79 表示部分相似。
       - 30 ~ 59 表示关系较远。
       - 0 ~ 29 表示语义无关。

    3. **输出格式（严格 JSON）**：
    {{
        "相似度评分": "...",
        "关键词匹配": ["..."],
        "句法分析": "...",
        "词向量分析": "...",
        "计算推导过程": "..."
    }}
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]
        ai_reply = re.sub(r"```json\n(.*?)\n```", r"\1", ai_reply, flags=re.DOTALL).strip()

        try:
            ai_json = json.loads(ai_reply)
            if "计算推导过程" in ai_json:
                ai_json["计算推导过程"] = ai_json["计算推导过程"].replace("\\n", "\n")
            return ai_json
        except json.JSONDecodeError:
            return {"error": "无法解析大模型返回的 JSON", "raw_reply": ai_reply}
    else:
        return {
            "error": f"请求失败，状态码：{response.status_code}",
            "response_text": response.text
        }
