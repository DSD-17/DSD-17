import os
import numpy as np
from sentence_transformers import SentenceTransformer

# 屏蔽 Hugging Face 符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 加载支持中文的轻量级向量模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def compute_similarity(text1, text2):
    """计算两句话的余弦相似度，并展示详细计算过程"""

    print("=" * 50)
    print(f"✅ **输入的句子**:")
    print(f"📌 句子 1: {text1}")
    print(f"📌 句子 2: {text2}")
    print("=" * 50)

    print("🚀 **正在计算相似度，请稍候...**\n")

    # 计算文本向量
    vector1 = model.encode(text1, convert_to_numpy=True)
    vector2 = model.encode(text2, convert_to_numpy=True)

    print("🧬 **文本向量（部分显示）**:")
    print(f"🔹 向量1（前5维）：{vector1[:5]}...")
    print(f"🔹 向量2（前5维）：{vector2[:5]}...\n")

    # 计算余弦相似度
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm1 * norm2)

    # 打印计算过程
    print("📝 **计算过程**:")
    print(f"🔹 向量1 · 向量2（点积）= {dot_product:.4f}")
    print(f"🔹 ||向量1||（模长）= {norm1:.4f}")
    print(f"🔹 ||向量2||（模长）= {norm2:.4f}")
    print(f"🔹 余弦相似度 = {dot_product:.4f} / ({norm1:.4f} * {norm2:.4f})")
    print("=" * 50)
    print(f"🎯 **最终相似度**: {cosine_similarity:.4f}")
    print("=" * 50)

# **🌟 1. 先进行一次示例评估**
print("\n🔍 **示例评估**（程序启动时自动执行）")
example_sentence1 = "今天天气很好，我们去公园散步吧。"
example_sentence2 = "天气不错，咱们一起去公园走走？"
compute_similarity(example_sentence1, example_sentence2)

# **🌟 2. 让用户输入新的句子**
while True:
    print("\n📢 请输入两句话进行评估（按回车确认，每次输入一行）")
    user_sentence1 = input("✏️ 句子 1: ").strip()
    user_sentence2 = input("✏️ 句子 2: ").strip()

    if user_sentence1 and user_sentence2:
        print("\n🚀 **开始评估**...\n")
        compute_similarity(user_sentence1, user_sentence2)
    else:
        print("\n❌ 请输入有效的句子！")
#今天是三月的最后一天
#明天是四月的第一天
