from semantic_similarity import compute_similarity, evaluate_sentences
from sentence_transformers import SentenceTransformer
import requests
import json
import time

# ==== 模型预加载 ====
print("✅ 正在加载向量模型，请稍等……")
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 提前缓存模型，避免启动时提示


# ==== ChatChat API 调用 ====
def get_chatgpt_sentence(prompt):
    API_ENDPOINT = "http://695ccc08.r31.cpolar.top/chat/chat/completions"#新建的临时隧道，可能定期变动
    HEADERS = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            ai_reply = result["choices"][0]["message"]["content"]
            return ai_reply.strip()
        else:
            print(f"ChatChat 请求失败，状态码：{response.status_code}")
            return None
    except Exception as e:
        print("ChatChat 请求异常：", e)
        return None


# ==== ChatChat 重试机制 ====
def retry_get_chatgpt_sentence(prompt, retries=3, delay=5):
    for attempt in range(retries):
        sentence = get_chatgpt_sentence(prompt)
        if sentence:
            return sentence
        else:
            print(f"⚠️ 第 {attempt + 1} 次尝试失败，等待 {delay} 秒后重试...")
            time.sleep(delay)
    print("❌ ChatChat 多次请求失败，跳过该模型。")
    return None


# ==== 通义千问 API 调用 ====
def get_qianwen_sentence(prompt):
    API_KEY = "你的API-KEY"
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-plus",
        "input": {
            "prompt": prompt
        },
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 200
        }
    }

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        if response.status_code == 200:
            data = response.json()
            ai_reply = data["output"]["text"]
            return ai_reply.strip()
        else:
            print(f"千问请求失败，状态码：{response.status_code}, 错误信息：{response.text}")
            return None
    except Exception as e:
        print("千问请求异常：", e)
        return None


# ==== 单次评估逻辑 ====
def run_similarity_evaluation(prompt):
    print("\n📌 正在请求模型生成句子...")
    sentence1 = retry_get_chatgpt_sentence(prompt)
    sentence2 = get_qianwen_sentence(prompt)

    if sentence1 and sentence2:
        print(f"\n🧠 ChatChat 生成: {sentence1}")
        print(f"🧠 千问生成: {sentence2}")

        print("\n🔍 向量模型评估结果：")
        vector_result = compute_similarity(sentence1, sentence2)
        for key, value in vector_result.items():
            print(f"{key}: {value}")

        print("\n🔍 大模型评估结果：")
        llm_result = evaluate_sentences(sentence1, sentence2)
        for key, value in llm_result.items():
            print(f"{key}: {value}")
    else:
        print("\n⚠️ 未能获取模型生成的句子，跳过本轮评估。")


# ==== 程序主循环 ====
if __name__ == "__main__":
    default_prompt = "请生成一个情境下的句子，描述人们在春天户外散步的场景。"

    print("🎯 程序启动中，先进行一次默认示例评估：请生成一个情境下的句子，描述人们在春天户外散步的场景。")
    run_similarity_evaluation(default_prompt)

    while True:
        user_prompt = input("\n✏️ 请输入一个新的提示词（回车退出）：\n>> ").strip()
        if not user_prompt:
            print("👋 程序结束，再见！")
            break
        run_similarity_evaluation(user_prompt)
