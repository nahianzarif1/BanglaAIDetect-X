import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.inference import BanglaAIDetector

# Test with multiple AI samples from the pasted data
ai_samples = [
    "শোবার আগে মোবাইল ব্যবহার কমাও এবং হালকা গান শুনলে মন শান্ত হবে।",
    "কাজগুলো ছোট অংশে ভাগ করো এবং একেকটা শেষ করার পর বিরতি নাও।",
    "তোমার চিন্তা স্বাভাবিক। পরিবারের সাথে খোলাখুলি কথা বললে চাপ কিছুটা কমতে পারে।",
    "প্রতি ঘন্টায় ১০ মিনিট বিরতি নাও এবং গুরুত্বপূর্ণ বিষয়গুলো বারবার পুনরাবৃত্তি করো।",
    "মনে রেখো, ভুল সবারই হয়। নিজেকে দোষারোপ না করে ধীরে ধীরে উন্নতির দিকে মন দাও।",
]

detector = BanglaAIDetector()

for i, text in enumerate(ai_samples, 1):
    result = detector.predict(text)
    print(f"\n=== Sample {i} ===")
    print(f"Text: {text}")
    print(f"Length: {len(text.split())} words")
    print(f"Label: {result['label']}")
    print(f"AI Probability: {result['ai_probability']}")
    print(f"Human Probability: {result['human_probability']}")
    print(f"Confidence: {result['confidence']}")