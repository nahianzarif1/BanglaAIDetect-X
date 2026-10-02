import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.inference import BanglaAIDetector

# Test with a sample from the new AI dataset
ai_text = "শোবার আগে মোবাইল ব্যবহার কমাও এবং হালকা গান শুনলে মন শান্ত হবে।"

detector = BanglaAIDetector()
result = detector.predict(ai_text)

print("AI Text from new dataset:")
print(ai_text)
print(f"Length: {len(ai_text.split())} words")
print("\nPrediction:")
print(result)

# Also test a longer version
longer_ai = ai_text + " " + "এছাড়াও বিছানায় যাওয়ার আগে একটি গরম গোসল নিলে মন শান্ত হবে। সর্বোপরি প্রতিদিন একই সময়ে ঘুমানোর চেষ্টা করো।"
print("\n\nLonger AI text:")
print(longer_ai)
print(f"Length: {len(longer_ai.split())} words")
print("\nPrediction:")
print(detector.predict(longer_ai))