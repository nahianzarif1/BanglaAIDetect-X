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
print("\nPrediction:")
print(result)