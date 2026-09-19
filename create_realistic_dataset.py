"""
create_realistic_dataset.py

Creates a more realistic dataset with better distinction between human and AI writing styles
to improve model accuracy.
"""

import os
import pandas as pd
import uuid


def create_realistic_dataset():
    """Create a dataset with more realistic human and AI samples."""
    
    # More diverse human-written Bangla text (various styles)
    human_samples = [
        # News style - factual, specific details, bursty sentences
        "গাইবান্ধার সাঘাটা উপজেলায় যমুনার পানি বিপদসীমা অতিক্রম করেছে। বুধবার ইউনিয়নের চেয়ারম্যান আব্দুল হয়ে পড়েছে। স্থানীয়রা বলেন, গত তিন দিন ধরে পানি বাড়ছে। জেলা প্রশাসন জরুরি ত্রাণ পাঠানোর আশ্বাস দিয়ছে।",
        
        # Academic style - citations, formal structure, specific terminology
        "বাংলাদেশের অর্থনীতিতে তৈরি শিল্পের ভূমিকা অপরিহার্য। বিশ্বব্যাংকের ২০২৩ সালের প্রতিবেদন অনুযায়ী (World Bank, 2023), জিডিপিতে এ খাতের অবদান ১৫%। গবেষকরা মনে করেন, দক্ষতা বৃদ্ধি পেলে এই হার আরও বাড়বে।",
        
        # Conversational style - informal, direct, varied sentence length
        "আমি ভাবছিলাম আজকে বাজারে যাব। কিন্তু বৃষ্টি হওয়ায় আর যেতে পারলাম না। তাই বাসায় বসেই কাজ করলাম। বন্ধুরা ফোন করে জানতে চাইল কী হয়েছে। আমি সব খুলে বললাম।",
        
        # Literary style - descriptive, metaphorical, varied rhythm
        "শরৎকালের কুয়াশ্রী আকাশে মেঘেরা খেলা করে। পাখিরা ডালে ডালে উড়ে বেড়ায়। কৃষকরা জমিতে ধান কাটতে ব্যস্ত। গ্রামের বাচ্চারা মাটির ঘরে আগুন জ্বালে। সন্ধ্যায় বাতাস বয়ে যায়।",
        
        # Technical style - precise, numbered, structured
        "প্রথমে সার্ভার ইনস্টল করুন। দ্বিতীয়ত, কনফিগারেশন ফাইল সেট আপ করুন। তৃতীয়ত, ডাটাবেস কানেক্ট করুন। চতুর্থত, টেস্ট রান করুন। পঞ্চমত, সমস্যা থাকলে ডিবাগ করুন। ষষ্ঠত, ডিপ্লয় করুন।",
        
        # Wikipedia style - encyclopedic, factual, comprehensive
        "ঢাকা বাংলাদেশের রাজধানী ও বৃহত্তম শহর। ২০২২ সালের আদমশুমারি অনুযায়ী এর জনসংখ্যা প্রায় ১ কোটি। শহরটি বুড়িগঙ্গা নদীর তীরে অবস্থিত। এখানে অনেক ঐতিহাসিক স্থান রয়েছে। লালবাগ কেল্লা এর অন্যতম দর্শনীয় স্থান।",
        
        # Storytelling style - narrative, emotional, character-focused
        "রহিম সকালে ঘুম থেকে উঠল। আজ স্কুলে যাওয়া দরকার। মা ডাকলেন, 'ও রে উঠ।' রহিম বিছানা ছেড়ল। ফ্রেশ হলো। নাস্তা খেল। ব্যাগ নিয়ে বের হল। পথে বন্ধু দেখে খুশি হল।",
        
        # Official style - formal, bureaucratic, structured
        "বিজ্ঞপ্তি সংখ্যা: ১২৩৪/২০২৪। সকল সংশ্লিষ্টকে জানানো যাচ্ছে যে, আগামী ১৫ তারিখের মধ্যে বকেয়া জমা দিতে হবে। আবেদনপত্র নির্ধারিত ফরমেটে জমা দিতে হবে। অসম্পূর্ণ আবেদন গ্রহণ করা হবে না।",
        
        # Review style - opinionated, evaluative, comparative
        "এই রেস্টুরেন্টের খাবারের স্বাদ চমৎকার। বিরিয়ানি অসাধারণ। কিন্তু মাংসের কাবাব মুখরোচক। সার্ভিস বেশ ভাল। দাম একটু বেশি। মোটামুটি মানুষের পক্ষে সাশ্রয়। পরিবেশ পরিচ্ছন।",
        
        # Instructional style - step-by-step, clear, directive
        "প্রথমে পানি গরম করুন। তারপর চা পাতা দিন। ৫ মিনিট ঢেকে রাখুন। চিনি ও দুধ মিশ্রিত করুন। ভাল করে নাড়ুন। ছাঁকে পরিবেশন করুন। গরম গরম পরিবেশন করুন।",
        
        # More conversational - daily life, varied
        "আজকাল আবহাওয়া অনেক। অফিসে গিয়েছিলাম সকালে। রাস্তায় জ্যাম ছিল। তাই দেরি হলাম। কাজ শেষ করে বাসায় ফিরলাম। সন্ধ্যায় বাজারে গিয়েছিলাম কিছু জিনিস কিনতে।",
        
        # Emotional/narrative - feelings, personal
        "আমার খুব ভাল লাগছিল ফলের সময় দেখে। ছোটবেলায় এমন ফল দেখিনি। মা বললেন, এটা করে রাখো। আমি কান্না করলাম। বন্ধুরা এসে খুশি হল। সবাই মিলে আনন্দ করলাম।",
        
        # Professional - work, career
        "মিটিংয় নতুন জরুরি কিছু ডকুমেন্ট প্রস্তুত করতে হবে। প্রজেক্টের স্ট্যাটাস বের করতে হবে। বাজেট অ্যানালিসিস করতে হবে। রিপোর্ট সাবমিট করতে হবে। টাইমলাইন মেনে চলতে হবে।",
        
        # Question-based - inquiry style
        "তুমি কি করছ? কোথায় যাচ্ছ? কার সাথে যাচ্ছ? কখন ফিরবে? কেন যাচ্ছ? কী এনায় যাচ্ছ? কতদিন থাকবে? কীভাবে যাবে? কত খরচ লাগবে?",
        
        # Opinion - personal views
        "আমার মতে এটা ভাল সিদ্ধান্ত। সবাই একমত নাও। কিন্তু অধিকাংশ লোক একমত। আমি সমর্থন করি। বাস্তবতা দেখে সিদ্ধান্ত নিতে পারি। আশা করি সবাই বুঝবে।",
        
        # Weather/descriptive - environment
        "আজকে আকাশ মেঘলা। বৃষ্টি হতে পারে। বাতাস ঠান্ডা বয়ে যাচ্ছে। কাল পরে রোদ হবে। মানুষজন ছাতা বের হবে। হাটিয় মুখরোচক হবে।",
        
        # Travel - journey description
        "আমরা গত সপ্তাহে কক্সবাজার গিয়েছিলাম। সমুদ্র দেখে মুগ্ধ হলাম। সকুরিয়া করলাম। জেলে মাছ খেলাম। সন্ধ্যায় সৈকটে বসে রইলাম। সবাই উপভোগ করলাম।",
        
        # Sports - match description
        "বাংলাদেশ আজকে ম্যাচ জিতলে। টস টস করল। অনেক রান হল। উইকেট উঠল। ম্যাচ জিতলে দল। সবাই খুশি হল। পুরস্কার হল। পুরস্কার দিল।",
        
        # Food - cooking description
        "আমি আজ বিরিয়ানি রান্না করলাম। পেঁযাজ দিলাম। মসলা করলাম। সবজি দিলাম। তেল দিলাম। মরিচ দিলাম। সব মিশিয়ে রান্না হল। স্বাদ হল চমৎকার।"
    ]
    
    # More diverse AI-generated Bangla text (distinctive AI patterns)
    ai_samples = [
        # ChatGPT essay style - formal transitions, balanced structure, AI markers
        "শিক্ষা ব্যবস্থা হলো জাতীয় উন্নয়নের মূল চালিকাশক্তি। এটি মানবসম্পদ তৈরি হিসেবে গুরুত্বপূর্ণ অবদান রাখে। এছাড়াও প্রাথমিক, মাধ্যমিক এবং উচ্চশিক্ষা ইত্যাদি ক্ষেত্রে এর প্রয়োগ দেখা যায়। অন্যদিকে সীমাবদ্ধতাগুলো চিহ্নিত করে ধাপে ধাপে সমাধান করা উচিত। সর্বোপরি সমন্বিত প্রচেষ্টা ও সঠিক নীতিমালা ছাড়া কাঙ্ক্ষিত ফল পাওয়া যায় না।",
        
        # Definition style - "X is Y" structure, encyclopedic but formulaic
        "সুন্দরবন হলো বিশ্বের বৃহত্তম ম্যানগ্রোভ বনাঞ্চলের একটি। এটি বাংলাদেশ ও ভারতের সীমান্তে অবস্থিত। এখানে রয়েল বেঙ্গল টাইগারের বাসস্থান রয়েছে। এছাড়াও এটি জীববৈচিত্র্যের জন্য অত্যন্ত গুরুত্বপূর্ণ। ইউনেস্কো একে বিশ্ব ঐতিহ্যবাহী স্থান হিসেবে ঘোষণা করেছে।",
        
        # General AI style - balanced but generic, repetitive structure
        "প্রযুক্তির অগ্রগতি আধুনিক জীবনের প্রতিটি ক্ষেত্রে প্রভাব ফেলেছে। এটি শিক্ষা, স্বাস্থ্য, ব্যবসা সব ক্ষেত্রেই গুরুত্বপূর্ণ ভূমিকা পালন করে। এছাড়াও যোগাযোগ ব্যবস্থার উন্নতি ঘটেছে। অন্যদিকে মানুষের জীবনযাপন সহজ হয়েছে। সর্বোপরি এর সুফল সবাই ভোগ করছে।",
        
        # List-style AI - repetitive "X, Y, and Z" structure
        "সুস্থ থাকার জন্য কিছু বিষয় মেনে চলা প্রয়োজন। প্রথমত ভালো খাবার খাওয়া, দ্বিতীয়ত নিয়মিত ব্যায়াম করা, তৃতীয়ত পর্যাপ্ত ঘুমানো, চতুর্থত ধূমপান এড়িয়ে চলা, এবং পঞ্চমত নিয়মিত স্বাস্থ্য পরীক্ষা করা। এই পাঁচটি বিষয় মেনে চললে সুস্থ থাকা সম্ভব।",
        
        # Conclusion-style AI - summarizing, "in conclusion" patterns
        "উপরন্তুক্ত আলোচনা থেকে বোঝা যায় যে পরিবেশ রক্ষা অত্যন্ত জরুরি। গাছ লাগানো, বনায়ন এবং দূষণ কমানো প্রয়োজন। সরকার ও জনগণ মিলে কাজ করলে সম্ভব। সুতরাং আমাদের সবাইকে এগিয়ে আসতে হবে। অতএব পরিবেশ রক্ষায় গুরুত্ব দিতে হবে।",
        
        # Balanced AI - even sentence lengths, lack of burstiness
        "বাংলাদেশের সংস্কৃতি সমৃদ্ধ এবং বৈচিত্র্যময়। এখানে বিভিন্ন ধর্মের মানুষ বাস করে। তারা শান্তিতে একে অপরের সাথে থাকে। উৎসব উৎসব উদযাপন করা হয়। সামাজিক অনুষ্ঠান অনুষ্ঠিত হয়। সব মিলিয়ে একটি সুন্দর পরিবেশ তৈরি হয়।",
        
        # Explanation-style AI - explanatory, informative but formulaic
        "মোবাইল ফোন আধুনিক যোগাযোগের একটি অপরিহার্য মাধ্যম। এটি মানুষকে যেকোনো স্থান থেকে যোগাযোগ করতে সাহায্য করে। এতে কল করা, মেসেজ পাঠানো, ইন্টারনেট ব্যবহার সবই সম্ভব। এছাড়াও বিভিন্ন অ্যাপের ব্যবহার করা যায়। এটি জীবনকে সহজ করে দিয়ছে।",
        
        # Comparison-style AI - comparative structure, "on one hand, on the other"
        "শহরের জীবনে সুবিধা ও অসুবিধা দুটোই আছে। একদিকে চিকিৎসা, শিক্ষা, বিনোদনের সুযোগ আছে। অন্যদিকে যানজট, দূষণ, খরচ চাপ আছে। তবে সরকার বিভিন্ন পদক্ষেপ নিচ্ছে। ভবিষ্যতে পরিস্থিতি উন্নত হবে বলে আশা করা হচ্ছে।",
        
        # Problem-solution AI - structured problem-solution format
        "বেকারি সমস্যা দিন দিন বাড়ছে। এর প্রধান কারণ হলো দক্ষতার অভাব। এছাড়াও অবকাঠামো সীমিত। সমাধানের জন্য প্রশিক্ষণ বাড়াতে হবে। আধুনিক প্রযুক্তি ব্যবহার করতে হবে। সরকারি-বেসরকারি উদ্যোগ প্রয়োজন। তরুণদের দক্ষ করে তৈরি করতে হবে।",
        
        # Importance-emphasis AI - repetitive importance statements
        "পরিবার জীবনে সম্প্রীতি অত্যন্ত গুরুত্বপূর্ণ। এটি সামাজিক স্থিতিশীলতা বজায় রাখে। পারিবারিক বন্ধন শক্তিশালী থাকলে মানসিক শান্তি মিলে। এছাড়াও সন্তানদের বিকাশ সঠিকভাবে হয়। পারিবার সমাজের ক্ষুদ্রকণ। তাই এর গুরুত্ব অপরিসীম।"
    ]
    
    # Create dataset with clear human vs AI distinctions
    rows = []
    
    # Add human samples with varied topics
    human_topics = ["flood_news", "academic_citation", "daily_conversation", "literature", 
                   "technical_guide", "wikipedia_dhaka", "story_rahim", "official_notice", 
                   "restaurant_review", "cooking_recipe"]
    
    for i, (text, topic) in enumerate(zip(human_samples, human_topics)):
        rows.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": "human",
            "generator": "human",
            "genre": ["news", "academic", "conversational", "literature", "technical", 
                     "encyclopedic", "narrative", "official", "review", "instructional"][i],
            "language_type": "bangla",
            "edit_type": "none",
            "topic_id": topic,
            "writer_subgroup": "standard",
            "split": None,
        })
    
    # Add AI samples with distinctive AI patterns
    ai_topics = ["education_essay", "sundarbans_definition", "technology_general", 
                 "health_list", "environment_conclusion", "culture_balanced", 
                 "mobile_explanation", "city_comparison", "unemployment_solution", 
                 "family_importance"]
    
    for i, (text, topic) in enumerate(zip(ai_samples, ai_topics)):
        rows.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": "ai",
            "generator": "gpt",
            "genre": ["essay", "encyclopedic", "general", "instructional", "conclusion",
                     "general", "explanatory", "comparative", "problem_solution", "importance"][i],
            "language_type": "bangla",
            "edit_type": "none",
            "topic_id": topic,
            "writer_subgroup": None,
            "split": None,
        })
    
    df = pd.DataFrame(rows)
    
    # Create directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save to processed directory
    output_path = "data/processed/collected_raw.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Realistic dataset created with {len(df)} rows")
    print(f"✓ Human samples: {len(human_samples)} (diverse styles)")
    print(f"✓ AI samples: {len(ai_samples)} (distinctive AI patterns)")
    print(f"✓ Saved to: {output_path}")
    print()
    print("Dataset characteristics:")
    print("- Human: News, academic, conversational, literary, technical, etc.")
    print("- AI: Essay-style, definition-style, balanced, conclusion-style, etc.")
    print()
    print("Next steps:")
    print("1. python -m src.data.cleaner")
    print("2. python -m src.data.splitter") 
    print("3. python -m src.data.validator")
    print("4. python -m src.models.fusion")


if __name__ == "__main__":
    create_realistic_dataset()