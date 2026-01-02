# check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ .env 파일에서 API 키를 찾을 수 없습니다.")
else:
    genai.configure(api_key=api_key)
    print("🔍 내 키로 사용 가능한 모델 목록:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")