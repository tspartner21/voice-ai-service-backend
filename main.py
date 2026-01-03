import os
import shutil
import json
import sqlite3
import base64
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/images", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_NAME = "bookings.db"

# --- DB 초기화 ---
def init_db():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', full_name TEXT, phone TEXT, address TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS products (id TEXT PRIMARY KEY, category TEXT, title TEXT, price TEXT, rating TEXT, image_url TEXT, desc TEXT, persona TEXT, situation TEXT, mission TEXT, examples TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS bookings (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, theme_id TEXT, theme_title TEXT, start_date TEXT, end_date TEXT, people INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

            cursor.execute("INSERT OR IGNORE INTO users (username, password, role, full_name) VALUES ('admin', 'admin', 'admin', 'Admin')")
            cursor.execute("INSERT OR IGNORE INTO users (username, password, role, full_name) VALUES ('user', 'user', 'user', 'Tester')")

            seed_data = [
                ("kpop", "basic", "🎤 K-POP 콘서트", "Free", "5.0", "", "콘서트장 상황극", "열정적인 MC", "콘서트장", "응원하기", '["Scream!", "Encore!"]'),
                ("store", "basic", "🏪 편의점 알바", "Free", "5.0", "", "편의점 상황극", "친절한 알바생", "편의점", "계산하기", '["How much?", "I need a bag."]'),
                ("date", "basic", "💕 홍대 첫 데이트", "Free", "5.0", "", "데이트 상황극", "설레는 상대방", "홍대 맛집", "주문하기", '["You look pretty.", "Lets eat."]'),
                ("offline_hongdae", "offline", "🔥 홍대 언어교환", "35,000원", "4.9", "https://via.placeholder.com/400", "현지인 친구", "모임장", "언어교환", "자기소개", '["Hello"]')
            ]
            for p in seed_data:
                cursor.execute("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?)", p)
            conn.commit()
    except Exception as e:
        print(f"DB Init Error: {e}")

init_db()

# --- Models ---
class AuthRequest(BaseModel):
    username: str; password: str
class RegisterRequest(BaseModel):
    username: str; password: str; full_name: str; phone: str; address: str
class BookingRequest(BaseModel):
    username: str; theme_id: str; start_date: str; end_date: str; people: int
class CancelRequest(BaseModel):
    booking_id: int

# --- [Deep Tech Algorithm] 고도화된 오디오 유사도 분석 ---
def analyze_audio_similarity(user_path, target_path):
    print(f"📡 [Deep Tech] 신호 정밀 분석 시작: {user_path}")
    try:
        # 1. 오디오 로드 (16kHz)
        y1, sr1 = librosa.load(user_path, sr=16000)
        y2, sr2 = librosa.load(target_path, sr=16000)

        # 2. 전처리: 무음 제거 (Trim)
        y1, _ = librosa.effects.trim(y1)
        y2, _ = librosa.effects.trim(y2)

        # 3. MFCC 특징 추출 (n_mfcc=13)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

        # 4. [핵심 기술 1] CMN (Cepstral Mean Normalization)
        # 성우와 사용자의 '음색(Tone)' 차이를 제거하고 '발음 패턴'만 남김
        mfcc1 -= (np.mean(mfcc1, axis=1, keepdims=True) + 1e-8)
        mfcc2 -= (np.mean(mfcc2, axis=1, keepdims=True) + 1e-8)

        # 5. [핵심 기술 2] DTW + Cosine Distance
        # 유클리드 거리 대신 코사인 거리를 사용하여 '패턴 유사도' 측정
        dist, path = fastdtw(mfcc1.T, mfcc2.T, dist=cosine, radius=10)

        # 6. 점수화 로직 (Calibrated Scoring)
        avg_dist = dist / len(path)
        print(f"🧮 패턴 거리(Cosine): {avg_dist:.4f}")

        # 임계값 설정 (Cosine 거리는 보통 0~2 사이, 0이 완전 일치)
        base_threshold = 0.6

        if avg_dist > base_threshold:
            final_score = 10
        else:
            # 선형 비례 점수화
            similarity = 1 - (avg_dist / base_threshold)
            final_score = int(similarity * 100)

        # 보너스 점수 (패턴이 일정 수준 이상 맞으면 가산점)
        if final_score > 60:
            final_score = min(100, final_score + 15)

        print(f"✅ 최종 산출 점수: {final_score}")
        return final_score

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return 0

# --- API Endpoints ---
@app.get("/themes")
def get_themes():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.cursor().execute("SELECT * FROM products").fetchall()
            themes = {}
            for row in rows:
                item = dict(row)
                try: item['examples'] = json.loads(item['examples'])
                except: item['examples'] = ["Hello"]
                if item['category'] == 'basic': item['icon'] = "📚"
                themes[item['id']] = item
            return themes
    except: return {}

@app.post("/login")
def login(req: AuthRequest):
    with sqlite3.connect(DB_NAME) as conn:
        user = conn.cursor().execute("SELECT username, role FROM users WHERE username=? AND password=?", (req.username, req.password)).fetchone()
    if user: return {"status": "success", "username": user[0], "role": user[1]}
    raise HTTPException(status_code=401, detail="Login failed")

@app.post("/register")
def register(req: RegisterRequest):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            if conn.cursor().execute("SELECT username FROM users WHERE username=?", (req.username,)).fetchone():
                raise HTTPException(status_code=400, detail="User exists")
            conn.cursor().execute("INSERT INTO users (username, password, role, full_name, phone, address) VALUES (?, ?, 'user', ?, ?, ?)", (req.username, req.password, req.full_name, req.phone, req.address))
            conn.commit()
        return {"status": "success"}
    except: raise HTTPException(status_code=500, detail="Error")

@app.post("/book")
def book(req: BookingRequest):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            row = conn.cursor().execute("SELECT title FROM products WHERE id=?", (req.theme_id,)).fetchone()
            title = row[0] if row else "Unknown"
            conn.cursor().execute("INSERT INTO bookings (username, theme_id, theme_title, start_date, end_date, people) VALUES (?,?,?,?,?,?)", (req.username, req.theme_id, title, req.start_date, req.end_date, req.people))
            conn.commit()
        return {"status": "success"}
    except: raise HTTPException(status_code=500, detail="Error")

@app.get("/bookings/my")
def my_bookings(username: str):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.cursor().execute("SELECT * FROM bookings WHERE username=? ORDER BY id DESC", (username,)).fetchall()]
    except: return []

@app.get("/bookings/all")
def all_bookings():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.cursor().execute("SELECT * FROM bookings ORDER BY id DESC").fetchall()]
    except: return []

@app.post("/bookings/cancel")
def cancel(req: CancelRequest):
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute("DELETE FROM bookings WHERE id=?", (req.booking_id,))
        conn.commit()
    return {"status": "success"}

# --- [핵심] Deep Tech AI Talk ---
@app.post("/talk")
async def talk_to_ai(file: UploadFile = File(...), theme_id: str = Form(...)):
    filename = file.filename
    print(f"📁 오디오 업로드: {filename}")

    user_audio_path = f"temp_audio/input_{filename}"
    target_audio_path = f"temp_audio/target_{filename}.mp3"

    try:
        with open(user_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Whisper STT (힌트 제공)
        print("🎤 STT 변환 중...")
        with open(user_audio_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="The user speaks English. Please transcribe accurately."
            )
        user_text = transcript.text
        print(f"🗣️ 인식된 텍스트: {user_text}")

        if len(user_text.strip()) < 1:
            return {"error": "목소리가 감지되지 않았습니다."}

        # 2. 페르소나 조회
        persona, situation = "Tutor", "Practice"
        try:
            with sqlite3.connect(DB_NAME) as conn:
                row = conn.cursor().execute("SELECT persona, situation FROM products WHERE id=?", (theme_id,)).fetchone()
                if row: persona, situation = row
        except: pass

        # 3. LLM 호출 (문법/설명/번역)
        SYSTEM_PROMPT = f"""
        Role: You are '{persona}' in '{situation}'.
        Task: User speaks English. Provide natural Korean translation.
        Output JSON Only:
        {{
            "korean": "Target Korean sentence",
            "romanized": "...",
            "english": "...",
            "grammar_point": "Key grammar rule",
            "explanation": "Context explanation"
        }}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User said: '{user_text}'. Return JSON."}
            ],
            response_format={ "type": "json_object" }
        )

        data = json.loads(response.choices[0].message.content)
        target_korean = data.get("korean", "다시 시도해주세요.")

        # 4. [Deep Tech] 비교용 원어민 오디오 생성
        tts_res = openai_client.audio.speech.create(model="tts-1", voice="nova", input=target_korean, speed=1.0)
        tts_res.stream_to_file(target_audio_path)

        # 5. [Deep Tech] 유사도 분석
        score = analyze_audio_similarity(user_audio_path, target_audio_path)
        data['tech_score'] = score

        # 6. 전체 오디오 생성 (문장 + 설명)
        full_text = f"{target_korean}... {data.get('explanation')}... 중요 문법은 {data.get('grammar_point')} 입니다."
        full_tts = openai_client.audio.speech.create(model="tts-1", voice="nova", input=full_text, speed=1.0)
        audio_b64 = base64.b64encode(full_tts.content).decode('utf-8')

        return {
            "user_text": user_text,
            "structured_data": data,
            "audio_base64": audio_b64
        }

    except Exception as e:
        print(f"🚨 Server Error: {e}")
        return {"error": str(e)}
    finally:
        for p in [user_audio_path, target_audio_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)