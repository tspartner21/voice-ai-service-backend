import os
import shutil
import json
import sqlite3
import re
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ 경고: .env 파일에 OPENAI_API_KEY가 없습니다.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 저장소
os.makedirs("static/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 💾 DB 초기화 ---
DB_NAME = "bookings.db"

def init_db():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()

            # 사용자 테이블 (확장됨)
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS users (
                                                                username TEXT PRIMARY KEY,
                                                                password TEXT NOT NULL,
                                                                role TEXT NOT NULL DEFAULT 'user',
                                                                full_name TEXT,
                                                                phone TEXT,
                                                                address TEXT
                           )
                           ''')

            # 상품 테이블
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS products (
                                                                   id TEXT PRIMARY KEY, category TEXT, title TEXT, price TEXT, rating TEXT,
                                                                   image_url TEXT, desc TEXT, persona TEXT, situation TEXT, mission TEXT, examples TEXT)''')

            # 예약 테이블
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS bookings (
                                                                   id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, theme_id TEXT, theme_title TEXT,
                                                                   start_date TEXT, end_date TEXT, people INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

            # 기본 계정
            cursor.execute("INSERT OR IGNORE INTO users (username, password, role, full_name) VALUES ('admin', 'admin', 'admin', 'Admin')")
            cursor.execute("INSERT OR IGNORE INTO users (username, password, role, full_name) VALUES ('user', 'user', 'user', 'Tester')")

            # [데이터 복구] 12개 기초 회화 + 3개 오프라인 퀘스트
            seed_data = [
                # Basic Training (12개)
                ("kpop", "basic", "🎤 K-POP 콘서트", "Free", "5.0", "", "콘서트장 상황극", "열정적인 MC", "콘서트장", "응원하기", '["Scream!", "Encore!"]'),
                ("store", "basic", "🏪 편의점 알바", "Free", "5.0", "", "편의점 상황극", "친절한 알바생", "편의점", "계산하기", '["How much?", "I need a bag."]'),
                ("date", "basic", "💕 홍대 첫 데이트", "Free", "5.0", "", "데이트 상황극", "설레는 상대방", "홍대 맛집", "주문하기", '["You look pretty.", "Lets eat."]'),
                ("taxi", "basic", "🚕 택시 타기", "Free", "5.0", "", "택시 상황극", "베테랑 기사님", "택시 안", "목적지 말하기", '["Go to Gangnam.", "Stop here."]'),
                ("club", "basic", "💃 클럽 입장", "Free", "5.0", "", "클럽 입구 상황극", "엄격한 가드", "클럽 입구", "신분증 제시", '["Here is my ID.", "Entrance fee?"]'),
                ("drama", "basic", "🎬 드라마 촬영장", "Free", "5.0", "", "촬영장 구경", "촬영 스태프", "촬영 현장", "양해 구하기", '["Can I watch?", "Who is he?"]'),
                ("bar", "basic", "🍸 이태원 바", "Free", "5.0", "", "바 주문", "센스있는 바텐더", "Bar", "칵테일 주문", '["One beer please.", "Recommendation?"]'),
                ("cafe", "basic", "☕ 카페 주문", "Free", "5.0", "", "카페 주문", "상냥한 바리스타", "카페", "커피 주문", '["Iced Americano.", "To go please."]'),
                ("hospital", "basic", "🏥 약국/병원", "Free", "5.0", "", "아픈 증상 설명", "의사", "병원", "증상 말하기", '["I have a headache.", "Medicine please."]'),
                ("subway", "basic", "🚇 지하철역", "Free", "5.0", "", "길 묻기", "역무원", "지하철", "환승 묻기", '["Where is Line 2?", "Is this Gangnam?"]'),
                ("school_class", "basic", "🏫 초등 교실", "Free", "5.0", "", "선생님과 대화", "담임 선생님", "교실", "숙제 제출", '["Here is homework.", "I am sorry."]'),
                ("school_sports", "basic", "🏃 학교 운동회", "Free", "5.0", "", "친구 응원", "단짝 친구", "운동장", "응원하기", '["Run faster!", "Fighting!"]'),

                # Offline Quest (3개)
                ("offline_hongdae", "offline", "🔥 홍대 언어교환 & 야시장", "35,000원", "4.9", "https://images.unsplash.com/photo-1538485399081-7191377e8241?w=800", "현지인 친구 사귀기", "모임장", "언어교환", "자기소개", '["Hello", "My hobby is cooking"]'),
                ("offline_kpop", "offline", "💃 K-POP 댄스 & 이태원 펍", "55,000원", "4.8", "https://images.unsplash.com/photo-1545128485-c400e7702796?w=800", "BTS 안무 배우기", "댄스강사", "댄스레슨", "동작 배우기", '["One more time!", "Cheers!"]'),
                ("offline_drama", "offline", "🍖 4박5일 K-Drama 패키지", "450,000원", "5.0", "https://images.unsplash.com/photo-1596280806440-424a5eb23b12?w=800", "드라마 촬영지 투어", "가이드", "촬영장", "사진찍기", '["Can I take a photo?", "I love this drama"]')
            ]

            for p in seed_data:
                cursor.execute("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?)", p)

            conn.commit()
        print("✅ DB Initialized & Themes Restored")
    except Exception as e:
        print(f"❌ DB Init Error: {e}")

init_db()

# --- Helper: JSON Clean ---
def clean_json(text):
    try:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(text)
    except:
        # Fallback JSON
        return {
            "korean_sentence": "다시 말씀해 주세요.",
            "romanized": "Dasi mal-hae-juseyo",
            "eng_meaning": "Please say it again.",
            "kor_explanation": "잘 못 들었습니다.",
            "eng_explanation": "I couldn't hear you well.",
            "feedback": ""
        }

# --- Models ---
class AuthRequest(BaseModel):
    username: str; password: str
class RegisterRequest(BaseModel):
    username: str; password: str; full_name: str; phone: str; address: str
class BookingRequest(BaseModel):
    username: str; theme_id: str; start_date: str; end_date: str; people: int
class CancelRequest(BaseModel):
    booking_id: int

# --- API ---
@app.get("/themes")
def get_themes():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.cursor().execute("SELECT * FROM products").fetchall()
            themes = {}
            icon_map = {"kpop":"🎤", "store":"🏪", "date":"💕", "taxi":"🚕", "club":"💃", "drama":"🎬", "bar":"🍸", "cafe":"☕", "hospital":"🏥", "subway":"🚇", "school_class":"🏫", "school_sports":"🏃"}
            for row in rows:
                item = dict(row)
                try: item['examples'] = json.loads(item['examples'])
                except: item['examples'] = ["Hello"]
                if item['category'] == 'basic': item['icon'] = icon_map.get(item['id'], "📚")
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
            conn.cursor().execute("INSERT INTO users (username, password, role, full_name, phone, address) VALUES (?, ?, 'user', ?, ?, ?)",
                                  (req.username, req.password, req.full_name, req.phone, req.address))
            conn.commit()
        return {"status": "success"}
    except HTTPException as e: raise e
    except: raise HTTPException(status_code=500, detail="Error")

@app.post("/admin/products")
async def add_product(id: str=Form(...), title: str=Form(...), price: str=Form(...), desc: str=Form(...), file: UploadFile=File(None)):
    try:
        url = "https://via.placeholder.com/400"
        if file:
            path = f"static/images/{file.filename}"
            with open(path, "wb") as b: shutil.copyfileobj(file.file, b)
            url = f"http://localhost:8000/{path}"
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                  (id, 'offline', title, price, 'New', url, desc, 'Guide', 'Tour', 'Enjoy', json.dumps(["Hello"])))
            conn.commit()
        return {"status": "success"}
    except: raise HTTPException(status_code=500, detail="Error")

@app.post("/book")
def book(req: BookingRequest):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            row = conn.cursor().execute("SELECT title FROM products WHERE id=?", (req.theme_id,)).fetchone()
            title = row[0] if row else "Unknown"
            conn.cursor().execute("INSERT INTO bookings (username, theme_id, theme_title, start_date, end_date, people) VALUES (?,?,?,?,?,?)",
                                  (req.username, req.theme_id, title, req.start_date, req.end_date, req.people))
            conn.commit()
        return {"status": "success", "message": "Booked!"}
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

# --- [핵심 수정] AI Talk (오디오 포맷 및 로직 강화) ---
@app.post("/talk")
async def talk_to_ai(file: UploadFile = File(...), theme_id: str = Form(...)):
    # 1. 파일 저장 (확장자 유지)
    filename = file.filename
    temp_filename = f"temp_{filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Whisper STT
        with open(temp_filename, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )

        user_text = transcript.text
        if len(user_text.strip()) < 1:
            return {"error": "No voice detected"}

        # 3. DB에서 페르소나 조회
        persona, situation = "Tutor", "Practice"
        try:
            with sqlite3.connect(DB_NAME) as conn:
                row = conn.cursor().execute("SELECT persona, situation FROM products WHERE id=?", (theme_id,)).fetchone()
                if row: persona, situation = row
        except: pass

        # 4. LLM 호출
        SYSTEM_PROMPT = f"""
        Role: You are '{persona}' in '{situation}'.
        Task: User speaks English. 
        1. Translate to Korean. 
        2. Romanize it. 
        3. Explain in Korean & English.
        4. Return JSON ONLY.
        {{
            "korean_sentence": "...",
            "romanized": "...",
            "eng_meaning": "...",
            "kor_explanation": "...",
            "eng_explanation": "...",
            "feedback": "..."
        }}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"User said: '{user_text}'"}],
            response_format={ "type": "json_object" }
        )

        data = clean_json(response.choices[0].message.content)

        # 5. TTS 생성 (문장 -> 설명 -> 5회 반복)
        k_sent = data.get('korean_sentence', '')
        k_expl = data.get('kor_explanation', '')
        tts_text = f"{k_sent}. {k_expl}. 자, 5번 반복합니다. " + ", ".join([k_sent] * 5)

        tts_res = openai_client.audio.speech.create(model="tts-1", voice="nova", input=tts_text)
        audio_b64 = base64.b64encode(tts_res.content).decode('utf-8')

        return {
            "user_text": user_text,
            "phonetic": data.get('romanized',''),
            "korean_text": k_sent,
            "eng_meaning": data.get('eng_meaning',''),
            "kor_explanation": k_expl,
            "eng_explanation": data.get('eng_explanation',''),
            "feedback": data.get('feedback',''),
            "audio_base64": audio_b64
        }

    except Exception as e:
        print(f"Talk Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)