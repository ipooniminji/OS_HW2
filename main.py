from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from typing import List
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SpamGuard")

# FastAPI 앱 초기화
app = FastAPI(
    title="안심 문자 분류기 API (MLOps 고도화 버전)",
    description="스미싱 진단 및 Explainable AI(xAI) 스팸 키워드 근거 추출 엔진 추가",
    version="5.0.0"
)

# 예측 요청 데이터 모델
class PredictRequest(BaseModel):
    text: str

# 예측 응답 데이터 모델 (keywords 추가)
class PredictResponse(BaseModel):
    label: str
    probability: dict
    message: str
    confidence_level: str
    keywords: List[str] = []

# 데이터셋 보강 (사칭, 미기입, 반송 등 스미싱 패턴 집중 추가)
TRAINING_DATA = [
    # --- 스팸(Spam): 광고, 피싱, 도박, 택배 사칭 강도 높임 ---
    ("[Web발신] 내부자 정보 입수! 내일 상한가 확정 종목 무료 공개. 리딩방 초대: http://bit.ly/xxx", "spam"),
    ("[광고] (주)대박투자증권 - 공시 발표 전 선취매 찬스! 최소 300% 수익 보장", "spam"),
    ("VIP 전용 급등주 추천. 지금 바로 입장하세요. 오늘만 무료", "spam"),
    ("해외선물 리딩방 신규 회원 모집 중. 매일 50만원 수익 달성", "spam"),
    ("코인 떡상 예감! 단톡방에서 무료로 픽 드립니다. (광고)", "spam"),
    
    ("[Web발신] 귀하는 서민 정부지원금 대상자입니다. 한도 5천만원, 저금리 대환대출. 신청 마감 임박!", "spam"),
    ("[광고] 무직자 당일 대출 가능. 낮은 이자로 즉시 입금해드립니다.", "spam"),
    ("신용불량자도 무조건 승인되는 최저금리 대출 특별 안내", "spam"),
    ("고객님, 특별 금리 인하 요건에 충족되어 안내드립니다. 대출 한도 즉시 조회하기: url", "spam"),
    ("정부지원 생계자금 신청 기간 만료 전입니다. 지금 바로 전화주세요", "spam"),
    
    ("[Web발신] 로또 1등 당첨 번호 무료 제공! 이번 주 예상 번호 확인하기", "spam"),
    ("신규 가입 시 5만 포인트 즉시 지급! 첫 가입 롤링 100% 환전 가능 무제한", "spam"),
    ("바카라/슬롯 안전 공원 보장. 환전 지연 절대 없음. 접속 주소: www.bet24.com", "spam"),
    ("로또 1등 조합기 무료 다운로드. 100% 당첨 보장 및 즉시 환전", "spam"),
    ("스포츠 토토 회원가입 이벤트! 첫 충전시 30% 추가 포인트 당첨 찬스", "spam"),
    
    # 🚨 **중요 스미싱 패턴** 🚨
    ("우체국택배 배송 기사입니다. 상세주소 미기입으로 반송된 택배가 있어 주소 회신 바랍니다.", "spam"),
    ("CJ대한통운: 배송 불가 (주소 불일치). 카톡으로 주소지 다시 남겨주세요.", "spam"),
    ("택배 배송정보가 잘못되었습니다. 즉시 카톡으로 올바른 주소 회신바랍니다.", "spam"),
    ("고객님 우체국택배입니다. 수취인 부재로 수하물 반송 대기중. 앱에서 조회하세요.", "spam"),
    ("한진택배: 도로명 주소 오류로 반송 처리될 예정입니다. 주소수정 요망", "spam"),
    ("[웹발신] 서울중앙지방검찰청입니다. 귀하의 명의가 대포통장 범죄에 연루되었습니다. 즉시 확인 요망", "spam"),
    ("[결제확인] 해외결제 승인 완료 859,000원. 본인 아닐 시 즉시 직통 소비자원 1588-1234로 신고바람", "spam"),
    ("국민건강보험공단 알림: 특정 건강검진 결과 통보서가 발송되었습니다. 모바일 전자문서 확인 요망", "spam"),
    ("통신사 미납 요금 납부 안내. 고객님의 발신이 곧 정지될 예정입니다. 간편 결제하기", "spam"),
    ("[알림] 결제하신 상품 불량으로 인해 환불처리 대기중입니다. 연락처로 회신 바랍니다.", "spam"),

    # --- 일상(Ham): 무해한 평범한 택배 및 일상 대화 ---
    ("배송 기사입니다. 문 앞에 택배 두고 갑니다. 좋은 하루 되세요", "ham"),
    ("안녕하세요 택배기사입니다. 소화전에 물건 넣고 갑니다", "ham"),
    ("고객님, 주문하신 상품이 오늘 오후 2시에서 4시 사이에 배송될 예정입니다.", "ham"),
    ("배달의민족 주문이 접수되었습니다! 40분 내외로 도착 예정입니다.", "ham"),
    ("엄마 나 지금 집 앞인데, 짐 좀 들어줘 무거워", "ham"),
    ("어디쯤이야? 나 먼저 카페에 들어가 있을게", "ham"),
    ("오늘 저녁 메뉴 뭐 할까? 오랜만에 고기 구워 먹을까?", "ham"),
    ("아빠 생신 선물 샀어? 주말에 같이 보러 가자", "ham"),
    ("내일 아침 9시까지 역 앞에서 보자 늦지 말고", "ham"),
    ("이번주에 캠핑 갈래? 날씨 엄청 좋대", "ham"),
    ("강아지 산책 시키고 올게. 밥 먼저 먹고 있어", "ham"),
    ("선생님, 지난번 말씀해주신 자료 파일 다시 한 번만 보내주실 수 있나요?", "ham"),
    ("팀장님 저 오늘 몸이 너무 안 좋아서 1시간 일찍 퇴근해도 될까요ㅠㅠ", "ham"),
    ("저녁에 운동 끝나고 전화할게", "ham"),
    ("야 너 어제 왜 안나왔어? 다들 기다렸잖아", "ham"),
    ("혹시 집에 참기름 있어? 없으면 사갈게", "ham"),
    ("팀장님 안녕하십니까, 내일 오전 주간 회의 자료 초안 메일로 송부드렸습니다.", "ham"),
    ("어제 말씀드린 회의록 정리해서 사내 망에 공유해 주시겠어요?", "ham"),
    ("네트워크 정기 점검으로 인해 10분간 사내 VPN 접속이 일시 끊어질 수 있습니다.", "ham")
]

# 위험 주의 키워드 사전 (Explainability 용도)
DANGEROUS_KEYWORDS = [
    "대출", "정부지원금", "마감 임박", "[광고]", "광고", "(주)", 
    "상한가", "리딩방", "선취매", "수익 보장", "급등주",
    "포인트", "환전", "로또", "1등", "조합기", "바카라", "슬롯",
    "미기입", "반송", "주소 회신", "주소 불일치", "어플 설치", "apk",
    "대포통장", "검찰청", "소비자원", "해외결제", "미납", "환불처리"
]

def extract_suspicious_keywords(text: str) -> List[str]:
    """텍스트 내 포함된 위험 키워드를 추출합니다."""
    found = []
    # 단순 문자열 비교. 조금 더 고도화하려면 regex 활용 가능.
    for kw in DANGEROUS_KEYWORDS:
        if kw in text:
            found.append(kw)
    return found

# TfidfVectorizer 설정
advanced_vectorizer = TfidfVectorizer(
    token_pattern=r"[^\s]+", 
    ngram_range=(1, 3)
)

model_pipeline = make_pipeline(advanced_vectorizer, MultinomialNB())

@app.on_event("startup")
async def startup_event():
    logger.info("분류기 AI 모델 및 키워드 추출기 학습 시작...")
    X_train = [text for text, label in TRAINING_DATA]
    y_train = [label for text, label in TRAINING_DATA]
    model_pipeline.fit(X_train, y_train)
    logger.info("모델 학습 완료. 운영 서비스 준비 완료.")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️안심 문자 분류기🛡️</title>
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Pretendard', -apple-system, sans-serif;
        }

        body {
            background-color: #f3f4f6;
            color: #1f2937;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 24px;
        }

        .container {
            background: #ffffff;
            border-radius: 12px;
            padding: 40px;
            width: 100%;
            max-width: 580px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        h1 {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #111827;
            text-align: center;
            letter-spacing: -0.5px;
        }

        p.subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 32px;
            font-size: 15px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: #f9fafb;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 16px;
            color: #1f2937;
            font-size: 16px;
            resize: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            outline: none;
            line-height: 1.5;
        }

        textarea:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            background: #ffffff;
        }

        button {
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 16px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            justify-content: center;
            align-items: center;
            letter-spacing: 0.5px;
        }

        button:hover {
            background-color: #1d4ed8;
        }

        button:disabled {
            background-color: #9ca3af;
            cursor: not-allowed;
        }

        #result-card {
            margin-top: 24px;
            padding: 24px;
            border-radius: 8px;
            display: none;
            text-align: center;
            border: 1px solid transparent;
        }
        
        #result-card.show {
            display: block;
            animation: fadeIn 0.4s ease;
        }

        .result-spam {
            background: #fef2f2;
            border-color: #fecaca;
            color: #991b1b;
        }

        .result-ham {
            background: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }
        
        .result-uncertain {
            background: #fefce8;
            border-color: #fef08a;
            color: #854d0e;
        }
        
        .result-icon {
            font-size: 36px;
            margin-bottom: 8px;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .result-desc {
            font-size: 14px;
            margin-bottom: 16px;
            opacity: 0.9;
        }

        /* 위험 키워드 시각화 CSS */
        .result-keywords {
            display: none;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
            padding: 12px;
            background-color: rgba(255, 255, 255, 0.4);
            border-radius: 8px;
        }

        .kw-label {
            font-size: 13px;
            font-weight: 600;
            color: #4b5563;
        }

        .keyword-badge {
            background-color: #fee2e2;
            color: #dc2626;
            border: 1px solid #fca5a5;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: -0.3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        .probability {
            font-size: 13px;
            padding: 4px 12px;
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 6px;
            display: inline-block;
            font-weight: 600;
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️안심 문자 분류기🛡️</h1>
        <p class="subtitle">스미싱 및 스팸 문자를 로컬 환경에서 감지합니다.</p>
        
        <div class="input-group">
            <textarea id="message-input" placeholder="스팸일지 모르는 문장, 안전하게 텍스트를 기입해서 먼저 검사해보세요."></textarea>
            <button id="inspect-btn" onclick="analyzeText()">
                <span id="btn-text">안심 텍스트 검증</span>
                <div id="btn-spinner" class="spinner"></div>
            </button>
        </div>

        <div id="result-card">
            <div id="result-icon" class="result-icon"></div>
            <div id="result-title" class="result-title"></div>
            <div id="result-desc" class="result-desc"></div>
            <!-- 키워드 박스 추가 -->
            <div id="result-keywords" class="result-keywords"></div>
            <div id="result-prob" class="probability"></div>
        </div>
    </div>

    <script>
        async function analyzeText() {
            const inputEl = document.getElementById('message-input');
            const btnEl = document.getElementById('inspect-btn');
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');
            const resultCard = document.getElementById('result-card');
            const kwCard = document.getElementById('result-keywords');
            
            const text = inputEl.value.trim();
            if (!text) return;

            btnEl.disabled = true;
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            resultCard.classList.remove('show');
            kwCard.innerHTML = ''; // 키워드 초기화
            kwCard.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResult(data);
                } else {
                    alert('에러 발생: ' + data.detail);
                }
            } catch (err) {
                alert('서버와 연결을 실패했습니다.');
            } finally {
                btnEl.disabled = false;
                btnText.style.display = 'block';
                btnSpinner.style.display = 'none';
            }
        }

        function showResult(data) {
            const resultCard = document.getElementById('result-card');
            const iconEl = document.getElementById('result-icon');
            const titleEl = document.getElementById('result-title');
            const descEl = document.getElementById('result-desc');
            const probEl = document.getElementById('result-prob');
            const kwCard = document.getElementById('result-keywords');
            
            resultCard.className = '';
            const label = data.label;
            
            // 위험 키워드가 전송되어 왔다면 그려주기 (보류 또는 스팸 상태)
            if (data.keywords && data.keywords.length > 0) {
                const titleSpan = document.createElement('span');
                titleSpan.className = 'kw-label';
                titleSpan.innerText = '검출된 위험 키워드 :';
                kwCard.appendChild(titleSpan);
                
                data.keywords.forEach(kw => {
                    const badge = document.createElement('span');
                    badge.className = 'keyword-badge';
                    badge.innerText = kw;
                    kwCard.appendChild(badge);
                });
                kwCard.style.display = 'flex';
            }
            
            if (label === 'uncertain') {
                resultCard.classList.add('result-uncertain');
                iconEl.innerHTML = '⚠️';
                titleEl.innerText = '주의: 판단 유보';
                descEl.innerText = data.message;
                const maxProbLabel = data.probability.spam > data.probability.ham ? 'Spam' : 'Ham';
                const maxProbValue = (Math.max(data.probability.spam, data.probability.ham) * 100).toFixed(1);
                probEl.innerText = `현재 확률: ${maxProbValue}% (${maxProbLabel})`;
                
            } else {
                const probability = (data.probability[label] * 100).toFixed(1);
                
                if (label === 'spam') {
                    resultCard.classList.add('result-spam');
                    iconEl.innerHTML = '🛑';
                    titleEl.innerText = '위험: 스팸/스미싱 (Spam)';
                    descEl.innerText = data.message;
                    probEl.innerText = `검출 신뢰도: ${probability}%`;
                } else {
                    resultCard.classList.add('result-ham');
                    iconEl.innerHTML = '✅';
                    titleEl.innerText = '정상 메시지 (Ham)';
                    descEl.innerText = data.message;
                    probEl.innerText = `안심 신뢰도: ${probability}%`;
                }
            }
            
            resultCard.classList.add('show');
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_TEMPLATE

@app.post("/predict", response_model=PredictResponse)
def predict_spam(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
    
    prediction_array = model_pipeline.predict([text])
    probabilities_array = model_pipeline.predict_proba([text])
    
    prediction = prediction_array[0]
    probabilities = probabilities_array[0]
    
    classes = model_pipeline.classes_
    prob_dict = {
        classes[0]: float(probabilities[0]),
        classes[1]: float(probabilities[1])
    }
    
    max_prob = max(prob_dict.values())
    found_keywords = []

    if prediction == "spam" and max_prob < 0.60:
        final_label = "uncertain"
        confidence_level = "보류"
        message = "일부 스팸 패턴이 보이나 확정하기 어렵습니다. 발신자에 주의하세요."
    elif prediction == "ham" and max_prob < 0.55:
        final_label = "uncertain"
        confidence_level = "보류"
        message = "판단을 내리기 애매한 패턴입니다."
    else:
        final_label = prediction
        confidence_level = "확정"
        if prediction == "spam":
            message = "스미싱 또는 악성 스팸 패턴이 명확합니다. 클릭 및 회신 금지."
        else:
            message = "위험 요소가 발견되지 않았습니다."
            
    # xAI (Explainability) 로직
    # 만약 스팸으로 의심되거나 확정된 경우 무슨 키워드 때문인지 추출하여 반환 및 로깅
    if final_label in ["spam", "uncertain"]:
        found_keywords = extract_suspicious_keywords(text)
        short_text = text.replace('\n', ' ')[:30] + '...'
        if found_keywords:
            logger.warning(f"[Spam Guard ALERT] 분류: {final_label} | 확률: {max_prob*100:.1f}% | 적발된 키워드: {found_keywords} | 원문: {short_text}")
        else:
            logger.info(f"[Spam Guard INFO] 분류: {final_label} | 확률: {max_prob*100:.1f}% | 위험 키워드는 없으나 AI 패턴으로 의심됨 | 원문: {short_text}")
    else:
        logger.info(f"[Spam Guard SAFE] 정상 분류 (Ham) | 원문: {text.replace('\n', ' ')[:30]}...")

    return PredictResponse(
        label=final_label,
        probability=prob_dict,
        message=message,
        confidence_level=confidence_level,
        keywords=found_keywords
    )
