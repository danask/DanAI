async function sendMessage() {
    const input = document.getElementById('promptInput');
    const taskType = document.getElementById('taskType').value;
    const sendBtn = document.getElementById('sendBtn');
    const prompt = input.value.trim();

    if (!prompt) return;

    appendMessage(prompt, 'user-msg');
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    const loadingId = appendMessage('생성 중...', 'agent-msg');

    try {
        const res = await fetch('/agent/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, task_type: taskType })
        });
        const data = await res.json();
        
        document.getElementById(loadingId).innerText = data.result || '응답을 받지 못했습니다.';
    } catch (err) {
        document.getElementById(loadingId).innerText = '오류가 발생했습니다: ' + err.message;
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

async function reviewMistakes() {
    const reviewBtn = document.getElementById('reviewBtn');
    reviewBtn.disabled = true;

    appendMessage('복습할 문장을 골라줘', 'user-msg');
    const loadingId = appendMessage('복습 문제 준비 중...', 'agent-msg');

    try {
        const res = await fetch('/review?count=3');
        const data = await res.json();

        if (!data.quiz) {
            document.getElementById(loadingId).innerText = data.message || '복습할 오답이 아직 없습니다.';
            return;
        }

        document.getElementById(loadingId).innerText = data.quiz;
    } catch (err) {
        document.getElementById(loadingId).innerText = '오류가 발생했습니다: ' + err.message;
    } finally {
        reviewBtn.disabled = false;
    }
}

async function fetchCanadaNewsSummary() {
    const caNewsBtn = document.getElementById('caNewsBtn');
    caNewsBtn.disabled = true;

    // 1단계: 뉴스 수집 상태 표시
    const loadingId = appendMessage('📡 CityNews, CBC, CTV RSS에서 최신 기사를 수집하는 중...', 'agent-msg');
    const msgElement = document.getElementById(loadingId);

    try {
        // 2단계: Ollama 요약 진행 상태 변경 (1.5초 후 텍스트 업데이트)
        setTimeout(() => {
            if (msgElement && caNewsBtn.disabled) {
                msgElement.innerText = '🤖 기사 수집 완료. Ollama가 한국어로 번역 및 Top 10 요약문을 생성하고 있습니다... (약 20~30초 소요)';
            }
        }, 2000);

        const res = await fetch('/news/canada-summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || '서버 응답 오류');
        }

        const data = await res.json();
        msgElement.innerText = data.result || '요약 결과를 가져올 수 없습니다.';

    } catch (err) {
        msgElement.innerText = '❌ 오류 발생: ' + err.message;
    } finally {
        caNewsBtn.disabled = false;
    }
}

// 1. 저장된 최신 뉴스 즉시 불러오기 (DB 조회 - 속도 매우 빠름)
async function fetchDbNewsSummary() {
    const dbNewsBtn = document.getElementById('dbNewsBtn');
    dbNewsBtn.disabled = true;

    const loadingId = appendMessage('📡 DB에서 최신 뉴스 요약을 불러옵니다...', 'agent-msg');
    const msgElement = document.getElementById(loadingId);

    try {
        const res = await fetch('/news/latest');
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || '저장된 뉴스가 없습니다.');
        }

        const data = await res.json();
        const dateStr = data.created_at ? new Date(data.created_at).toLocaleString() : '';
        
        // UTC 시간 변환 함수 호출
        const formattedDate = formatToLocalTime(data.created_at);
        const formattedMarkdown = `**[수집 일시: ${formattedDate}]**\n\n${data.summary || '요약 내용이 없습니다.'}`;
        msgElement.innerHTML = marked.parse(formattedMarkdown);

    } catch (err) {
        msgElement.innerText = '❌ 오류 발생: ' + err.message;
    } finally {
        dbNewsBtn.disabled = false;
    }
}

// 2. 실시간 뉴스 수집 및 AI 요약 실행 (Ollama 호출 - 약 20~30초 소요)
async function fetchLiveNewsSummary() {
    const liveNewsBtn = document.getElementById('liveNewsBtn');
    liveNewsBtn.disabled = true;

    const loadingId = appendMessage('🚀 실시간 RSS 수집 및 Ollama 요약 진행 중...', 'agent-msg');
    const msgElement = document.getElementById(loadingId);

    try {
        const res = await fetch('/news/canada-summary', { method: 'POST' });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || '요약 생성 실패');
        }

        const data = await res.json();
        
        // innerHTML과 marked.parse 사용
        msgElement.innerHTML = marked.parse(data.summary || data.result || '요약 결과가 없습니다.');

    } catch (err) {
        msgElement.innerText = '❌ 오류 발생: ' + err.message;
    } finally {
        liveNewsBtn.disabled = false;
    }
}

function formatToLocalTime(utcDateString) {
    if (!utcDateString) return '';
    
    // DB의 UTC 문자열 뒤에 'Z'가 없는 경우 붙여서 ISO 표준(UTC)임을 명시
    const isoString = utcDateString.endsWith('Z') ? utcDateString : utcDateString + 'Z';
    const date = new Date(isoString);

    // timeZone 옵션을 제외하면 접속한 기기(OS)의 타임존이 자동 적용됨
    return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });
}

function appendMessage(text, className) {
    const chatBox = document.getElementById('chatBox');
    const msgDiv = document.createElement('div');
    const msgId = 'msg-' + Date.now();
    msgDiv.id = msgId;
    msgDiv.className = 'message ' + className;
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgId;
}