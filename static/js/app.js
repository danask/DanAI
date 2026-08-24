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