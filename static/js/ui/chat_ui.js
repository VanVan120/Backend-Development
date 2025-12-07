import { APIService } from '../services/api.js';

export class ChatUI {
    constructor(contextProvider) {
        this.contextProvider = contextProvider || (() => null);
        this.chatWindow = document.getElementById('chat-window');
        this.chatInput = document.getElementById('chat-input');
        this.messagesContainer = document.getElementById('chat-messages');
        this.sendBtn = document.getElementById('chat-send-btn');
        this.toggleBtn = document.getElementById('chat-toggle-btn');
        this.closeBtn = document.getElementById('chat-close-btn');
        this.minimizeBtn = document.getElementById('chat-minimize-btn');
        this.notificationDot = document.getElementById('chat-notification-dot');
        
        this.initEventListeners();
        this.loadHistory();
        this.checkUnreadStatus();
    }

    initEventListeners() {
        // Toggle
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggle());
        }

        // Close & Minimize
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.toggle());
        }
        if (this.minimizeBtn) {
            this.minimizeBtn.addEventListener('click', () => this.toggle());
        }

        // Send Message
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.handleSend());
        }
        if (this.chatInput) {
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleSend();
            });
        }

        // Quick Actions
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.chatInput.value = e.target.innerText;
                this.handleSend();
            });
        });
    }

    toggle() {
        this.chatWindow.classList.toggle('d-none');
        this.toggleBtn.classList.toggle('d-none');
        
        if (!this.chatWindow.classList.contains('d-none')) {
            this.chatInput.focus();
            // Scroll to bottom when opened
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            // Hide notification dot when opened
            if (this.notificationDot) this.notificationDot.classList.add('d-none');
            sessionStorage.setItem('chat_has_unread', 'false');
        }
    }

    async handleSend() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        const context = this.contextProvider();

        this.addMessage('user', message);
        this.chatInput.value = '';
        this.chatInput.disabled = true;
        if (this.sendBtn) this.sendBtn.disabled = true;

        // Typing indicator
        const typingId = this.addTypingIndicator();

        try {
            const data = await APIService.sendChatMessage(message, context);
            
            this.removeMessage(typingId);

            if (data.error) {
                this.addMessage('bot', 'Sorry, I encountered an error: ' + data.error);
            } else {
                this.addMessage('bot', data.reply);
            }

        } catch (error) {
            console.error("Chat Error:", error);
            this.removeMessage(typingId);
            this.addMessage('bot', 'Sorry, I am having trouble connecting right now.');
        } finally {
            this.chatInput.disabled = false;
            if (this.sendBtn) this.sendBtn.disabled = false;
            this.chatInput.focus();
        }
    }

    loadHistory() {
        const history = JSON.parse(sessionStorage.getItem('chat_history') || '[]');
        history.forEach(msg => {
            this.addMessage(msg.sender, msg.text, false);
        });
    }

    checkUnreadStatus() {
        const hasUnread = sessionStorage.getItem('chat_has_unread') === 'true';
        if (hasUnread && this.notificationDot) {
            this.notificationDot.classList.remove('d-none');
        }
    }

    addMessage(sender, text, save = true) {
        // Save to session storage
        if (save) {
            const history = JSON.parse(sessionStorage.getItem('chat_history') || '[]');
            history.push({ sender, text, timestamp: Date.now() });
            sessionStorage.setItem('chat_history', JSON.stringify(history));

            // Show notification if bot sends message and window is closed
            if (sender === 'bot' && this.chatWindow.classList.contains('d-none')) {
                if (this.notificationDot) this.notificationDot.classList.remove('d-none');
                sessionStorage.setItem('chat_has_unread', 'true');
            }
        }

        const msgDiv = document.createElement('div');
        const msgId = 'msg-' + Date.now();
        msgDiv.id = msgId;
        
        if (sender === 'user') {
            msgDiv.className = 'd-flex justify-content-end chat-message-enter';
            msgDiv.innerHTML = `
                <div class="user-message-bubble bg-primary text-white p-3 rounded-4 shadow-sm small" style="max-width: 80%; border-top-right-radius: 4px !important; background: linear-gradient(135deg, #3b82f6, #2563eb);">
                    ${text}
                </div>
            `;
        } else {
            // Parse Markdown for bot messages
            const parsedText = typeof marked !== 'undefined' ? marked.parse(text) : text;
            
            msgDiv.className = 'd-flex gap-3 chat-message-enter';
            msgDiv.innerHTML = `
                <div class="rounded-circle bg-white shadow-sm d-flex align-items-center justify-content-center flex-shrink-0 border border-light-subtle" style="width: 36px; height: 36px;">
                    <i class="fas fa-robot text-primary small"></i>
                </div>
                <div class="d-flex flex-column gap-1" style="max-width: 80%;">
                    <div class="bot-message-bubble bg-white p-3 rounded-4 shadow-sm border border-light-subtle text-dark" 
                         style="border-top-left-radius: 4px !important; border-left: 3px solid #3b82f6;">
                        ${parsedText}
                    </div>
                    <span class="text-muted ms-1 fw-medium" style="font-size: 10px; letter-spacing: 0.5px;">AI ASSISTANT • JUST NOW</span>
                </div>
            `;
        }
        
        this.messagesContainer.appendChild(msgDiv);
        this.scrollToBottom();
        return msgId;
    }

    addTypingIndicator() {
        const msgDiv = document.createElement('div');
        const msgId = 'typing-' + Date.now();
        msgDiv.id = msgId;
        msgDiv.className = 'd-flex gap-3 chat-message-enter';
        msgDiv.innerHTML = `
            <div class="rounded-circle bg-white shadow-sm d-flex align-items-center justify-content-center flex-shrink-0 border border-light" style="width: 32px; height: 32px;">
                <i class="fas fa-robot text-primary small"></i>
            </div>
            <div class="d-flex flex-column gap-1">
                <div class="bot-message-bubble d-flex align-items-center gap-1 bg-white p-3 rounded-4 shadow-sm border border-light" style="height: 42px; border-top-left-radius: 4px !important;">
                    <div class="typing-dot bg-secondary rounded-circle" style="width: 6px; height: 6px;"></div>
                    <div class="typing-dot bg-secondary rounded-circle" style="width: 6px; height: 6px;"></div>
                    <div class="typing-dot bg-secondary rounded-circle" style="width: 6px; height: 6px;"></div>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(msgDiv);
        this.scrollToBottom();
        return msgId;
    }

    removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
}
