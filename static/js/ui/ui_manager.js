export class UIManager {
    static showHome() {
        document.getElementById('home-view').classList.remove('hidden-section');
        document.getElementById('triage-view').classList.add('hidden-section');
        document.getElementById('model-a-view').classList.add('hidden-section');
        document.getElementById('model-b-view').classList.add('hidden-section');
    }

    static showAnalysis() {
        document.getElementById('home-view').classList.add('hidden-section');
        document.getElementById('triage-view').classList.remove('hidden-section');
        document.getElementById('model-a-view').classList.add('hidden-section');
        document.getElementById('model-b-view').classList.add('hidden-section');
        
        // Ensure we are in the "Upload" state, not "Loading" state
        this.hideLoading();
        document.getElementById('main-file-input').value = "";
        
        document.getElementById('triage-view').scrollIntoView({ behavior: 'smooth' });
    }

    static showLoading() {
        document.getElementById('triage-loading').classList.remove('hidden-section');
        document.querySelector('.triage-upload-area').classList.add('hidden-section');
    }

    static hideLoading() {
        document.getElementById('triage-loading').classList.add('hidden-section');
        document.querySelector('.triage-upload-area').classList.remove('hidden-section');
    }

    static resetApp() {
        document.getElementById('home-view').classList.add('hidden-section');
        document.getElementById('triage-view').classList.remove('hidden-section');
        document.getElementById('model-a-view').classList.add('hidden-section');
        document.getElementById('model-b-view').classList.add('hidden-section');
        
        this.hideLoading();
        document.getElementById('main-file-input').value = "";
    }
}
