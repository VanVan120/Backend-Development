import { APIService } from './services/api.js';
import { UIManager } from './ui/ui_manager.js';
import { ModelAUI } from './ui/model_a_ui.js';
import { ModelBUI } from './ui/model_b_ui.js';
import { ChatUI } from './ui/chat_ui.js';

class App {
    constructor() {
        this.currentAnalysisData = null;
        this.modelAUI = new ModelAUI();
        this.modelBUI = new ModelBUI();
        // Pass context provider to ChatUI
        this.chatUI = new ChatUI(() => this.currentAnalysisData);
        
        this.initEventListeners();
        this.initDragAndDrop();
        this.initActiveNav();
        this.initTestimonials();
        this.initNewsletter();
    }

    initNewsletter() {
        const form = document.getElementById('newsletter-form');
        const btn = document.getElementById('newsletter-btn');
        const input = document.getElementById('newsletter-email');

        if (form && btn && input) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault(); // Prevent default form submission

                const email = input.value.trim();
                if (!email) {
                    this.showToast("Please enter a valid email address.", "error");
                    return;
                }

                const originalText = btn.innerText;
                btn.innerText = "Sending...";
                btn.disabled = true;

                try {
                    const response = await fetch('/api/subscribe', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ email: email })
                    });

                    if (response.ok) {
                        this.showToast("Thank you for subscribing!", "success");
                        input.value = "";
                    } else {
                        const data = await response.json();
                        this.showToast("Subscription failed: " + (data.detail || "Unknown error"), "error");
                    }
                } catch (error) {
                    console.error("Newsletter Error:", error);
                    this.showToast("An error occurred. Please try again later.", "error");
                } finally {
                    btn.innerText = originalText;
                    btn.disabled = false;
                }
            });
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast-notification ${type}`;
        
        let iconClass = 'fa-info-circle';
        let title = 'Notification';
        
        if (type === 'success') {
            iconClass = 'fa-check-circle';
            title = 'Success';
        } else if (type === 'error') {
            iconClass = 'fa-exclamation-circle';
            title = 'Error';
        }

        toast.innerHTML = `
            <div class="toast-icon">
                <i class="fas ${iconClass}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;

        container.appendChild(toast);

        // Trigger animation
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 400); // Wait for transition to finish
        }, 5000);
    }

    initTestimonials() {
        let openTimeout;
        let closeTimeout;

        const openModal = (card) => {
            const modal = document.getElementById('testimonial-modal');
            const modalBody = document.getElementById('testimonial-modal-body');
            
            // Extract data from the card
            const ratingHtml = card.querySelector('.text-warning').innerHTML;
            const quoteText = card.querySelector('p').innerText.replace(/"/g, ''); 
            const userImgSrc = card.querySelector('img').src;
            const userName = card.querySelector('h6').innerText;
            const userRole = card.querySelector('small').innerText;

            // Construct new elegant layout
            const newContent = `
                <div class="testimonial-modal-body">
                    <i class="fas fa-quote-left testimonial-modal-quote-icon"></i>
                    
                    <div class="testimonial-modal-rating animate-fade-up delay-100">
                        ${ratingHtml}
                    </div>
                    
                    <p class="testimonial-modal-text animate-fade-up delay-200">
                        "${quoteText}"
                    </p>
                </div>

                <div class="testimonial-modal-footer animate-fade-up delay-300">
                    <div class="testimonial-user-info">
                        <img src="${userImgSrc}" alt="${userName}">
                        <div>
                            <div class="d-flex align-items-center">
                                <h6 class="fw-bold mb-0 text-dark" style="font-size: 1.1rem;">${userName}</h6>
                                <span class="verified-badge"><i class="fas fa-check-circle me-1"></i> Verified</span>
                            </div>
                            <small class="text-muted text-uppercase fw-bold" style="font-size: 0.75rem; letter-spacing: 1px;">${userRole}</small>
                        </div>
                    </div>
                    
                    <button class="btn-helpful" onclick="this.classList.toggle('active'); const icon = this.querySelector('i'); if(this.classList.contains('active')) { icon.classList.remove('far'); icon.classList.add('fas'); } else { icon.classList.remove('fas'); icon.classList.add('far'); } event.stopPropagation();">
                        <i class="far fa-thumbs-up"></i> Helpful
                    </button>
                </div>
            `;

            modalBody.innerHTML = newContent;
            modal.classList.remove('hidden-section');
        };

        window.expandTestimonial = (card, isClick = false) => {
            clearTimeout(closeTimeout);
            
            if (isClick) {
                clearTimeout(openTimeout);
                openModal(card);
            } else {
                // Hover behavior - wait 2 seconds
                openTimeout = setTimeout(() => {
                    openModal(card);
                }, 2000);
            }
        };

        window.closeTestimonial = (event) => {
            clearTimeout(openTimeout); // Cancel any pending open
            
            // Add delay to allow moving mouse into the modal content
            closeTimeout = setTimeout(() => {
                const modal = document.getElementById('testimonial-modal');
                modal.classList.add('hidden-section');
            }, 100);
        };

        // Keep modal open when hovering the modal content itself
        const modalContent = document.querySelector('.testimonial-modal-content');
        if (modalContent) {
            modalContent.addEventListener('mouseenter', () => {
                clearTimeout(closeTimeout);
            });
            modalContent.addEventListener('mouseleave', () => {
                window.closeTestimonial();
            });
        }
    }

    initActiveNav() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.navbar-links a');
        
        // Reset all
        navLinks.forEach(link => link.classList.remove('active'));

        // Check for hash first (Services)
        if (window.location.hash === '#start-screening') {
            const servicesLink = document.getElementById('nav-services');
            if (servicesLink) servicesLink.classList.add('active');
            return;
        }

        // Check path
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPath || (currentPath === '/' && href === '/')) {
                // If we are on home page, check if we are in analysis mode
                if (currentPath === '/' && !document.getElementById('triage-view')?.classList.contains('hidden-section')) {
                     // Analysis is visible, so Services should be active, not Home
                     if (link.id === 'nav-home') return; 
                }
                link.classList.add('active');
            }
        });
    }

    updateNavState(view) {
        const navLinks = document.querySelectorAll('.navbar-links a');
        navLinks.forEach(link => link.classList.remove('active'));

        if (view === 'home') {
            const homeLink = document.getElementById('nav-home');
            if (homeLink) homeLink.classList.add('active');
        } else if (view === 'services') {
            const servicesLink = document.getElementById('nav-services');
            if (servicesLink) servicesLink.classList.add('active');
        }
    }

    initEventListeners() {
        // Navigation
        window.showHome = () => {
            if (document.getElementById('home-view')) {
                UIManager.showHome();
                this.updateNavState('home');
            } else {
                window.location.href = '/';
            }
        };

        window.showAnalysis = () => {
            if (document.getElementById('triage-view')) {
                UIManager.showAnalysis();
                this.updateNavState('services');
            } else {
                window.location.href = '/#start-screening';
            }
        };

        // Expose resetApp globally for "Back" buttons
        window.resetApp = () => {
            if (document.getElementById('triage-view')) {
                UIManager.showAnalysis();
                this.updateNavState('services');
            } else {
                window.location.href = '/#start-screening';
            }
        };

        // Check for redirect hash
        if (window.location.hash === '#start-screening') {
            if (document.getElementById('triage-view')) {
                setTimeout(() => {
                    UIManager.showAnalysis();
                    this.updateNavState('services');
                    history.replaceState(null, null, ' ');
                }, 100);
            }
        }
        
        // Upload
        const fileInput = document.getElementById('main-file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleMainUpload(e.target));
        }

        // Model A Controls
        const btnOriginal = document.getElementById('ma-btn-original');
        if (btnOriginal) btnOriginal.addEventListener('click', () => this.modelAUI.switchImage('original'));

        // Model A Back Button
        const btnBackA = document.getElementById('btn-back-model-a');
        if (btnBackA) {
            btnBackA.addEventListener('click', () => {
                UIManager.showAnalysis();
                this.updateNavState('services');
            });
        }

        // Model A Bottom New Analysis Button
        const btnNewUploadA = document.getElementById('btn-new-upload-a-bottom');
        if (btnNewUploadA) {
            btnNewUploadA.addEventListener('click', () => {
                UIManager.showAnalysis();
                this.updateNavState('services');
            });
        }
        
        const btnHeatmap = document.getElementById('ma-btn-heatmap');
        if (btnHeatmap) btnHeatmap.addEventListener('click', () => this.modelAUI.switchImage('heatmap'));
        
        const btnReportA = document.getElementById('btn-report-a');
        if (btnReportA) btnReportA.addEventListener('click', () => this.downloadReport('A'));

        // Use event delegation for Email buttons to ensure they work even if DOM updates
        document.addEventListener('click', (e) => {
            const btnA = e.target.closest('#btn-email-a');
            if (btnA) {
                e.preventDefault();
                console.log('Email Button A Clicked');
                this.handleEmailReport('A');
            }

            const btnB = e.target.closest('#btn-email-b');
            if (btnB) {
                e.preventDefault();
                console.log('Email Button B Clicked');
                this.handleEmailReport('B');
            }
        });

        // Model B Controls
        const btnReportB = document.getElementById('btn-report-b');
        if (btnReportB) btnReportB.addEventListener('click', () => this.downloadReport('B'));

        // Model B Back Button
        const btnBackB = document.getElementById('btn-back-model-b');
        if (btnBackB) {
            btnBackB.addEventListener('click', () => {
                UIManager.showAnalysis();
                this.updateNavState('services');
            });
        }

        // Model B Bottom New Analysis Button
        const btnNewUploadB = document.getElementById('btn-new-upload-b-bottom');
        if (btnNewUploadB) {
            btnNewUploadB.addEventListener('click', () => {
                UIManager.showAnalysis();
                this.updateNavState('services');
            });
        }

        // Chat Controls are now handled inside ChatUI class
    }

    handleMainUpload(input) {
        if (!input.files || !input.files[0]) return;
        this.processFile(input.files[0]);
    }

    async processFile(file) {
        UIManager.showLoading();

        try {
            const data = await APIService.analyzeImage(file);

            if (data.error) {
                alert("Error: " + data.error);
                UIManager.resetApp();
                return;
            }

            this.currentAnalysisData = data;

            if (data.model_used === "Model A (Histopathology)") {
                // Inject AI suggestion into the analysis object
                if (data.final_analysis) {
                    data.final_analysis.ai_suggestion = data.ai_suggestion;
                }
                this.modelAUI.show(data.final_analysis, file);
            } else if (data.model_used === "Model B (Clinical Screening)") {
                // Inject AI suggestion into the analysis object
                if (data.final_analysis) {
                    data.final_analysis.ai_suggestion = data.ai_suggestion;
                }
                this.modelBUI.show(data.final_analysis, file);
            } else {
                alert("Unknown result type.");
                UIManager.resetApp();
            }

        } catch (e) {
            console.error(e);
            alert("Analysis failed. Please try again.");
            UIManager.resetApp();
        }
    }

    async downloadReport(modelType) {
        if (!this.currentAnalysisData) {
            alert("No analysis data available. Please upload an image first.");
            return;
        }

        const btnId = modelType === 'A' ? 'btn-report-a' : 'btn-report-b';
        const btn = document.getElementById(btnId);
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        btn.disabled = true;

        try {
            const reportType = modelType === 'A' ? 'expert' : 'public';
            
            let imageSrc;
            if (modelType === 'A') {
                imageSrc = this.modelAUI.getImageSrc();
            } else {
                imageSrc = this.modelBUI.getImageSrc();
            }

            const payload = {
                report_type: reportType,
                analysis_data: this.currentAnalysisData,
                image_base64: imageSrc
            };

            const blob = await APIService.generateReport(payload);
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Oral_Cancer_Report_${modelType}_${new Date().toISOString().slice(0,10)}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();

        } catch (error) {
            console.error("Report Error:", error);
            alert("Failed to generate report. Please try again.");
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    async handleEmailReport(modelType) {
        if (!this.currentAnalysisData) {
            alert("No analysis data available. Please upload an image first.");
            return;
        }

        const modalEl = document.getElementById('emailReportModal');
        const emailInput = document.getElementById('reportEmailInput');
        const sendBtn = document.getElementById('btn-send-email-confirm');
        const errorMsg = document.getElementById('emailError');
        
        if (!modalEl || !emailInput || !sendBtn) {
            console.error("Email modal elements not found");
            // Fallback to prompt if modal fails
            const email = prompt("Please enter your email address:");
            if (email) this.sendEmail(modelType, email);
            return;
        }

        // Reset modal state
        emailInput.value = '';
        emailInput.classList.remove('is-invalid');
        errorMsg.classList.add('d-none');
        
        const modal = new bootstrap.Modal(modalEl);
        modal.show();

        // Handle Send Click
        const handleSend = async () => {
            const email = emailInput.value.trim();
            if (!email || !email.includes('@')) {
                emailInput.classList.add('is-invalid');
                errorMsg.classList.remove('d-none');
                return;
            }

            modal.hide();
            await this.sendEmail(modelType, email);
        };

        // Remove existing listeners to avoid duplicates
        const newBtn = sendBtn.cloneNode(true);
        sendBtn.parentNode.replaceChild(newBtn, sendBtn);
        newBtn.addEventListener('click', handleSend);
    }

    async sendEmail(modelType, email) {
        const btnId = modelType === 'A' ? 'btn-email-a' : 'btn-email-b';
        const btn = document.getElementById(btnId);
        let originalText = '';
        
        if (btn) {
            originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Sending...';
            btn.disabled = true;
        }

        try {
            const reportType = modelType === 'A' ? 'expert' : 'public';
            
            let imageSrc;
            if (modelType === 'A') {
                imageSrc = this.modelAUI.getImageSrc();
            } else {
                imageSrc = this.modelBUI.getImageSrc();
            }

            const payload = {
                report_type: reportType,
                analysis_data: this.currentAnalysisData,
                image_base64: imageSrc,
                email: email
            };

            await APIService.sendReportEmail(payload);
            alert(`Report successfully sent to ${email}!`);

        } catch (error) {
            console.error("Email Report Error:", error);
            alert("Failed to send email. Please try again later.");
        } finally {
            if (btn) {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
    }

    initDragAndDrop() {
        const dropZone = document.querySelector('.triage-upload-area');
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.style.borderColor = 'var(--primary-color)';
                dropZone.style.backgroundColor = '#eff6ff';
                dropZone.style.transform = 'scale(1.02)';
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.style.borderColor = '#cbd5e1';
                dropZone.style.backgroundColor = 'var(--card-bg)';
                dropZone.style.transform = 'scale(1)';
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                this.processFile(files[0]);
            }
        }, false);
    }

    initSecurity() {
        // Disable right-click on images
        document.addEventListener('contextmenu', (e) => {
            if (e.target.tagName === 'IMG') {
                e.preventDefault();
            }
        });

        // Disable drag-and-drop for images
        document.addEventListener('dragstart', (e) => {
            if (e.target.tagName === 'IMG') {
                e.preventDefault();
            }
        });
    }
}

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.initSecurity();
    window.resetApp = () => UIManager.resetApp();
    
    // Global Fullscreen Toggle
    window.toggleFullscreen = function(element) {
        if (!document.fullscreenElement) {
            if (element.requestFullscreen) {
                element.requestFullscreen();
            } else if (element.webkitRequestFullscreen) { /* Safari */
                element.webkitRequestFullscreen();
            } else if (element.msRequestFullscreen) { /* IE11 */
                element.msRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) { /* Safari */
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) { /* IE11 */
                document.msExitFullscreen();
            }
        }
    };
});
