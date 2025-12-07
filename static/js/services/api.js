export class APIService {
    static async analyzeImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }

    static async generateReport(payload) {
        const response = await fetch('/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error("Failed to generate report");
        }
        return await response.blob();
    }

    static async sendChatMessage(message, context) {
        const payload = {
            message: message,
            context: context
        };

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        return await response.json();
    }
}
