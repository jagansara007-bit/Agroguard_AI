import { YieldAnalysis, StressSensitivity, CropStatus } from "./types";
import { dbService } from "./db";

// Use local backend URL
// Use current hostname to avoid localhost vs 127.0.0.1 mismatches
const API_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

// Helper to convert base64 to Blob
const base64ToBlob = (base64: string): Blob => {
    const byteString = atob(base64.split(',')[1]);
    const mimeString = base64.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
};

/**
 * Main Analysis Function
 * Now purely local, calling the Python backend which runs YOLOv11 + ResNet/EfficientNet
 */
export const analyzeCropHealth = async (
    imageData: string | null,
    cropType: string,
    sensitivity: StressSensitivity = 'Standard'
): Promise<YieldAnalysis> => {
    if (!imageData) {
        throw new Error("No image data provided");
    }

    try {
        const blob = base64ToBlob(imageData);
        const formData = new FormData();
        formData.append('file', blob, 'image.jpg');

        // Call the new /analyze endpoint in the Python backend
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Backend Error: ${response.statusText}`);
        }

        // The backend now returns the exact YieldAnalysis JSON structure
        const result: YieldAnalysis = await response.json();
        return result;

    } catch (error: any) {
        console.error("Local AI Engine Failure:", error);
        // Fallback if local server is completely dead (though likely the app relies on it now)
        return {
            expectedLoss: 0,
            confidenceScore: 0,
            riskLevel: "Low",
            recommendations: [
                "Ensure Local AI Server is running (backend/main.py).",
                `Error: ${error.message || 'Unknown network error'}`
            ],
            diseaseDetected: "Server Error",
            diseaseDescription: `Could not connect to the local inference engine at ${API_URL}. Check if the server is running and CORS is allowed.`,
            similarityScore: 0,
            symptomlessStressDetected: false,
            stressProbability: 0,
            treatmentUrgency: "Monitoring",
            detailedMetrics: {
                leafCoverage: 0,
                spreadVelocity: "Static",
                climateRiskFactor: 0
            }
        };
    }
};

export interface SyncProgress {
    current: number;
    total: number;
    status: 'idle' | 'syncing' | 'completed' | 'failed';
    currentCrop?: string;
}

// Re-implement sync simply as "Process Queue" for offline-to-local-server handling
// (Since we are "offline" from the internet, but might be "offline" from the local server if it wasn't running)
let isProcessingQueue = false;

export const processOfflineQueue = async (onProgress?: (p: SyncProgress) => void) => {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    const pending = dbService.getPendingRecords();
    if (pending.length === 0) { isProcessingQueue = false; return; }

    const settings = dbService.getSettings();

    try {
        // Check if backend is reachable first
        try {
            await fetch(`${API_URL}/`);
        } catch {
            isProcessingQueue = false;
            return; // Backend not running, can't process
        }

        for (let i = 0; i < pending.length; i++) {
            const record = pending[i];
            if (onProgress) onProgress({ current: i + 1, total: pending.length, status: 'syncing', currentCrop: record.cropType });

            try {
                const result = await analyzeCropHealth(record.imageUrl || null, record.cropType, settings.stressSensitivity);

                // Update record with actual result
                dbService.updateRecord(record.id, {
                    analysis: result,
                    status: result.riskLevel === 'High' ? CropStatus.CRITICAL : (result.riskLevel === 'Medium' ? CropStatus.DISEASED : CropStatus.HEALTHY),
                    isPendingSync: false
                });
            } catch (err) {
                console.error("Queue process error for", record.id, err);
            }
        }
        if (onProgress) onProgress({ current: pending.length, total: pending.length, status: 'completed' });
    } finally {
        isProcessingQueue = false;
    }
};
