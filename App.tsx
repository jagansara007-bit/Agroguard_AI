
import React, { useState, useEffect, useCallback } from 'react';
import { dbService } from './db';
import { CropRecord, DashboardStats, CropStatus, UserSettings } from './types';
import { processOfflineQueue, SyncProgress } from './aiService';
import Dashboard from './components/Dashboard';
import Analyzer from './components/Analyzer';
import History from './components/History';
import Analytics from './components/Analytics';
import Insurance from './components/Insurance';
import Sidebar from './components/Sidebar';
import Settings from './components/Settings';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'analyze' | 'history' | 'analytics' | 'insurance' | 'settings'>('dashboard');
  const [records, setRecords] = useState<CropRecord[]>([]);
  const [settings, setSettings] = useState<UserSettings>(dbService.getSettings());
  const [stats, setStats] = useState<DashboardStats>({
    totalAnalyzed: 0,
    healthyPercentage: 0,
    averageYieldLoss: 0,
    activeAlerts: 0
  });
  const [isAiServerActive, setIsAiServerActive] = useState(false);
  const [syncState, setSyncState] = useState<SyncProgress>({ current: 0, total: 0, status: 'idle' });

  // Check YOLO backend status
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const url = `${window.location.protocol}//${window.location.hostname}:8000/`;
        const res = await fetch(url);
        if (res.ok) {
          if (!isAiServerActive) console.log("AI Engine Connected at", url);
          setIsAiServerActive(true);
        } else {
          setIsAiServerActive(false);
        }
      } catch (err) {
        if (isAiServerActive) console.error("AI Engine Disconnected:", err);
        setIsAiServerActive(false);
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadData = useCallback(() => {
    const allRecords = dbService.getRecords();
    setRecords(allRecords);

    if (allRecords.length > 0) {
      const healthy = allRecords.filter(r => r.status === CropStatus.HEALTHY).length;
      const avgLoss = allRecords.reduce((acc, r) => acc + r.analysis.expectedLoss, 0) / allRecords.length;
      const alerts = allRecords.filter(r => r.analysis.expectedLoss >= settings.insuranceThreshold).length;

      setStats({
        totalAnalyzed: allRecords.length,
        healthyPercentage: Math.round((healthy / allRecords.length) * 100),
        averageYieldLoss: Math.round(avgLoss),
        activeAlerts: alerts
      });
    }
  }, [settings.insuranceThreshold]);

  const processBackgroundSync = useCallback(async () => {
    if (syncState.status === 'syncing' || !isAiServerActive) return;

    const pending = dbService.getPendingRecords();
    if (pending.length === 0) return;

    await processOfflineQueue((progress) => {
      setSyncState(progress);
      if (progress.status === 'completed') {
        loadData();
        setTimeout(() => setSyncState(prev => ({ ...prev, status: 'idle' })), 3000);
      }
    });
  }, [syncState.status, loadData, isAiServerActive]);

  useEffect(() => {
    loadData();

    if (isAiServerActive && settings.autoSync) {
      processBackgroundSync();
    }
  }, [loadData, processBackgroundSync, settings.autoSync, isAiServerActive]);

  const handleNewRecord = (record: CropRecord) => {
    dbService.saveRecord(record);
    loadData();
    setActiveTab('dashboard');
  };

  const handleSettingsUpdate = (newSettings: UserSettings) => {
    dbService.saveSettings(newSettings);
    setSettings(newSettings);
    loadData();
  };

  return (
    <div className="flex min-h-screen bg-gray-50 text-gray-900">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 p-4 md:p-8 overflow-y-auto">
        <header className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold text-emerald-800">AgroGuard AI</h1>
            <p className="text-gray-500">Intelligent Crop Health & Yield Management</p>
          </div>

          <div className="flex items-center gap-3">
            {syncState.status === 'syncing' ? (
              <span className="flex items-center gap-2 px-3 py-1 bg-emerald-600 text-white text-sm font-medium rounded-full border border-emerald-700 shadow-sm animate-pulse">
                <i className="fas fa-sync-alt fa-spin"></i> {syncState.currentCrop || 'Analyzing'} {syncState.current}/{syncState.total}
              </span>
            ) : syncState.status === 'completed' ? (
              <span className="flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-medium rounded-full border border-emerald-200">
                <i className="fas fa-check-circle"></i> Queue Processed
              </span>
            ) : (
              <button
                onClick={processBackgroundSync}
                className="flex items-center gap-2 px-3 py-1 bg-indigo-100 text-indigo-700 text-sm font-medium rounded-full border border-indigo-200 hover:bg-indigo-200 transition-colors"
                title="Process Pending Scans"
              >
                <i className="fas fa-microchip"></i> AI Processor
              </button>
            )}

            {/* AI Server Status Indicator */}
            <div className={`flex items-center gap-2 px-4 py-2 bg-white rounded-xl shadow-sm border ${isAiServerActive ? 'border-emerald-100' : 'border-rose-100'}`}>
              <div className={`w-3 h-3 rounded-full ${isAiServerActive ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`}></div>
              <span className="text-sm font-medium">{isAiServerActive ? 'AI Engine Ready' : 'AI Engine Offline'}</span>
            </div>
          </div>
        </header>

        {activeTab === 'dashboard' && <Dashboard stats={stats} recentRecords={records.slice(0, 5)} onAnalyze={() => setActiveTab('analyze')} />}
        {activeTab === 'analyze' && <Analyzer onResult={handleNewRecord} isOnline={isAiServerActive} sensitivity={settings.stressSensitivity} stressThreshold={settings.stressThreshold} />}
        {activeTab === 'history' && <History records={records} onUpdate={loadData} />}
        {activeTab === 'analytics' && <Analytics records={records} />}
        {activeTab === 'insurance' && <Insurance records={records} threshold={settings.insuranceThreshold} />}
        {activeTab === 'settings' && <Settings settings={settings} onUpdate={handleSettingsUpdate} />}
      </main>
    </div>
  );
};

export default App;
