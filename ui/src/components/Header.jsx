import React from 'react';
import { Activity, Clock, Upload } from 'lucide-react';

const Header = ({ 
    status, 
    mode, 
    setMode, 
    showHistory, 
    setShowHistory, 
    showLogs, 
    setShowLogs, 
    fileInputRef, 
    handleFileUpload 
}) => {
    return (
        <header className="bg-gray-800 p-4 border-b border-gray-700 flex justify-between items-center">
            <div className="flex items-center gap-2">
                <Activity className="text-blue-400" />
                <h1 className="text-xl font-bold">SCT Professional Dashboard</h1>
            </div>
            <div className="flex items-center gap-4">
                <div className="flex bg-gray-700 rounded p-1 gap-1">
                    <button 
                        onClick={() => setMode('live')}
                        className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${mode === 'live' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                    >
                        <Activity size={12} /> Live
                    </button>
                    <button 
                        onClick={() => setShowHistory(!showHistory)}
                        className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${showHistory ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                    >
                        <Clock size={12} /> History
                    </button>
                    <button 
                        onClick={() => fileInputRef.current.click()}
                        className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${mode === 'file' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                    >
                        <Upload size={12} /> Load File
                    </button>
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        className="hidden" 
                        accept=".jsonl,.json" 
                        onChange={handleFileUpload}
                    />
                    <button 
                        onClick={() => setShowLogs(!showLogs)}
                        className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${showLogs ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                    >
                        <div className="font-mono">Logs</div>
                    </button>
                </div>
                <span className={`px-2 py-1 rounded text-xs ${status.running ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-300'}`}>
                    {status.running ? 'RUNNING' : 'IDLE'}
                </span>
            </div>
        </header>
    );
};

export default Header;
