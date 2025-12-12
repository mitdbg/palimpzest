import React from 'react';
import { X, Play, Clock } from 'lucide-react';

const HistoryModal = ({ history, onClose, onLoadRun }) => {
    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
                <div className="p-4 border-b border-gray-700 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Clock className="text-blue-400" size={20} />
                        <h2 className="text-lg font-bold text-white">Run History</h2>
                    </div>
                    <button 
                        onClick={onClose}
                        className="text-gray-400 hover:text-white p-1 rounded hover:bg-gray-700"
                    >
                        <X size={20} />
                    </button>
                </div>
                
                <div className="flex-1 overflow-y-auto p-4 space-y-2">
                    {history.length === 0 ? (
                        <div className="text-center text-gray-500 py-8">No history available</div>
                    ) : (
                        history.map((run) => (
                            <div 
                                key={run.run_id}
                                className="bg-gray-900 border border-gray-700 rounded p-3 hover:border-blue-500 transition-colors group"
                            >
                                <div className="flex justify-between items-start mb-2">
                                    <div className="font-mono text-xs text-gray-500">{run.run_id}</div>
                                    <div className="text-xs text-gray-400">{new Date(run.timestamp).toLocaleString()}</div>
                                </div>
                                <div className="text-sm text-gray-200 mb-3 line-clamp-2">{run.query}</div>
                                <div className="flex justify-end">
                                    <button 
                                        onClick={() => onLoadRun(run.run_id)}
                                        className="flex items-center gap-1 text-xs bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded"
                                    >
                                        <Play size={12} /> Load Replay
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default HistoryModal;
