import React from 'react';
import { X } from 'lucide-react';

const SettingsModal = ({ 
    isOpen, 
    onClose, 
    runConfig, 
    setRunConfig, 
    resources, 
    availableModels,
    devMode,
    setDevMode
}) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="bg-gray-900 border border-gray-700 rounded-lg w-[500px] max-h-[80vh] overflow-y-auto shadow-2xl">
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <h2 className="text-lg font-semibold text-white">Configuration</h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">
                        <X size={20} />
                    </button>
                </div>
                
                <div className="p-6 space-y-6">
                    {/* Developer Mode Toggle */}
                    <div className="flex items-center justify-between bg-gray-800 p-3 rounded border border-gray-700">
                        <div>
                            <div className="text-sm font-medium text-white">Developer Mode</div>
                            <div className="text-xs text-gray-400">Enable detailed metrics, raw data, and debug traces</div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                            <input 
                                type="checkbox" 
                                className="sr-only peer"
                                checked={devMode}
                                onChange={(e) => setDevMode(e.target.checked)}
                            />
                            <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                    </div>

                    {/* Graph Snapshot Selection */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Graph Snapshot</label>
                        <select 
                            className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                            value={runConfig.index}
                            onChange={(e) => setRunConfig({...runConfig, index: e.target.value})}
                        >
                            {resources.indices.length === 0 && <option value="">Loading snapshots...</option>}
                            {resources.indices.map(idx => (
                                <option key={idx} value={idx}>{idx}</option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Select a graph snapshot from CURRENT_WORKSTREAM/exports</p>
                    </div>

                    {/* Model Selection */}
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Orchestrator Model</label>
                        <select 
                            className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                            value={runConfig.model}
                            onChange={(e) => setRunConfig({...runConfig, model: e.target.value})}
                        >
                            {availableModels.map(m => (
                                <option key={m} value={m}>{m.split('/').pop()}</option>
                            ))}
                        </select>
                    </div>

                    {/* Advanced Config */}
                    <div className="space-y-4 pt-4 border-t border-gray-700">
                        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Advanced</h3>
                        
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">Ranking Model</label>
                            <select 
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                                value={runConfig.ranking_model}
                                onChange={(e) => setRunConfig({...runConfig, ranking_model: e.target.value})}
                            >
                                <option value="">Default (Orchestrator)</option>
                                <option value="cross-encoder/ms-marco-MiniLM-L-6-v2">Local Reranker (MiniLM)</option>
                                <option value="Qwen/Qwen3-Reranker-0.6B">Local Reranker (Qwen 0.6B)</option>
                                {availableModels.map(m => (
                                    <option key={`rank-${m}`} value={m}>{m.split('/').pop()}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-xs text-gray-400 mb-1">Admittance Model</label>
                            <select 
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                                value={runConfig.admittance_model}
                                onChange={(e) => setRunConfig({...runConfig, admittance_model: e.target.value})}
                            >
                                <option value="">Default (Orchestrator)</option>
                                {availableModels.map(m => (
                                    <option key={`adm-${m}`} value={m}>{m.split('/').pop()}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-xs text-gray-400 mb-1">Termination Model</label>
                            <select 
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                                value={runConfig.termination_model}
                                onChange={(e) => setRunConfig({...runConfig, termination_model: e.target.value})}
                            >
                                <option value="">Default (Orchestrator)</option>
                                {availableModels.map(m => (
                                    <option key={`term-${m}`} value={m}>{m.split('/').pop()}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-xs text-gray-400 mb-1">Entry Points (K)</label>
                            <input 
                                type="number"
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                                value={runConfig.entry_points}
                                onChange={(e) => setRunConfig({...runConfig, entry_points: parseInt(e.target.value) || 5})}
                                min="1"
                                max="50"
                            />
                        </div>

                        <div>
                            <label className="block text-xs text-gray-400 mb-1">Max Steps</label>
                            <input 
                                type="number"
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                                value={runConfig.max_steps || 200}
                                onChange={(e) => setRunConfig({...runConfig, max_steps: parseInt(e.target.value) || 200})}
                                min="1"
                                max="1000"
                            />
                        </div>

                        <div>
                            <label className="block text-xs text-gray-400 mb-1">Edge Type Filter</label>
                            <input 
                                type="text"
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                                placeholder="e.g. 'contains', 'related_to' or leave empty for all"
                                value={runConfig.edge_type || ""}
                                onChange={(e) => setRunConfig({...runConfig, edge_type: e.target.value})}
                            />
                        </div>
                    </div>
                </div>
                
                <div className="p-4 border-t border-gray-700 flex justify-end">
                    <button 
                        onClick={onClose}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SettingsModal;
