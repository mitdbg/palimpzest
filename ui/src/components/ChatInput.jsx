import React, { useState, useEffect } from 'react';
import { Send, Square, Paperclip, ChevronDown, Filter, X } from 'lucide-react';

const ChatInput = ({ 
    runConfig, 
    setRunConfig, 
    runQuery, 
    stopQuery, 
    isRunning, 
    onOpenSettings,
    mode = 'centered' // 'centered' or 'bottom'
}) => {
    const [localQuery, setLocalQuery] = useState(runConfig.query);
    const [showModelMenu, setShowModelMenu] = useState(false);
    const [showFilterMenu, setShowFilterMenu] = useState(false);

    useEffect(() => {
        setLocalQuery(runConfig.query);
    }, [runConfig.query]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isRunning && localQuery.trim()) {
                runQuery();
            }
        }
    };

    const handleChange = (e) => {
        setLocalQuery(e.target.value);
        setRunConfig(prev => ({ ...prev, query: e.target.value }));
    };

    const removeFilter = (index) => {
        const newFilters = [...(runConfig.filters || [])];
        newFilters.splice(index, 1);
        setRunConfig(prev => ({ ...prev, filters: newFilters }));
    };

    const addQuickFilter = (field, op, val) => {
        setRunConfig(prev => ({
            ...prev,
            filters: [...(prev.filters || []), { field, operator: op, value: val }]
        }));
        setShowFilterMenu(false);
    };

    const containerClasses = mode === 'centered' 
        ? "w-full max-w-2xl mx-auto" 
        : "w-full max-w-4xl mx-auto";

    // Simplified model list for the quick picker
    const quickModels = [
        "openrouter/x-ai/grok-4.1-fast",
        "gpt-4o",
        "claude-3-5-sonnet-20240620"
    ];

    return (
        <div className={containerClasses}>
            {/* Active Filter Chips */}
            {runConfig.filters && runConfig.filters.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2 px-1">
                    {runConfig.filters.map((f, i) => (
                        <div key={i} className="flex items-center gap-1 bg-blue-900/40 border border-blue-700/50 rounded-full px-3 py-1 text-xs text-blue-200">
                            <span className="font-mono opacity-70">{f.field}</span>
                            <span className="text-blue-400">{f.operator}</span>
                            <span className="font-semibold">{f.value}</span>
                            <button onClick={() => removeFilter(i)} className="ml-1 hover:text-white">
                                <X size={12} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            <div className="relative bg-gray-800 border border-gray-700 rounded-xl shadow-lg focus-within:ring-2 focus-within:ring-blue-500/50 transition-all">
                <textarea
                    value={localQuery}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a question about the graph..."
                    className="w-full bg-transparent text-white p-4 pr-12 pb-12 rounded-xl resize-none outline-none min-h-[60px] max-h-[200px]"
                    rows={mode === 'centered' ? 3 : 1}
                    style={{ fieldSizing: 'content' }} 
                />
                
                {/* Bottom Bar inside Input */}
                <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <div className="relative">
                            <button 
                                onClick={() => setShowModelMenu(!showModelMenu)}
                                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 px-2 py-1 rounded hover:bg-gray-700 transition-colors"
                            >
                                {runConfig.model.split('/').pop()} <ChevronDown size={12} />
                            </button>
                            
                            {showModelMenu && (
                                <div className="absolute top-full left-0 mt-1 w-48 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-20 overflow-hidden">
                                    {quickModels.map(m => (
                                        <button
                                            key={m}
                                            onClick={() => {
                                                setRunConfig(prev => ({ ...prev, model: m }));
                                                setShowModelMenu(false);
                                            }}
                                            className={`w-full text-left px-3 py-2 text-xs hover:bg-gray-800 ${runConfig.model === m ? 'text-blue-400' : 'text-gray-300'}`}
                                        >
                                            {m.split('/').pop()}
                                        </button>
                                    ))}
                                    <div className="border-t border-gray-800 mt-1 pt-1">
                                        <button 
                                            onClick={() => {
                                                onOpenSettings();
                                                setShowModelMenu(false);
                                            }}
                                            className="w-full text-left px-3 py-2 text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800"
                                        >
                                            More options...
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>

                        <button 
                            onClick={() => onOpenSettings()}
                            className="flex items-center gap-1 text-xs text-gray-400 hover:text-blue-300 px-2 py-1 rounded hover:bg-gray-700 transition-colors"
                            title="Add Filter"
                        >
                            <Filter size={12} />
                            <span>Filter</span>
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        {isRunning ? (
                            <button 
                                onClick={stopQuery}
                                className="p-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                            >
                                <Square size={16} />
                            </button>
                        ) : (
                            <button 
                                onClick={runQuery}
                                disabled={!localQuery.trim()}
                                className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                <Send size={16} />
                            </button>
                        )}
                    </div>
                </div>
            </div>
            {mode === 'centered' && (
                <div className="mt-4 flex justify-center gap-4 text-sm text-gray-500">
                    <button onClick={onOpenSettings} className="hover:text-gray-300 transition-colors">Configure Search</button>
                </div>
            )}
        </div>
    );
};

export default ChatInput;
