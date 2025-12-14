import React, { useState } from 'react';
import { Activity, Database, List, DollarSign, Zap, FileText, Terminal, Download, ChevronRight, ChevronLeft } from 'lucide-react';

const RunDetails = ({ metrics, evidence, queue, currentAction, finalAnswer, devMode, onShowLogs, onExport, onSelectNode }) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [width, setWidth] = useState(320);
    const [isResizing, setIsResizing] = useState(false);

    const startResizing = (e) => {
        setIsResizing(true);
    };

    const stopResizing = () => {
        setIsResizing(false);
    };

    const resize = (e) => {
        if (isResizing) {
            const newWidth = window.innerWidth - e.clientX;
            if (newWidth > 200 && newWidth < 600) {
                setWidth(newWidth);
            }
        }
    };

    React.useEffect(() => {
        window.addEventListener('mousemove', resize);
        window.addEventListener('mouseup', stopResizing);
        return () => {
            window.removeEventListener('mousemove', resize);
            window.removeEventListener('mouseup', stopResizing);
        };
    }, [isResizing]);

    return (
        <div 
            className={`bg-gray-900 border-l border-gray-800 flex flex-col h-full relative group ${isResizing ? '' : 'transition-all duration-300 ease-in-out'}`}
            style={{ width: isCollapsed ? 48 : width }}
        >
            {/* Resize Handle */}
            <div 
                className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500/50 z-10 transition-colors"
                onMouseDown={startResizing}
            />

            {/* Collapse Toggle */}
            <button 
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="absolute -left-3 top-3 bg-gray-800 border border-gray-700 text-gray-400 rounded-full p-1 hover:text-white hover:bg-gray-700 z-20 opacity-0 group-hover:opacity-100 transition-opacity"
            >
                {isCollapsed ? <ChevronLeft size={12} /> : <ChevronRight size={12} />}
            </button>

            {isCollapsed ? (
                // Collapsed View
                <div className="flex flex-col items-center py-4 gap-6">
                    <div className="flex flex-col items-center gap-1 text-gray-500" title="Metrics">
                        <Activity size={16} />
                    </div>
                    <div className="flex flex-col items-center gap-1 text-gray-500" title="Evidence">
                        <Database size={16} />
                        <span className="text-[10px] font-mono">{evidence.length}</span>
                    </div>
                    {devMode && (
                        <div className="flex flex-col items-center gap-1 text-gray-500" title="Queue">
                            <List size={16} />
                            <span className="text-[10px] font-mono">{queue.length}</span>
                        </div>
                    )}
                </div>
            ) : (
                // Expanded View
                <>
                    {/* Metrics - Only show full metrics in Dev Mode */}
                    {devMode && (
                        <div className="p-4 border-b border-gray-800">
                            <div className="flex justify-between items-center mb-3">
                                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                                    <Activity size={12} /> Run Metrics
                                </h3>
                                <div className="flex gap-2">
                                    <button 
                                        onClick={onExport}
                                        className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
                                        title="Export Trace JSON"
                                    >
                                        <Download size={10} /> Export
                                    </button>
                                    <button 
                                        onClick={onShowLogs}
                                        className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                    >
                                        <Terminal size={10} /> Logs
                                    </button>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-gray-800/50 p-2 rounded border border-gray-800">
                                    <div className="text-[10px] text-gray-400 mb-1 flex items-center gap-1"><DollarSign size={10}/> Cost</div>
                                    <div className="font-mono font-medium text-sm text-white">${metrics.cost.toFixed(4)}</div>
                                </div>
                                <div className="bg-gray-800/50 p-2 rounded border border-gray-800">
                                    <div className="text-[10px] text-gray-400 mb-1 flex items-center gap-1"><Zap size={10}/> Calls</div>
                                    <div className="font-mono font-medium text-sm text-white">{metrics.calls}</div>
                                </div>
                                <div className="bg-gray-800/50 p-2 rounded border border-gray-800">
                                    <div className="text-[10px] text-gray-400 mb-1 flex items-center gap-1"><FileText size={10}/> Tokens</div>
                                    <div className="font-mono font-medium text-sm text-white">{(metrics.tokens / 1000).toFixed(1)}k</div>
                                </div>
                                <div className="bg-gray-800/50 p-2 rounded border border-gray-800">
                                    <div className="text-[10px] text-gray-400 mb-1">Shortcuts</div>
                                    <div className="font-mono font-medium text-sm text-white">{metrics.shortcuts}</div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Current Action */}
                    <div className="p-4 border-b border-gray-800 bg-gray-800/20">
                        <div className="text-xs text-gray-500 mb-1">Status</div>
                        <div className="text-xs font-mono text-green-400 break-all">
                            {currentAction}
                        </div>
                    </div>

                    {/* Evidence */}
                    <div className="flex-1 overflow-y-auto p-4 min-h-0 scrollbar-thin scrollbar-thumb-gray-800">
                        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2 sticky top-0 bg-gray-900 py-2 z-10">
                            <Database size={12} /> Evidence ({evidence.length})
                        </h3>
                        <div className="space-y-2 mb-6">
                            {evidence.length === 0 && <div className="text-xs text-gray-600 italic">No evidence collected yet.</div>}
                            {evidence.map((e, i) => (
                                <div 
                                    key={i} 
                                    onClick={() => onSelectNode?.({ id: e.node_id || e.id })}
                                    className="text-xs bg-gray-800/50 p-3 rounded border border-gray-700/50 hover:border-yellow-500/50 transition-colors group/item cursor-pointer hover:bg-gray-800"
                                >
                                    <div className="flex justify-between items-start mb-1">
                                        <div className="font-bold text-yellow-500 font-mono">{(e.node_id || e.id || "").substring(0, 8)}</div>
                                        <div className="text-gray-500">{e.score?.toFixed(2)}</div>
                                    </div>
                                    <div className="text-gray-300 line-clamp-3 group-hover/item:line-clamp-none transition-all">{e.summary}</div>
                                </div>
                            ))}
                        </div>

                        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2 sticky top-0 bg-gray-900 py-2 z-10">
                            <List size={12} /> Priority Queue
                        </h3>
                        {devMode ? (
                            <div className="space-y-1">
                                {queue.length === 0 && <div className="text-xs text-gray-600 italic">Queue empty.</div>}
                                {queue.map((q, i) => (
                                    <div key={i} className="text-xs bg-gray-800/30 p-2 rounded flex justify-between items-center border border-transparent hover:border-gray-700">
                                        <span className="font-mono truncate flex-1 mr-2 text-gray-400" title={q.summary || q.id}>
                                            {q.summary ? q.summary.substring(0, 25) + (q.summary.length > 25 ? '...' : '') : q.id.substring(0, 8)}
                                        </span>
                                        <span className="text-blue-400 font-mono">
                                            {typeof q.score === 'number' ? q.score.toFixed(2) : '-'}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-xs text-gray-600 italic">
                                {queue.length} items in queue. Enable Developer Mode to view.
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
};

export default RunDetails;
