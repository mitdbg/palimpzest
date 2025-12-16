import React, { useState } from 'react';
import { MessageSquare, Clock, Settings, Plus, ChevronLeft, ChevronRight, Compass, Trash2, X, Loader2 } from 'lucide-react';

const formatRelativeTime = (timestamp) => {
    if (!timestamp) return '';
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString();
};

const HistorySidebar = ({ 
    queries, 
    selectedQueryId, 
    setSelectedQueryId, 
    onNewChat, 
    onOpenSettings,
    runHistory,
    loadRun,
    onDeleteRun,
    onClearHistory,
    onExploreGraph,
    isExploring,
    isRunning,
    currentQuery
}) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [width, setWidth] = useState(256);
    const [isResizing, setIsResizing] = useState(false);
    const [hoveredRunId, setHoveredRunId] = useState(null);
    const [showClearConfirm, setShowClearConfirm] = useState(false);

    const startResizing = (e) => {
        setIsResizing(true);
    };

    const stopResizing = () => {
        setIsResizing(false);
    };

    const resize = (e) => {
        if (isResizing) {
            const newWidth = e.clientX;
            if (newWidth > 100 && newWidth < 400) {
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
            className={`bg-black border-r border-gray-800 flex flex-col h-full relative group ${isResizing ? '' : 'transition-all duration-300 ease-in-out'}`}
            style={{ width: isCollapsed ? 64 : width }}
        >
            {/* Resize Handle */}
            <div 
                className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500/50 z-10 transition-colors"
                onMouseDown={startResizing}
            />

            {/* Collapse Toggle */}
            <button 
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="absolute -right-3 top-3 bg-gray-800 border border-gray-700 text-gray-400 rounded-full p-1 hover:text-white hover:bg-gray-700 z-20 opacity-0 group-hover:opacity-100 transition-opacity"
            >
                {isCollapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
            </button>

            {/* New Chat Button */}
            <div className="p-3">
                <button 
                    onClick={onNewChat}
                    className={`w-full flex items-center gap-2 bg-gray-900 hover:bg-gray-800 text-white p-2 rounded-lg border border-gray-700 transition-colors text-sm font-medium ${isCollapsed ? 'justify-center' : ''}`}
                    title="New Chat"
                >
                    <Plus size={18} /> 
                    {!isCollapsed && <span>New Chat</span>}
                </button>

                <button
                    onClick={onExploreGraph}
                    className={`mt-2 w-full flex items-center gap-2 p-2 rounded-lg border transition-colors text-sm font-medium ${isCollapsed ? 'justify-center' : ''} ${isExploring ? 'bg-blue-600 border-blue-500 text-white' : 'bg-gray-900 hover:bg-gray-800 border-gray-700 text-gray-200'}`}
                    title="Explore Graph"
                >
                    <Compass size={18} />
                    {!isCollapsed && <span>Explore Graph</span>}
                </button>
            </div>

            {/* History List */}
            <div className="flex-1 overflow-y-auto px-2 scrollbar-thin scrollbar-thumb-gray-800">
                {!isCollapsed && queries.length > 0 && (
                    <>
                        <div className="text-xs font-semibold text-gray-500 px-2 mb-2 mt-2 uppercase tracking-wider">Session</div>
                        <div className="space-y-1 mb-6">
                            {queries.map(q => (
                                <button
                                    key={q.id}
                                    onClick={() => setSelectedQueryId(q.id)}
                                    className={`w-full text-left p-2 rounded-lg text-sm truncate flex items-center gap-2 transition-colors ${
                                        selectedQueryId === q.id 
                                            ? 'bg-gray-800 text-white' 
                                            : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
                                    }`}
                                >
                                    <MessageSquare size={14} className="shrink-0" />
                                    <span className="truncate">{q.text}</span>
                                </button>
                            ))}
                        </div>
                    </>
                )}

                {/* Currently Running Indicator */}
                {!isCollapsed && isRunning && currentQuery && (
                    <div className="px-2 mb-4">
                        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Running</div>
                        <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                            <div className="flex items-center gap-2 text-sm text-blue-400">
                                <Loader2 size={14} className="shrink-0 animate-spin" />
                                <span className="truncate">{currentQuery}</span>
                            </div>
                        </div>
                    </div>
                )}

                {!isCollapsed && runHistory.length > 0 && (
                    <>
                        <div className="flex items-center justify-between px-2 mb-2">
                            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">History</div>
                            {runHistory.length > 0 && (
                                showClearConfirm ? (
                                    <div className="flex items-center gap-1">
                                        <span className="text-xs text-gray-500">Clear all?</span>
                                        <button
                                            onClick={() => { onClearHistory?.(); setShowClearConfirm(false); }}
                                            className="text-xs text-red-400 hover:text-red-300 px-1"
                                        >
                                            Yes
                                        </button>
                                        <button
                                            onClick={() => setShowClearConfirm(false)}
                                            className="text-xs text-gray-500 hover:text-gray-300 px-1"
                                        >
                                            No
                                        </button>
                                    </div>
                                ) : (
                                    <button
                                        onClick={() => setShowClearConfirm(true)}
                                        className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
                                        title="Clear all history"
                                    >
                                        Clear
                                    </button>
                                )
                            )}
                        </div>
                        <div className="space-y-1">
                            {runHistory.slice(0, 30).map(run => (
                                <div
                                    key={run.run_id}
                                    className="relative group/item"
                                    onMouseEnter={() => setHoveredRunId(run.run_id)}
                                    onMouseLeave={() => setHoveredRunId(null)}
                                >
                                    <button
                                        onClick={() => loadRun(run.run_id)}
                                        className="w-full text-left p-2 rounded-lg text-sm flex flex-col gap-0.5 text-gray-400 hover:bg-gray-900 hover:text-gray-200 transition-colors pr-8"
                                    >
                                        <div className="flex items-center gap-2">
                                            <Clock size={14} className="shrink-0" />
                                            <span className="truncate flex-1">{run.query || "Untitled Run"}</span>
                                        </div>
                                        <div className="text-xs text-gray-600 ml-6">
                                            {formatRelativeTime(run.created_at)}
                                        </div>
                                    </button>
                                    {hoveredRunId === run.run_id && onDeleteRun && (
                                        <button
                                            onClick={(e) => { e.stopPropagation(); onDeleteRun(run.run_id); }}
                                            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-600 hover:text-red-400 transition-colors"
                                            title="Delete run"
                                        >
                                            <X size={14} />
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>
                    </>
                )}
                
                {isCollapsed && (
                    <div className="flex flex-col items-center gap-4 mt-4">
                        <button onClick={() => setIsCollapsed(false)} className="p-2 text-gray-500 hover:text-white rounded-lg hover:bg-gray-800" title="Expand History">
                            <Clock size={20} />
                        </button>
                    </div>
                )}
            </div>

            {/* User / Settings */}
            <div className="p-3 border-t border-gray-800">
                <button 
                    onClick={onOpenSettings}
                    className={`flex items-center gap-2 text-gray-400 hover:text-white text-sm transition-colors w-full p-2 rounded-lg hover:bg-gray-900 ${isCollapsed ? 'justify-center' : ''}`}
                    title="Settings"
                >
                    <Settings size={18} /> 
                    {!isCollapsed && <span>Settings</span>}
                </button>
            </div>
        </div>
    );
};

export default HistorySidebar;
