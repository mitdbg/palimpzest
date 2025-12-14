import React, { useState } from 'react';
import { MessageSquare, Clock, Settings, Plus, ChevronLeft, ChevronRight, PanelLeft } from 'lucide-react';

const HistorySidebar = ({ 
    queries, 
    selectedQueryId, 
    setSelectedQueryId, 
    onNewChat, 
    onOpenSettings,
    runHistory,
    loadRun
}) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [width, setWidth] = useState(256);
    const [isResizing, setIsResizing] = useState(false);

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

                {!isCollapsed && runHistory.length > 0 && (
                    <>
                        <div className="text-xs font-semibold text-gray-500 px-2 mb-2 uppercase tracking-wider">History</div>
                        <div className="space-y-1">
                            {runHistory.slice(0, 20).map(run => (
                                <button
                                    key={run.run_id}
                                    onClick={() => loadRun(run.run_id)}
                                    className="w-full text-left p-2 rounded-lg text-sm truncate flex items-center gap-2 text-gray-400 hover:bg-gray-900 hover:text-gray-200 transition-colors"
                                >
                                    <Clock size={14} className="shrink-0" />
                                    <span className="truncate">{run.query || "Untitled Run"}</span>
                                </button>
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
