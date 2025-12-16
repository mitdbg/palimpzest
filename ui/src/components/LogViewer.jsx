import React, { useState } from 'react';

const LogViewer = ({ logs, onClose }) => {
    const [height, setHeight] = useState(192); // 12rem = 192px
    const [isResizing, setIsResizing] = useState(false);

    const startResizing = (e) => {
        setIsResizing(true);
    };

    const stopResizing = () => {
        setIsResizing(false);
    };

    const resize = (e) => {
        if (isResizing) {
            const newHeight = window.innerHeight - e.clientY;
            if (newHeight > 100 && newHeight < 600) {
                setHeight(newHeight);
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
            className="fixed bottom-0 left-0 right-0 bg-gray-900 border-t border-gray-700 flex flex-col z-50 shadow-2xl"
            style={{ height }}
        >
            {/* Resize Handle */}
            <div 
                className="absolute top-0 left-0 right-0 h-1 cursor-row-resize hover:bg-blue-500/50 z-10 transition-colors"
                onMouseDown={startResizing}
            />
            
            <div className="p-2 bg-gray-800 text-xs font-bold text-gray-400 flex justify-between select-none">
                <span>Execution Trace & Logs</span>
                <button onClick={onClose} className="hover:text-white">Close</button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 font-mono text-xs space-y-1">
                {logs.length === 0 && <div className="text-gray-600 italic">No logs available</div>}
                {logs.map((log, i) => {
                    let content = null;
                    let colorClass = 'text-gray-300';
                    
                    if (log.event_type === 'stderr') {
                        colorClass = 'text-red-400';
                        content = log.data?.message || JSON.stringify(log.data);
                    } else if (log.event_type === 'stdout') {
                        content = log.data?.message || JSON.stringify(log.data);
                    } else if (log.event_type === 'search_step') {
                        colorClass = 'text-blue-300';
                        const nodeId = log.data?.node_id;
                        const shortId = typeof nodeId === 'string' ? `${nodeId.substring(0, 12)}...` : String(nodeId);
                        const decision = log.data?.decision || (log.data?.admit ? 'admit' : 'reject');
                        const score = (log.data?.score ?? log.data?.rank_score);
                        const summary = log.data?.summary;
                        const reason = log.data?.reason;
                        const model = log.data?.model;
                        const latency = log.data?.stats?.latency_s;
                        const cost = log.data?.stats?.cost_usd;
                        content = `Decision ${String(decision).toUpperCase()}: ${shortId}`;
                        if (typeof score === 'number') content += ` score=${score.toFixed(3)}`;
                        if (summary) content += ` | ${summary}`;
                        if (reason) content += ` | ${reason}`;
                        if (model) content += ` | model=${model}`;
                        if (typeof latency === 'number') content += ` | ${latency.toFixed(2)}s`;
                        if (typeof cost === 'number') content += ` | $${cost.toFixed(4)}`;
                    } else if (log.event_type === 'node_evaluation') {
                        colorClass = 'text-purple-300';
                        const nodeId = log.data?.node_id;
                        const shortId = typeof nodeId === 'string' ? `${nodeId.substring(0, 12)}...` : String(nodeId);
                        const verdict = log.data?.is_relevant ? 'ADMIT' : 'REJECT';
                        const score = log.data?.score;
                        const summary = log.data?.metadata?.summary;
                        const reasoning = log.data?.reasoning;
                        content = `Eval ${verdict}: ${shortId}`;
                        if (typeof score === 'number') content += ` score=${score.toFixed(3)}`;
                        if (summary) content += ` | ${summary}`;
                        if (reasoning) content += ` | ${reasoning}`;
                    } else if (log.event_type === 'evidence_collected') {
                        colorClass = 'text-yellow-300';
                        const nodeId = log.data?.node_id;
                        const shortId = typeof nodeId === 'string' ? `${nodeId.substring(0, 12)}...` : String(nodeId);
                        const contentPreview = (log.data?.content || '').toString();
                        const reasoning = log.data?.reasoning;
                        content = `Evidence: ${shortId}`;
                        if (contentPreview) content += ` | ${contentPreview.substring(0, 80)}${contentPreview.length > 80 ? '...' : ''}`;
                        if (reasoning) content += ` | ${reasoning}`;
                    } else if (log.event_type === 'result') {
                        colorClass = 'text-green-400 font-bold';
                        content = `Result: ${log.data.answer}`;
                    } else if (log.event_type === 'query_start') {
                        colorClass = 'text-white font-bold border-b border-gray-700 pb-1 mb-1 block';
                        content = `Query Started: ${log.data.query}`;
                    } else if (log.event_type === 'query_end') {
                        colorClass = 'text-gray-500 italic border-t border-gray-700 pt-1 mt-1 block';
                        content = `Query Ended. Visited ${log.data.visited} nodes.`;
                    } else {
                        content = JSON.stringify(log.data);
                    }

                    return (
                        <div key={i} className={`${colorClass} break-words`}>
                            <span className="text-gray-600 mr-2 select-none">[{new Date(log.ts_ms || Date.now()).toLocaleTimeString()}]</span>
                            {content}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default LogViewer;
