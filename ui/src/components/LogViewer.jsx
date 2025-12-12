import React from 'react';

const LogViewer = ({ logs, onClose }) => {
    return (
        <div className="h-48 bg-gray-900 border-t border-gray-700 flex flex-col">
            <div className="p-2 bg-gray-800 text-xs font-bold text-gray-400 flex justify-between">
                <span>System Logs</span>
                <button onClick={onClose} className="hover:text-white">Close</button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 font-mono text-xs space-y-1">
                {logs.length === 0 && <div className="text-gray-600 italic">No logs available</div>}
                {logs.map((log, i) => (
                    <div key={i} className={`${log.event_type === 'stderr' ? 'text-red-400' : 'text-gray-300'}`}>
                        <span className="text-gray-600 mr-2">[{new Date(log.timestamp * 1000).toLocaleTimeString()}]</span>
                        {log.data?.message || JSON.stringify(log.data)}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default LogViewer;
