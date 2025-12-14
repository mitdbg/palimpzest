import React, { useState } from 'react';
import { ChevronRight, FileText, Code } from 'lucide-react';

const NodeDetails = ({ node, onClose, devMode }) => {
    const [activeTab, setActiveTab] = useState('overview');

    if (!node) return null;

    return (
        <div className="fixed right-0 top-0 bottom-0 w-96 bg-gray-900 border-l border-gray-700 shadow-2xl z-50 flex flex-col animate-in slide-in-from-right duration-300">
            <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-800">
                <h2 className="font-bold text-lg">Node Details</h2>
                <button onClick={onClose} className="text-gray-400 hover:text-white">
                    <ChevronRight size={20} />
                </button>
            </div>

            {/* Tabs */}
            {devMode && (
                <div className="flex border-b border-gray-700">
                    <button 
                        onClick={() => setActiveTab('overview')}
                        className={`flex-1 py-2 text-xs font-medium flex items-center justify-center gap-2 ${activeTab === 'overview' ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        <FileText size={14} /> Overview
                    </button>
                    <button 
                        onClick={() => setActiveTab('debug')}
                        className={`flex-1 py-2 text-xs font-medium flex items-center justify-center gap-2 ${activeTab === 'debug' ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        <Code size={14} /> Debug
                    </button>
                </div>
            )}

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {activeTab === 'overview' ? (
                    <>
                        <div>
                            <div className="text-xs text-gray-500 uppercase font-bold mb-1">ID</div>
                            <div className="font-mono text-sm break-all bg-gray-800 p-2 rounded select-all">{node.id}</div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <div className="text-xs text-gray-500 uppercase font-bold mb-1">Type</div>
                                <div className="text-sm">{node.type}</div>
                            </div>
                            <div>
                                <div className="text-xs text-gray-500 uppercase font-bold mb-1">Level</div>
                                <div className="text-sm">{node.level}</div>
                            </div>
                        </div>

                        {node.score !== undefined && (
                            <div>
                                <div className="text-xs text-gray-500 uppercase font-bold mb-1">Score</div>
                                <div className="text-sm font-mono">
                                    {typeof node.score === 'number' ? node.score.toFixed(4) : 'N/A'}
                                </div>
                            </div>
                        )}

                        <div>
                            <div className="text-xs text-gray-500 uppercase font-bold mb-1">Summary</div>
                            <div className="text-sm bg-gray-800 p-3 rounded whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                                {node.summary || node.text || "No summary available."}
                            </div>
                        </div>
                        
                        {node.metadata && (
                            <div>
                                <div className="text-xs text-gray-500 uppercase font-bold mb-1">Metadata</div>
                                <pre className="text-xs bg-gray-800 p-3 rounded overflow-x-auto text-gray-300 whitespace-pre-wrap">
                                    {JSON.stringify(node.metadata, null, 2)}
                                </pre>
                            </div>
                        )}
                    </>
                ) : (
                    <>
                        {/* Debug Tab Content */}
                        <div>
                            <div className="text-xs text-gray-500 uppercase font-bold mb-1">Raw Data</div>
                            <pre className="text-xs bg-gray-950 p-2 rounded overflow-x-auto text-gray-400 font-mono">
                                {JSON.stringify(node, (key, value) => {
                                    if (key === 'x' || key === 'y' || key === 'vx' || key === 'vy' || key === 'fx' || key === 'fy' || key === 'index') return undefined;
                                    return value;
                                }, 2)}
                            </pre>
                        </div>

                        {/* Placeholder for Prompt/Response if we had it */}
                        {node.prompt && (
                            <div>
                                <div className="text-xs text-gray-500 uppercase font-bold mb-1 mt-4">Prompt</div>
                                <div className="text-xs bg-gray-950 p-2 rounded whitespace-pre-wrap font-mono text-green-400">
                                    {node.prompt}
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default NodeDetails;
