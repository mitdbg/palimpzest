import React from 'react';
import { Play, Square } from 'lucide-react';

const Sidebar = ({
    runConfig, setRunConfig,
    resources,
    availableModels,
    runQuery, stopQuery, status,
    queries, selectedQueryId, setSelectedQueryId,
    metrics,
    currentAction,
    finalAnswer,
    evidence,
    queue,
    showFullGraph, setShowFullGraph, isLoadingGraph,
    fullGraphData
}) => {
    const nodeCount = fullGraphData?.nodes?.length ?? null;
    const edgeCount = fullGraphData?.links?.length ?? null;

    return (
        <div className="w-80 bg-gray-850 border-r border-gray-700 flex flex-col">
            <div className="p-4 border-b border-gray-700 space-y-3">
                {/* Index Selection */}
                <div>
                    <label className="block text-xs text-gray-400 mb-1">Index</label>
                    <select 
                        className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
                        value={runConfig.index}
                        onChange={(e) => setRunConfig({...runConfig, index: e.target.value})}
                    >
                        {resources.indices.length === 0 && <option value="">Loading indices...</option>}
                        {resources.indices.map(idx => (
                            <option key={idx} value={idx}>{idx.split('/').pop()}</option>
                        ))}
                    </select>

                    {/* Graph Stats */}
                    <div className="mt-2 text-xs text-gray-500">
                        {isLoadingGraph ? (
                            <span>Graph: loading…</span>
                        ) : nodeCount != null && edgeCount != null ? (
                            <span>Graph: {nodeCount.toLocaleString()} nodes · {edgeCount.toLocaleString()} edges</span>
                        ) : (
                            <span>Graph: not loaded</span>
                        )}
                    </div>

                    {/* Full Graph Toggle */}
                    <div className="mt-2 flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                            <input 
                                type="checkbox" 
                                id="showFullGraph"
                                checked={showFullGraph}
                                onChange={(e) => setShowFullGraph(e.target.checked)}
                                className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                            />
                            <label htmlFor="showFullGraph" className="text-xs text-gray-400 cursor-pointer select-none">
                                {isLoadingGraph && showFullGraph ? "Loading Graph..." : "Show Full Graph"}
                            </label>
                        </div>
                    </div>
                </div>

                {/* Model Selection */}
                <div>
                    <label className="block text-xs text-gray-400 mb-1">Orchestrator Model</label>
                    <select 
                        className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
                        value={runConfig.model}
                        onChange={(e) => setRunConfig({...runConfig, model: e.target.value})}
                    >
                        {availableModels.map(m => (
                            <option key={m} value={m}>{m.split('/').pop()}</option>
                        ))}
                    </select>
                </div>

                {/* Granular Model Config */}
                <div className="pt-2 border-t border-gray-700">
                    <div className="text-xs font-semibold text-gray-400 mb-2">Advanced Model Config</div>
                    
                    <div className="mb-2">
                        <label className="block text-xs text-gray-500 mb-1">Ranking Model</label>
                        <select 
                            className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
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

                    <div className="mb-2">
                        <label className="block text-xs text-gray-500 mb-1">Admittance Model</label>
                        <select 
                            className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
                            value={runConfig.admittance_model}
                            onChange={(e) => setRunConfig({...runConfig, admittance_model: e.target.value})}
                        >
                            <option value="">Default (Orchestrator)</option>
                            {availableModels.map(m => (
                                <option key={`adm-${m}`} value={m}>{m.split('/').pop()}</option>
                            ))}
                        </select>
                    </div>

                    <div className="mb-2">
                        <label className="block text-xs text-gray-500 mb-1">Termination Model</label>
                        <select 
                            className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
                            value={runConfig.termination_model}
                            onChange={(e) => setRunConfig({...runConfig, termination_model: e.target.value})}
                        >
                            <option value="">Default (Orchestrator)</option>
                            {availableModels.map(m => (
                                <option key={`term-${m}`} value={m}>{m.split('/').pop()}</option>
                            ))}
                        </select>
                    </div>

                    <div className="mb-2">
                        <label className="block text-xs text-gray-500 mb-1">Entry Points (K)</label>
                        <input 
                            type="number"
                            className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
                            value={runConfig.entry_points}
                            onChange={(e) => setRunConfig({...runConfig, entry_points: parseInt(e.target.value) || 5})}
                            min="1"
                            max="50"
                        />
                    </div>
                </div>

                {/* Input Type Toggle */}
                <div className="flex gap-2 text-xs">
                    <button 
                        className={`flex-1 py-1 rounded ${runConfig.inputType === 'query' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                        onClick={() => setRunConfig({...runConfig, inputType: 'query'})}
                    >
                        Single Query
                    </button>
                    <button 
                        className={`flex-1 py-1 rounded ${runConfig.inputType === 'workload' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                        onClick={() => setRunConfig({...runConfig, inputType: 'workload'})}
                    >
                        Workload File
                    </button>
                </div>

                {/* Input Field */}
                {runConfig.inputType === 'query' ? (
                    <textarea 
                    className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm focus:border-blue-500 outline-none"
                    rows="3"
                    value={runConfig.query}
                    onChange={(e) => setRunConfig({...runConfig, query: e.target.value})}
                    placeholder="Enter your query..."
                    />
                ) : (
                    <select 
                        className="w-full bg-gray-900 border border-gray-700 rounded p-1 text-xs text-gray-300"
                        value={runConfig.workload}
                        onChange={(e) => setRunConfig({...runConfig, workload: e.target.value})}
                    >
                        {resources.workloads.map(wl => (
                            <option key={wl} value={wl}>{wl.split('/').pop()}</option>
                        ))}
                    </select>
                )}

                {/* Controls */}
                <div className="flex gap-2">
                <button 
                    onClick={runQuery}
                    disabled={status.running}
                    className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white py-1 px-3 rounded flex items-center justify-center gap-2 text-sm"
                >
                    <Play size={14} /> Run
                </button>
                <button 
                    onClick={stopQuery}
                    disabled={!status.running}
                    className="bg-red-900 hover:bg-red-800 disabled:opacity-50 text-red-200 py-1 px-3 rounded flex items-center justify-center"
                >
                    <Square size={14} />
                </button>
                </div>
            </div>
            
            {/* Query List (New) */}
            {queries.length > 1 && (
                <div className="p-4 border-b border-gray-700 max-h-40 overflow-y-auto">
                    <div className="text-xs text-gray-400 mb-1">Queries ({queries.length})</div>
                    <div className="space-y-1">
                        {queries.map(q => (
                            <div 
                                key={q.id} 
                                onClick={() => setSelectedQueryId(q.id)}
                                className={`text-xs p-1 rounded cursor-pointer truncate ${selectedQueryId === q.id ? 'bg-blue-900 text-blue-200' : 'hover:bg-gray-800 text-gray-400'}`}
                            >
                                {q.text}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Metrics Panel */}
            <div className="p-4 border-b border-gray-700 grid grid-cols-2 gap-2">
                <div className="bg-gray-800 p-2 rounded">
                    <div className="text-xs text-gray-400">Cost</div>
                    <div className="font-mono font-bold">${metrics.cost.toFixed(4)}</div>
                </div>
                <div className="bg-gray-800 p-2 rounded">
                    <div className="text-xs text-gray-400">Calls</div>
                    <div className="font-mono font-bold">{metrics.calls}</div>
                </div>
                <div className="bg-gray-800 p-2 rounded">
                    <div className="text-xs text-gray-400">Tokens</div>
                    <div className="font-mono font-bold">{(metrics.tokens / 1000).toFixed(1)}k</div>
                </div>
                <div className="bg-gray-800 p-2 rounded">
                    <div className="text-xs text-gray-400">Shortcuts</div>
                    <div className="font-mono font-bold">{metrics.shortcuts}</div>
                </div>
            </div>

            {/* Current Action */}
            <div className="p-4 border-b border-gray-700">
                <div className="text-xs text-gray-400 mb-1">Current Action</div>
                <div className="text-xs font-mono text-green-400 break-all">
                    {currentAction}
                </div>
            </div>

            {/* Sufficiency (Placeholder) */}
            <div className="p-4 border-b border-gray-700">
                <div className="text-xs text-gray-400 mb-1">Sufficiency</div>
                <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Status:</span>
                    <span className="font-bold text-blue-400">Exploring</span>
                </div>
            </div>

            {/* Final Answer */}
            {finalAnswer && (
                <div className="p-4 border-b border-gray-700 bg-blue-900/20">
                    <div className="text-xs text-blue-400 font-bold mb-1 uppercase">Final Answer</div>
                    <div className="text-xs text-gray-200 whitespace-pre-wrap max-h-60 overflow-y-auto">
                        {finalAnswer}
                    </div>
                </div>
            )}

            {/* Evidence List */}
            <div className="flex-1 overflow-y-auto p-4">
                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Evidence ({evidence.length})</h3>
                <div className="space-y-1 mb-4 max-h-60 overflow-y-auto">
                    {evidence.map((e, i) => (
                        <div key={i} className="text-xs bg-gray-800 p-2 rounded border-l-2 border-yellow-500 mb-1">
                            <div className="font-bold text-yellow-500">{(e.node_id || e.id || "").substring(0, 8)}</div>
                            <div className="text-gray-400 truncate">{e.summary}</div>
                        </div>
                    ))}
                </div>

                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Priority Queue</h3>
                <div className="space-y-1 mb-4 max-h-40 overflow-y-auto">
                    {queue.length === 0 && <div className="text-xs text-gray-600 italic">Empty</div>}
                    {queue.map((q, i) => (
                        <div key={i} className="text-xs bg-gray-800 p-1 rounded flex justify-between items-center mb-1">
                            <span className="font-mono truncate flex-1 mr-2" title={q.summary || q.id}>
                                {q.summary ? q.summary.substring(0, 30) + (q.summary.length > 30 ? '...' : '') : q.id.substring(0, 8)}
                            </span>
                            <span className="text-blue-400 whitespace-nowrap">
                                {typeof q.score === 'number' ? q.score.toFixed(2) : 'N/A'}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
