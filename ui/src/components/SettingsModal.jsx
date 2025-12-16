import React, { useState, useRef, useCallback, useEffect } from 'react';
import { X, ChevronDown, ChevronRight, GripVertical } from 'lucide-react';
import TypeColorEditor from './TypeColorEditor';

// Collapsible Section Component
const Section = ({ title, children, defaultOpen = false }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="border border-gray-700 rounded-lg overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-3 bg-gray-800/50 hover:bg-gray-800 text-left transition-colors"
            >
                <span className="text-xs font-medium text-gray-300 uppercase tracking-wider">{title}</span>
                {isOpen ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
            </button>
            {isOpen && <div className="p-3 space-y-3 bg-gray-900/50">{children}</div>}
        </div>
    );
};

// Compact slider with label
const Slider = ({ label, value, onChange, min, max, step, suffix = '', allowInfinite = false }) => {
    const isInfinite = allowInfinite && value == null;
    const rangeValue = isInfinite ? max : value;
    const displayValue = isInfinite ? '∞' : value;

    return (
        <div className="flex items-center gap-2">
            <span className="text-[11px] text-gray-500 w-24 shrink-0">{label}</span>
            <input
                type="range"
                className="flex-1 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                value={rangeValue}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                min={min}
                max={max}
                step={step}
            />
            {allowInfinite && (
                <button
                    type="button"
                    className={
                        "text-[10px] px-1.5 py-0.5 rounded border " +
                        (isInfinite
                            ? "border-blue-500/70 text-blue-300"
                            : "border-gray-700 text-gray-400 hover:text-gray-300")
                    }
                    onClick={() => onChange(isInfinite ? max : null)}
                    title={isInfinite ? 'Set finite charge range' : 'Set infinite charge range'}
                >
                    ∞
                </button>
            )}
            <span className="text-[11px] text-gray-400 w-12 text-right font-mono">{displayValue}{suffix}</span>
        </div>
    );
};

// Toggle switch
const Toggle = ({ label, checked, onChange, hint }) => (
    <div className="flex items-center justify-between">
        <div>
            <div className="text-xs text-gray-300">{label}</div>
            {hint && <div className="text-[10px] text-gray-500">{hint}</div>}
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" checked={checked} onChange={(e) => onChange(e.target.checked)} />
            <div className="w-9 h-5 bg-gray-700 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
        </label>
    </div>
);

const FilterEditor = ({ filters, onChange, graphData }) => {
    const addFilter = () => {
        onChange([...(filters || []), { field: '', operator: '==', value: '' }]);
    };

    const removeFilter = (index) => {
        const newFilters = [...(filters || [])];
        newFilters.splice(index, 1);
        onChange(newFilters);
    };

    const updateFilter = (index, key, val) => {
        const newFilters = [...(filters || [])];
        newFilters[index] = { ...newFilters[index], [key]: val };
        onChange(newFilters);
    };

    // Derive available fields from graph data
    const availableFields = React.useMemo(() => {
        if (!graphData?.nodes) return [];
        const fields = new Set();
        // Sample first 100 nodes to avoid performance hit
        graphData.nodes.slice(0, 100).forEach(node => {
            Object.keys(node).forEach(k => {
                if (!['id', 'x', 'y', 'vx', 'vy', 'index', 'type', 'pz_type'].includes(k)) {
                    fields.add(k);
                }
            });
        });
        return Array.from(fields).sort();
    }, [graphData]);

    return (
        <div className="space-y-2">
            <label className="block text-xs text-gray-400 mb-1">Manual Filters</label>
            {(filters || []).map((f, i) => (
                <div key={i} className="flex gap-2 items-center">
                    <input 
                        type="text" 
                        placeholder="Field" 
                        list={`fields-${i}`}
                        className="w-1/3 bg-gray-800 border border-gray-700 rounded p-1 text-xs text-white"
                        value={f.field}
                        onChange={(e) => updateFilter(i, 'field', e.target.value)}
                    />
                    <datalist id={`fields-${i}`}>
                        {availableFields.map(field => (
                            <option key={field} value={field} />
                        ))}
                    </datalist>
                    <select 
                        className="w-1/4 bg-gray-800 border border-gray-700 rounded p-1 text-xs text-white"
                        value={f.operator}
                        onChange={(e) => updateFilter(i, 'operator', e.target.value)}
                    >
                        <option value="==">==</option>
                        <option value="!=">!=</option>
                        <option value=">">&gt;</option>
                        <option value="<">&lt;</option>
                        <option value=">=">&gt;=</option>
                        <option value="<=">&lt;=</option>
                        <option value="contains">contains</option>
                    </select>
                    <input 
                        type="text" 
                        placeholder="Value" 
                        className="flex-1 bg-gray-800 border border-gray-700 rounded p-1 text-xs text-white"
                        value={f.value}
                        onChange={(e) => updateFilter(i, 'value', e.target.value)}
                    />
                    <button onClick={() => removeFilter(i)} className="text-red-400 hover:text-red-300">
                        <X size={14} />
                    </button>
                </div>
            ))}
            <button onClick={addFilter} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
                + Add Filter
            </button>
        </div>
    );
};

const SettingsModal = ({ 
    isOpen, 
    onClose, 
    runConfig, 
    setRunConfig, 
    resources, 
    availableModels,
    devMode,
    setDevMode,
    vizConfig,
    setVizConfig,
    onResetVizConfig,
    graphData
}) => {
    const [activeTab, setActiveTab] = useState('run');
    const [panelWidth, setPanelWidth] = useState(384); // 384px = w-96
    const isDragging = useRef(false);
    const dragStartX = useRef(0);
    const dragStartWidth = useRef(0);

    const handleMouseDown = useCallback((e) => {
        isDragging.current = true;
        dragStartX.current = e.clientX;
        dragStartWidth.current = panelWidth;
        document.body.style.cursor = 'ew-resize';
        document.body.style.userSelect = 'none';
    }, [panelWidth]);

    useEffect(() => {
        const handleMouseMove = (e) => {
            if (!isDragging.current) return;
            const delta = dragStartX.current - e.clientX;
            const newWidth = Math.min(800, Math.max(320, dragStartWidth.current + delta));
            setPanelWidth(newWidth);
        };

        const handleMouseUp = () => {
            isDragging.current = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, []);

    if (!isOpen) return null;

    return (
        <div 
            className={`fixed top-0 right-0 h-full bg-gray-900 border-l border-gray-700 shadow-2xl z-50 transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
            style={{ width: panelWidth }}
        >
            {/* Resize handle */}
            <div 
                className="absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-blue-500/30 transition-colors group flex items-center justify-center"
                onMouseDown={handleMouseDown}
            >
                <div className="w-0.5 h-12 bg-gray-600 group-hover:bg-blue-400 rounded transition-colors" />
            </div>
            <div className="flex flex-col h-full">
                <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-900">
                    <h2 className="text-lg font-semibold text-white">Configuration</h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">
                        <X size={20} />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-gray-700 shrink-0">
                    <button
                        onClick={() => setActiveTab('run')}
                        className={`flex-1 py-2 text-xs font-medium ${activeTab === 'run' ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        Run
                    </button>
                    <button
                        onClick={() => setActiveTab('viz')}
                        className={`flex-1 py-2 text-xs font-medium ${activeTab === 'viz' ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        Layout
                    </button>
                    <button
                        onClick={() => setActiveTab('colors')}
                        className={`flex-1 py-2 text-xs font-medium ${activeTab === 'colors' ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        Colors
                    </button>
                </div>
                
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {activeTab === 'run' ? (
                    <>
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
                        <p className="text-xs text-gray-500 mt-1">Select a graph from CURRENT_WORKSTREAM/exports or data/</p>
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
                            <label className="block text-xs text-gray-400 mb-1">Custom Admittance Instructions</label>
                            <textarea 
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none min-h-[80px]"
                                placeholder="Override the default admittance criteria (e.g., 'Accept any node related to pricing...')"
                                value={runConfig.admittance_instructions || ""}
                                onChange={(e) => setRunConfig({...runConfig, admittance_instructions: e.target.value})}
                            />
                            <p className="text-[10px] text-gray-500 mt-1">Leave empty to use auto-generated criteria based on your query.</p>
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

                        <Toggle 
                            label="Traverse through non-matches" 
                            checked={!!runConfig.expand_filtered_nodes}
                            onChange={(v) => setRunConfig({...runConfig, expand_filtered_nodes: v})}
                            hint="Enable this to find target nodes even if they are connected via nodes that don't match the filter"
                        />

                        <FilterEditor 
                            filters={runConfig.filters} 
                            onChange={(newFilters) => setRunConfig({...runConfig, filters: newFilters})} 
                            graphData={graphData}
                        />
                    </div>
                    </>
                    ) : activeTab === 'viz' ? (
                    <>
                    <div className="space-y-3">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Layout & Style</h3>
                            <button 
                                onClick={onResetVizConfig}
                                className="text-[10px] text-blue-400 hover:text-blue-300"
                            >
                                Reset All
                            </button>
                        </div>

                        {/* Quick toggles at top */}
                        <div className="bg-gray-800/50 rounded-lg p-3 space-y-2">
                            <Toggle 
                                label="Run Layout" 
                                checked={!!vizConfig?.runLayout}
                                onChange={(v) => setVizConfig(prev => ({ ...prev, runLayout: v }))}
                                hint="Enable D3 force simulation"
                            />
                            <Toggle 
                                label="Show Labels" 
                                checked={!!vizConfig?.showLabels}
                                onChange={(v) => setVizConfig(prev => ({ ...prev, showLabels: v }))}
                            />
                            <Toggle 
                                label="Hover Highlight" 
                                checked={!!vizConfig?.hoverHighlight}
                                onChange={(v) => setVizConfig(prev => ({ ...prev, hoverHighlight: v }))}
                                hint="Dim non-neighbors on hover"
                            />
                        </div>

                        {/* Force Layout Section */}
                        <Section title="Force Layout" defaultOpen={false}>
                            <Slider label="Repulsion" value={vizConfig?.chargeStrength ?? -800} onChange={(v) => setVizConfig(prev => ({ ...prev, chargeStrength: v }))} min={-2000} max={0} step={10} />
                            <Slider label="Charge Range" value={vizConfig?.chargeDistanceMax ?? null} onChange={(v) => setVizConfig(prev => ({ ...prev, chargeDistanceMax: v }))} min={20} max={800} step={10} allowInfinite />
                            <Slider label="Link Distance" value={vizConfig?.linkDistance ?? 60} onChange={(v) => setVizConfig(prev => ({ ...prev, linkDistance: v }))} min={10} max={300} step={5} />
                            <Slider label="Link Strength" value={vizConfig?.linkStrength ?? 0.3} onChange={(v) => setVizConfig(prev => ({ ...prev, linkStrength: v }))} min={0} max={2} step={0.05} />
                            <Slider label="Centering" value={vizConfig?.centerStrength ?? 0.02} onChange={(v) => setVizConfig(prev => ({ ...prev, centerStrength: v }))} min={0} max={0.2} step={0.005} />
                            <Slider label="Collision" value={vizConfig?.collisionRadius ?? 2} onChange={(v) => setVizConfig(prev => ({ ...prev, collisionRadius: v }))} min={0} max={20} step={0.5} />
                            <Slider label="Friction" value={vizConfig?.d3VelocityDecay ?? 0.3} onChange={(v) => setVizConfig(prev => ({ ...prev, d3VelocityDecay: v }))} min={0} max={1} step={0.02} />
                        </Section>

                        {/* Node Styling Section */}
                        <Section title="Nodes" defaultOpen={false}>
                            <Slider label="Size Scale" value={vizConfig?.nodeRadiusScale ?? 1.0} onChange={(v) => setVizConfig(prev => ({ ...prev, nodeRadiusScale: v }))} min={0.3} max={5} step={0.1} suffix="x" />
                            <Slider label="Base Size" value={vizConfig?.nodeBaseSize ?? 3} onChange={(v) => setVizConfig(prev => ({ ...prev, nodeBaseSize: v }))} min={1} max={10} step={0.5} />
                            <Slider label="Opacity" value={vizConfig?.nodeOpacity ?? 1.0} onChange={(v) => setVizConfig(prev => ({ ...prev, nodeOpacity: v }))} min={0.1} max={1} step={0.05} />
                            <div className="flex items-center gap-2 mt-2">
                                <span className="text-[11px] text-gray-500 w-24 shrink-0">Border</span>
                                <select 
                                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                                    value={vizConfig?.nodeBorder ?? 'none'}
                                    onChange={(e) => setVizConfig(prev => ({ ...prev, nodeBorder: e.target.value }))}
                                >
                                    <option value="none">None</option>
                                    <option value="thin">Thin</option>
                                    <option value="medium">Medium</option>
                                    <option value="thick">Thick</option>
                                </select>
                            </div>
                        </Section>

                        {/* Edge Styling Section */}
                        <Section title="Edges" defaultOpen={false}>
                            <Slider label="Width" value={vizConfig?.edgeWidth ?? 0.7} onChange={(v) => setVizConfig(prev => ({ ...prev, edgeWidth: v }))} min={0.1} max={4} step={0.1} />
                            <Slider label="Opacity" value={vizConfig?.edgeOpacity ?? 0.08} onChange={(v) => setVizConfig(prev => ({ ...prev, edgeOpacity: v }))} min={0} max={1} step={0.02} />
                            <Slider label="Dim Opacity" value={vizConfig?.dimOpacity ?? 0.02} onChange={(v) => setVizConfig(prev => ({ ...prev, dimOpacity: v }))} min={0} max={0.3} step={0.01} />
                            <Slider label="Path Opacity" value={vizConfig?.pathOpacity ?? 0.9} onChange={(v) => setVizConfig(prev => ({ ...prev, pathOpacity: v }))} min={0.2} max={1} step={0.05} />
                            <Slider label="Arrow Size" value={vizConfig?.arrowLength ?? 0} onChange={(v) => setVizConfig(prev => ({ ...prev, arrowLength: v }))} min={0} max={10} step={0.5} />
                            <Slider label="Curvature" value={vizConfig?.edgeCurvature ?? 0} onChange={(v) => setVizConfig(prev => ({ ...prev, edgeCurvature: v }))} min={0} max={0.5} step={0.05} />
                            <div className="flex items-center gap-2 mt-2">
                                <span className="text-[11px] text-gray-500 w-24 shrink-0">Style</span>
                                <select 
                                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                                    value={vizConfig?.edgeStyle ?? 'solid'}
                                    onChange={(e) => setVizConfig(prev => ({ ...prev, edgeStyle: e.target.value }))}
                                >
                                    <option value="solid">Solid</option>
                                    <option value="dashed">Dashed</option>
                                    <option value="dotted">Dotted</option>
                                </select>
                            </div>
                        </Section>

                        {/* Labels Section */}
                        <Section title="Labels" defaultOpen={false}>
                            <div className="flex items-center gap-2">
                                <span className="text-[11px] text-gray-500 w-24 shrink-0">Policy</span>
                                <select
                                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                                    value={vizConfig?.labelPolicy ?? 'important'}
                                    onChange={(e) => setVizConfig(prev => ({ ...prev, labelPolicy: e.target.value }))}
                                >
                                    <option value="hover-only">Hover only</option>
                                    <option value="important">Important nodes</option>
                                    <option value="all">All (zoom-gated)</option>
                                </select>
                            </div>
                            <Slider label="Font Size" value={vizConfig?.labelFontSize ?? 10} onChange={(v) => setVizConfig(prev => ({ ...prev, labelFontSize: v }))} min={6} max={16} step={1} suffix="px" />
                            <Slider label="Max Length" value={vizConfig?.labelMaxLength ?? 20} onChange={(v) => setVizConfig(prev => ({ ...prev, labelMaxLength: v }))} min={5} max={50} step={5} />
                        </Section>

                        {/* Performance Section */}
                        <Section title="Performance" defaultOpen={false}>
                            <Toggle
                                label="Perf HUD"
                                checked={!!vizConfig?.perfHud}
                                onChange={(v) => setVizConfig(prev => ({ ...prev, perfHud: v }))}
                                hint="Show FPS + frame/tick timing"
                            />
                            <div className="flex items-center gap-2">
                                <span className="text-[11px] text-gray-500 w-24 shrink-0">Max Edges</span>
                                <input
                                    type="number"
                                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                                    value={vizConfig?.maxEdges ?? 10000}
                                    onChange={(e) => setVizConfig(prev => ({ ...prev, maxEdges: parseInt(e.target.value) || 0 }))}
                                    min="0"
                                    max="200000"
                                />
                            </div>
                            <div className="text-[10px] text-gray-500">0 = render all (slow for dense graphs)</div>
                        </Section>
                    </div>
                    </>
                    ) : (
                    <>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Type Colors</h3>
                            <button 
                                onClick={onResetVizConfig}
                                className="text-xs text-blue-400 hover:text-blue-300 underline"
                            >
                                Reset All
                            </button>
                        </div>
                        <TypeColorEditor 
                            graphData={graphData}
                            vizConfig={vizConfig}
                            setVizConfig={setVizConfig}
                        />
                    </div>
                    </>
                    )}
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
