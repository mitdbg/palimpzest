import React, { useMemo, useState } from 'react';
import { Palette, ChevronDown, ChevronRight, Eye, EyeOff } from 'lucide-react';

// Available node shapes
const NODE_SHAPES = ['circle', 'square', 'diamond', 'triangle', 'hexagon'];

// Default settings for different node types
const DEFAULT_NODE_SETTINGS = {
    document: { color: '#6366f1', size: 1.5, opacity: 1.0, shape: 'square' },
    cms_block: { color: '#8b5cf6', size: 1.0, opacity: 1.0, shape: 'circle' },
    block: { color: '#8b5cf6', size: 1.0, opacity: 1.0, shape: 'circle' },
    chunk: { color: '#a78bfa', size: 0.8, opacity: 1.0, shape: 'circle' },
    entity: { color: '#14b8a6', size: 1.0, opacity: 1.0, shape: 'diamond' },
    concept: { color: '#06b6d4', size: 1.0, opacity: 1.0, shape: 'hexagon' },
    person: { color: '#f472b6', size: 1.0, opacity: 1.0, shape: 'triangle' },
    organization: { color: '#fb923c', size: 1.0, opacity: 1.0, shape: 'square' },
    location: { color: '#84cc16', size: 1.0, opacity: 1.0, shape: 'diamond' },
    static: { color: '#60a5fa', size: 1.0, opacity: 1.0, shape: 'circle' },
    candidate: { color: '#94a3b8', size: 1.0, opacity: 1.0, shape: 'circle' },
};

// Default settings for different edge types
// strength: 0 = very weak (nodes drift apart), 1 = strong (nodes stay close)
const DEFAULT_EDGE_SETTINGS = {
    'hierarchy:child': { color: '#6b7280', width: 1.0, opacity: 0.3, style: 'solid', strength: 0.7, distance: 1.0 },
    hierarchy: { color: '#475569', width: 1.0, opacity: 0.3, style: 'solid', strength: 0.7, distance: 1.0 },
    child_of: { color: '#475569', width: 1.0, opacity: 0.3, style: 'solid', strength: 0.7, distance: 1.0 },
    parent_of: { color: '#475569', width: 1.0, opacity: 0.3, style: 'solid', strength: 0.7, distance: 1.0 },
    reference: { color: '#22c55e', width: 1.5, opacity: 0.6, style: 'solid', strength: 0.3, distance: 1.0 },
    'domain:references': { color: '#10b981', width: 1.5, opacity: 0.6, style: 'solid', strength: 0.3, distance: 1.0 },
    mentions: { color: '#f97316', width: 1.5, opacity: 0.6, style: 'solid', strength: 0.2, distance: 1.0 },
    mentions_jira: { color: '#f97316', width: 1.5, opacity: 0.6, style: 'solid', strength: 0.2, distance: 1.0 },
    jira_mention: { color: '#f97316', width: 1.5, opacity: 0.6, style: 'solid', strength: 0.2, distance: 1.0 },
    related_to: { color: '#3b82f6', width: 1.0, opacity: 0.5, style: 'dashed', strength: 0.1, distance: 1.0 },
    similar_to: { color: '#8b5cf6', width: 1.0, opacity: 0.5, style: 'dashed', strength: 0.1, distance: 1.0 },
};

// Default fallbacks
const DEFAULT_NODE_FALLBACK = { color: '#64748b', size: 1.0, opacity: 1.0, shape: 'circle', visible: true };
const DEFAULT_EDGE_FALLBACK = { color: '#334155', width: 1.0, opacity: 0.2, style: 'solid', strength: 0.3, distance: 1.0, visible: true };

// Nice preset palette
const COLOR_PRESETS = [
    '#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16',
    '#22c55e', '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9',
    '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
    '#ec4899', '#f43f5e', '#64748b', '#475569', '#334155',
];

// Export defaults for use in GraphVisualization
export { DEFAULT_NODE_SETTINGS, DEFAULT_EDGE_SETTINGS, DEFAULT_NODE_FALLBACK, DEFAULT_EDGE_FALLBACK, NODE_SHAPES };

// Collapsible Section
const Section = ({ title, icon: Icon, defaultOpen = true, count, children }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="border border-gray-700 rounded-lg overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-750 text-left"
            >
                {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {Icon && <Icon size={12} className="text-gray-400" />}
                <span className="text-xs font-medium text-gray-300 flex-1">{title}</span>
                {count !== undefined && (
                    <span className="text-[10px] text-gray-500 bg-gray-700 px-1.5 py-0.5 rounded">{count}</span>
                )}
            </button>
            {isOpen && <div className="p-2 space-y-2 bg-gray-850">{children}</div>}
        </div>
    );
};

// Single type row editor for nodes
const NodeTypeRow = ({ type, count, settings, defaults, onChange, onSolo }) => {
    const color = settings?.color ?? defaults?.color ?? DEFAULT_NODE_FALLBACK.color;
    const size = settings?.size ?? defaults?.size ?? DEFAULT_NODE_FALLBACK.size;
    const opacity = settings?.opacity ?? defaults?.opacity ?? DEFAULT_NODE_FALLBACK.opacity;
    const shape = settings?.shape ?? defaults?.shape ?? DEFAULT_NODE_FALLBACK.shape;
    const visible = settings?.visible ?? true;

    const update = (key, val) => onChange({ ...settings, [key]: val });

    const handleVisibilityClick = (e) => {
        if (e.altKey || e.metaKey) {
            // Option/Alt + click = solo this type
            onSolo?.();
        } else {
            update('visible', !visible);
        }
    };

    // Shape icons for the dropdown
    const shapeIcons = {
        circle: '●',
        square: '■',
        diamond: '◆',
        triangle: '▲',
        hexagon: '⬢',
    };

    return (
        <div className={`flex items-center gap-2 p-1.5 rounded ${visible ? 'bg-gray-800' : 'bg-gray-800/50 opacity-60'}`}>
            {/* Visibility toggle */}
            <button
                onClick={handleVisibilityClick}
                className="text-gray-400 hover:text-white"
                title={visible ? 'Hide (⌥+click to solo)' : 'Show (⌥+click to solo)'}
            >
                {visible ? <Eye size={12} /> : <EyeOff size={12} />}
            </button>

            {/* Type name */}
            <div className="flex-1 text-xs text-gray-300 truncate min-w-0" title={type}>
                {type}
                {count !== undefined && <span className="text-gray-500 ml-1">({count})</span>}
            </div>

            {/* Shape */}
            <select
                value={shape}
                onChange={(e) => update('shape', e.target.value)}
                className="text-[11px] bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-gray-300 flex-shrink-0 w-8 text-center"
                title={`Shape: ${shape}`}
            >
                {NODE_SHAPES.map(s => (
                    <option key={s} value={s}>{shapeIcons[s]}</option>
                ))}
            </select>

            {/* Color */}
            <input
                type="color"
                value={color}
                onChange={(e) => update('color', e.target.value)}
                className="w-5 h-5 rounded cursor-pointer border border-gray-600 bg-transparent flex-shrink-0"
                title="Color"
            />

            {/* Size */}
            <input
                type="range"
                min="0.2"
                max="3"
                step="0.1"
                value={size}
                onChange={(e) => update('size', parseFloat(e.target.value))}
                className="w-12 h-1 flex-shrink-0"
                title={`Size: ${size.toFixed(1)}x`}
            />

            {/* Opacity */}
            <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={opacity}
                onChange={(e) => update('opacity', parseFloat(e.target.value))}
                className="w-10 h-1 flex-shrink-0"
                title={`Opacity: ${(opacity * 100).toFixed(0)}%`}
            />
        </div>
    );
};

// Single type row editor for edges
const EdgeTypeRow = ({ type, count, settings, defaults, onChange, onSolo, baseLinkStrength }) => {
    const color = settings?.color ?? defaults?.color ?? DEFAULT_EDGE_FALLBACK.color;
    const width = settings?.width ?? defaults?.width ?? DEFAULT_EDGE_FALLBACK.width;
    const opacity = settings?.opacity ?? defaults?.opacity ?? DEFAULT_EDGE_FALLBACK.opacity;
    const style = settings?.style ?? defaults?.style ?? DEFAULT_EDGE_FALLBACK.style;
    const strength = settings?.strength ?? defaults?.strength ?? DEFAULT_EDGE_FALLBACK.strength;
    const distance = settings?.distance ?? defaults?.distance ?? DEFAULT_EDGE_FALLBACK.distance;
    const visible = settings?.visible ?? true;

    const effectiveStrength = (baseLinkStrength ?? 0.7) * strength;

    const update = (key, val) => onChange({ ...settings, [key]: val });

    const handleVisibilityClick = (e) => {
        if (e.altKey || e.metaKey) {
            // Option/Alt + click = solo this type
            onSolo?.();
        } else {
            update('visible', !visible);
        }
    };

    return (
        <div className={`flex items-center gap-2 p-1.5 rounded ${visible ? 'bg-gray-800' : 'bg-gray-800/50 opacity-60'}`}>
            {/* Visibility toggle */}
            <button
                onClick={handleVisibilityClick}
                className="text-gray-400 hover:text-white"
                title={visible ? 'Hide (⌥+click to solo)' : 'Show (⌥+click to solo)'}
            >
                {visible ? <Eye size={12} /> : <EyeOff size={12} />}
            </button>

            {/* Type name */}
            <div className="flex-1 text-xs text-gray-300 truncate min-w-0" title={type}>
                {type}
                {count !== undefined && <span className="text-gray-500 ml-1">({count})</span>}
            </div>

            {/* Color */}
            <input
                type="color"
                value={color}
                onChange={(e) => update('color', e.target.value)}
                className="w-5 h-5 rounded cursor-pointer border border-gray-600 bg-transparent flex-shrink-0"
                title="Color"
            />

            {/* Width */}
            <input
                type="number"
                min="0"
                max="10"
                step="0.1"
                value={width}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    update('width', Number.isFinite(v) ? v : 0);
                }}
                className="w-12 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white font-mono flex-shrink-0"
                title={`Width: ${width.toFixed(2)}`}
            />

            {/* Strength (link force) */}
            <input
                type="number"
                min="0"
                max="2"
                step="0.05"
                value={strength}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    update('strength', Number.isFinite(v) ? v : 0);
                }}
                className="w-12 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white font-mono flex-shrink-0"
                title={`Strength multiplier: ${strength.toFixed(3)} (effective: ${effectiveStrength.toFixed(3)})`}
            />

            {/* Distance multiplier (link length) */}
            <input
                type="number"
                min="0.1"
                max="10"
                step="0.1"
                value={distance}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    update('distance', Number.isFinite(v) ? v : 1.0);
                }}
                className="w-12 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white font-mono flex-shrink-0"
                title={`Distance multiplier: ${distance.toFixed(2)}x (applies to the global Link Distance)`}
            />

            {/* Opacity */}
            <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    update('opacity', Number.isFinite(v) ? v : 0);
                }}
                className="w-12 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white font-mono flex-shrink-0"
                title={`Opacity: ${(opacity * 100).toFixed(0)}%`}
            />

            {/* Style */}
            <select
                value={style}
                onChange={(e) => update('style', e.target.value)}
                className="text-[10px] bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-gray-300 flex-shrink-0"
                title="Line style"
            >
                <option value="solid">—</option>
                <option value="dashed">- -</option>
                <option value="dotted">···</option>
            </select>
        </div>
    );
};

// Color picker row for state colors
const ColorPicker = ({ color, onChange, label }) => (
    <div className="flex items-center gap-2 p-1.5 bg-gray-800 rounded">
        <div className="flex-1 text-xs text-gray-300 truncate" title={label}>
            {label}
        </div>
        <input
            type="color"
            value={color}
            onChange={(e) => onChange(e.target.value)}
            className="w-5 h-5 rounded cursor-pointer border border-gray-600 bg-transparent"
        />
        <input
            type="text"
            value={color}
            onChange={(e) => onChange(e.target.value)}
            className="w-14 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white font-mono"
        />
    </div>
);

const TypeColorEditor = ({ graphData, vizConfig, setVizConfig }) => {
    // Extract unique types from graph data
    const { nodeTypes, edgeTypes } = useMemo(() => {
        const nodes = graphData?.nodes || [];
        const links = graphData?.links || graphData?.edges || [];

        const nodeTypeCounts = new Map();
        const edgeTypeCounts = new Map();

        nodes.forEach(n => {
            const t = n.pz_type || n.type;
            if (t) nodeTypeCounts.set(t, (nodeTypeCounts.get(t) || 0) + 1);
        });

        links.forEach(l => {
            const t = l.pz_type || l.type;
            if (t) edgeTypeCounts.set(t, (edgeTypeCounts.get(t) || 0) + 1);
        });

        return {
            nodeTypes: Array.from(nodeTypeCounts.entries()).sort((a, b) => b[1] - a[1]),
            edgeTypes: Array.from(edgeTypeCounts.entries()).sort((a, b) => b[1] - a[1]),
        };
    }, [graphData]);

    // Get settings for a node type
    const getNodeSettings = (type) => {
        return vizConfig?.nodeTypeSettings?.[type] || {};
    };

    // Get settings for an edge type
    const getEdgeSettings = (type) => {
        return vizConfig?.edgeTypeSettings?.[type] || {};
    };

    // Update node type settings
    const setNodeSettings = (type, settings) => {
        setVizConfig(prev => ({
            ...prev,
            nodeTypeSettings: {
                ...prev.nodeTypeSettings,
                [type]: settings,
            },
        }));
    };

    // Update edge type settings
    const setEdgeSettings = (type, settings) => {
        setVizConfig(prev => ({
            ...prev,
            edgeTypeSettings: {
                ...prev.edgeTypeSettings,
                [type]: settings,
            },
        }));
    };

    // Get state color
    const getStateColor = (key) => {
        return vizConfig?.stateColors?.[key] || '#888888';
    };

    // Update state color
    const setStateColor = (key, color) => {
        setVizConfig(prev => ({
            ...prev,
            stateColors: {
                ...prev.stateColors,
                [key]: color,
            },
        }));
    };

    // Bulk operations
    const showAllNodes = () => {
        const updated = { ...vizConfig.nodeTypeSettings };
        nodeTypes.forEach(([type]) => {
            updated[type] = { ...updated[type], visible: true };
        });
        setVizConfig(prev => ({ ...prev, nodeTypeSettings: updated }));
    };

    const hideAllNodes = () => {
        const updated = { ...vizConfig.nodeTypeSettings };
        nodeTypes.forEach(([type]) => {
            updated[type] = { ...updated[type], visible: false };
        });
        setVizConfig(prev => ({ ...prev, nodeTypeSettings: updated }));
    };

    const showAllEdges = () => {
        const updated = { ...vizConfig.edgeTypeSettings };
        edgeTypes.forEach(([type]) => {
            updated[type] = { ...updated[type], visible: true };
        });
        setVizConfig(prev => ({ ...prev, edgeTypeSettings: updated }));
    };

    const hideAllEdges = () => {
        const updated = { ...vizConfig.edgeTypeSettings };
        edgeTypes.forEach(([type]) => {
            updated[type] = { ...updated[type], visible: false };
        });
        setVizConfig(prev => ({ ...prev, edgeTypeSettings: updated }));
    };

    // Reset all type settings to defaults
    const resetAllSettings = () => {
        setVizConfig(prev => ({
            ...prev,
            nodeTypeSettings: {},
            edgeTypeSettings: {},
        }));
    };

    // Solo a specific node type (hide all others)
    const soloNodeType = (targetType) => {
        const updated = {};
        nodeTypes.forEach(([type]) => {
            updated[type] = { ...vizConfig.nodeTypeSettings?.[type], visible: type === targetType };
        });
        setVizConfig(prev => ({ ...prev, nodeTypeSettings: updated }));
    };

    // Solo a specific edge type (hide all others)
    const soloEdgeType = (targetType) => {
        const updated = {};
        edgeTypes.forEach(([type]) => {
            updated[type] = { ...vizConfig.edgeTypeSettings?.[type], visible: type === targetType };
        });
        setVizConfig(prev => ({ ...prev, edgeTypeSettings: updated }));
    };

    if (!graphData && nodeTypes.length === 0) {
        return (
            <div className="text-xs text-gray-500 italic p-2 bg-gray-800 rounded border border-gray-700">
                Load a graph to configure type settings
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {/* Node Types */}
            {nodeTypes.length > 0 && (
                <Section title="Node Types" icon={Palette} count={nodeTypes.length} defaultOpen={true}>
                    <div className="flex gap-1 mb-2">
                        <button
                            onClick={showAllNodes}
                            className="text-[10px] px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                            Show All
                        </button>
                        <button
                            onClick={hideAllNodes}
                            className="text-[10px] px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                            Hide All
                        </button>
                    </div>
                    <div className="space-y-1 max-h-48 overflow-y-auto">
                        {nodeTypes.map(([type, count]) => (
                            <NodeTypeRow
                                key={`node-${type}`}
                                type={type}
                                count={count}
                                settings={getNodeSettings(type)}
                                defaults={DEFAULT_NODE_SETTINGS[type]}
                                onChange={(s) => setNodeSettings(type, s)}
                                onSolo={() => soloNodeType(type)}
                            />
                        ))}
                    </div>
                </Section>
            )}

            {/* Edge Types */}
            {edgeTypes.length > 0 && (
                <Section title="Edge Types" icon={Palette} count={edgeTypes.length} defaultOpen={true}>
                    <div className="flex gap-1 mb-2">
                        <button
                            onClick={showAllEdges}
                            className="text-[10px] px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                            Show All
                        </button>
                        <button
                            onClick={hideAllEdges}
                            className="text-[10px] px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                            Hide All
                        </button>
                    </div>
                    <div className="space-y-1 max-h-48 overflow-y-auto">
                        {edgeTypes.map(([type, count]) => (
                            <EdgeTypeRow
                                key={`edge-${type}`}
                                type={type}
                                count={count}
                                settings={getEdgeSettings(type)}
                                defaults={DEFAULT_EDGE_SETTINGS[type]}
                                baseLinkStrength={vizConfig?.linkStrength}
                                onChange={(s) => setEdgeSettings(type, s)}
                                onSolo={() => soloEdgeType(type)}
                            />
                        ))}
                    </div>
                </Section>
            )}

            {/* Reset All */}
            <button
                onClick={resetAllSettings}
                className="w-full text-[11px] px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
            >
                Reset All Type Settings to Defaults
            </button>

            {/* State Colors */}
            <Section title="Highlight States" icon={Palette} defaultOpen={false}>
                <ColorPicker
                    label="Evidence"
                    color={getStateColor('evidence')}
                    onChange={(c) => setStateColor('evidence', c)}
                />
                <ColorPicker
                    label="On Path"
                    color={getStateColor('onPath')}
                    onChange={(c) => setStateColor('onPath', c)}
                />
                <ColorPicker
                    label="Active Path"
                    color={getStateColor('activePath')}
                    onChange={(c) => setStateColor('activePath', c)}
                />
                <ColorPicker
                    label="Visited"
                    color={getStateColor('visited')}
                    onChange={(c) => setStateColor('visited', c)}
                />
                <ColorPicker
                    label="Entry Point"
                    color={getStateColor('entry')}
                    onChange={(c) => setStateColor('entry', c)}
                />
            </Section>

            {/* Preset Palette */}
            <Section title="Color Palette" defaultOpen={false}>
                <div className="flex flex-wrap gap-1">
                    {COLOR_PRESETS.map(color => (
                        <div
                            key={color}
                            className="w-5 h-5 rounded cursor-pointer border border-gray-600 hover:scale-110 transition-transform"
                            style={{ backgroundColor: color }}
                            title={`Click to copy: ${color}`}
                            onClick={() => navigator.clipboard.writeText(color)}
                        />
                    ))}
                </div>
                <div className="text-[10px] text-gray-500 mt-1">Click a color to copy to clipboard</div>
            </Section>
        </div>
    );
};

export default TypeColorEditor;
