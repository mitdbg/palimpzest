import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react';
import GraphSearch from './GraphSearch';
import { 
    DEFAULT_NODE_SETTINGS, 
    DEFAULT_EDGE_SETTINGS, 
    DEFAULT_NODE_FALLBACK, 
    DEFAULT_EDGE_FALLBACK 
} from './TypeColorEditor';

// ==================== DEFAULT COLORS (legacy, for state colors) ====================
const DEFAULT_STATE_COLORS = {
    evidence: '#f59e0b',
    onPath: '#10b981',
    activePath: '#06b6d4',
    visited: '#3b82f6',
    entry: '#ec4899',
};

// Get full settings for a node type (color, size, opacity, shape, visible)
const getNodeTypeSettings = (nodeType, vizConfig) => {
    const userSettings = vizConfig?.nodeTypeSettings?.[nodeType] || {};
    const defaults = DEFAULT_NODE_SETTINGS[nodeType] || DEFAULT_NODE_FALLBACK;
    return {
        color: userSettings.color ?? defaults.color ?? DEFAULT_NODE_FALLBACK.color,
        size: userSettings.size ?? defaults.size ?? DEFAULT_NODE_FALLBACK.size,
        opacity: userSettings.opacity ?? defaults.opacity ?? DEFAULT_NODE_FALLBACK.opacity,
        shape: userSettings.shape ?? defaults.shape ?? DEFAULT_NODE_FALLBACK.shape,
        visible: userSettings.visible ?? true,
    };
};

// Get full settings for an edge type (color, width, opacity, style, strength, visible)
const getEdgeTypeSettings = (edgeType, vizConfig) => {
    const userSettings = vizConfig?.edgeTypeSettings?.[edgeType] || {};
    const defaults = DEFAULT_EDGE_SETTINGS[edgeType] || DEFAULT_EDGE_FALLBACK;
    return {
        color: userSettings.color ?? defaults.color ?? DEFAULT_EDGE_FALLBACK.color,
        width: userSettings.width ?? defaults.width ?? DEFAULT_EDGE_FALLBACK.width,
        opacity: userSettings.opacity ?? defaults.opacity ?? DEFAULT_EDGE_FALLBACK.opacity,
        style: userSettings.style ?? defaults.style ?? DEFAULT_EDGE_FALLBACK.style,
        strength: userSettings.strength ?? defaults.strength ?? DEFAULT_EDGE_FALLBACK.strength,
        distance: userSettings.distance ?? defaults.distance ?? DEFAULT_EDGE_FALLBACK.distance,
        visible: userSettings.visible ?? true,
    };
};

// Create color getter functions that use vizConfig
const getNodeColor = (node, vizConfig) => {
    const stateColors = vizConfig?.stateColors || DEFAULT_STATE_COLORS;
    
    // Priority: state > type
    if (node.isEvidence) return stateColors.evidence || DEFAULT_STATE_COLORS.evidence;
    if (node.isOnPath) return stateColors.onPath || DEFAULT_STATE_COLORS.onPath;
    if (node.isActivePath) return stateColors.activePath || DEFAULT_STATE_COLORS.activePath;
    if (node.visited) return stateColors.visited || DEFAULT_STATE_COLORS.visited;
    if (node.isFiltered) return '#ef4444'; // Red for filtered nodes
    
    const nodeType = node.pz_type || node.type;
    if (nodeType === 'entry') return stateColors.entry || DEFAULT_STATE_COLORS.entry;
    
    const settings = getNodeTypeSettings(nodeType, vizConfig);
    return settings.color;
};

const getEdgeColor = (edge, vizConfig) => {
    const stateColors = vizConfig?.stateColors || DEFAULT_STATE_COLORS;
    
    if (edge.isOnPath) return stateColors.onPath || DEFAULT_STATE_COLORS.onPath;
    if (edge.isActivePath) return stateColors.activePath || DEFAULT_STATE_COLORS.activePath;
    
    const edgeType = edge.pz_type || edge.type || '';
    const settings = getEdgeTypeSettings(edgeType, vizConfig);
    return settings.color;
};

// Node sizing based on type and importance
const getNodeRadius = (node, vizConfig) => {
    if (node.isEvidence) return 8;
    const nodeType = node.pz_type || node.type;
    if (nodeType === 'entry') return 5;
    
    const baseSize = vizConfig?.nodeBaseSize ?? 3;
    const settings = getNodeTypeSettings(nodeType, vizConfig);
    return baseSize * settings.size;
};

const getId = (v) => (typeof v === 'object' && v !== null ? v.id : v);

const hexToRgb = (hex) => {
    if (typeof hex !== 'string') return null;
    const h = hex.replace('#', '');
    if (h.length !== 6) return null;
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null;
    return { r, g, b };
};

const withAlpha = (hex, alpha) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Simple deterministic hash for jitter (FNV-1a 32-bit)
const hash32 = (s) => {
    const str = String(s ?? '');
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
};

const computeInitSpacing = (nodeCount, vizConfig) => {
    const linkDistance = vizConfig?.linkDistance ?? 90;
    const chargeStrength = vizConfig?.chargeStrength ?? -400;
    const chargeFactor = clamp(Math.sqrt(Math.abs(chargeStrength) / 800), 0.7, 1.8);
    const sizeFactor = nodeCount >= 20000 ? 1.25 : (nodeCount >= 5000 ? 1.1 : 1.0);
    return clamp(linkDistance * 0.9 * chargeFactor * sizeFactor, 12, 220);
};

const initializeNodePositions = ({ nodes, width, height, vizConfig }) => {
    const n = nodes?.length || 0;
    if (!n) return;

    const w = Math.max(300, width || 0);
    const h = Math.max(300, height || 0);
    const minDim = Math.min(w, h);

    const spacing = computeInitSpacing(n, vizConfig);
    const idealR = spacing * Math.sqrt(n / Math.PI);
    const targetR = clamp(idealR, 120, 9000);
    const aspect = w / h;
    const golden = Math.PI * (3 - Math.sqrt(5));

    let maxDepth = 0;
    for (const nd of nodes) {
        if (typeof nd?._depth === 'number') maxDepth = Math.max(maxDepth, nd._depth);
    }
    const useDepthRings = maxDepth >= 2 && n >= 600;
    const ringStep = spacing * 2.2;
    const baseR = spacing * 3;
    const ringCap = baseR + maxDepth * ringStep;
    const ringScale = useDepthRings && ringCap > targetR ? (targetR / ringCap) : 1.0;

    for (let i = 0; i < n; i++) {
        const nd = nodes[i];
        if (!nd) continue;
        if (nd.fx !== undefined || nd.fy !== undefined) continue;

        const h32 = hash32(nd.id);
        const jitter = ((h32 % 1000) / 1000 - 0.5) * spacing * 0.35;
        const angleOffset = ((h32 >>> 10) % 1000) / 1000 * 2 * Math.PI;
        const theta = i * golden + angleOffset;

        let r;
        if (useDepthRings && typeof nd._depth === 'number') {
            r = (baseR + nd._depth * ringStep) * ringScale;
            r += ((h32 >>> 20) % 1000) / 1000 * spacing * 0.6;
        } else {
            r = targetR * Math.sqrt((i + 0.5) / n);
            r += jitter;
        }

        let x = r * Math.cos(theta);
        let y = r * Math.sin(theta);

        if (aspect > 1.05) x *= clamp(aspect, 1.0, 2.0);
        else if (aspect < 0.95) y *= clamp(1 / aspect, 1.0, 2.0);

        const bound = Math.max(minDim * 3, 800);
        x = clamp(x, -bound, bound);
        y = clamp(y, -bound, bound);

        nd.x = x;
        nd.y = y;
        nd.vx = 0;
        nd.vy = 0;
    }
};

// Dynamic legend component that shows actual types from the graph
const DynamicLegend = ({ data, vizConfig }) => {
    // Extract unique types from current data
    const nodeTypes = useMemo(() => {
        const typeCounts = new Map();
        for (const n of data.nodes || []) {
            const t = n.pz_type || n.type || 'unknown';
            typeCounts.set(t, (typeCounts.get(t) || 0) + 1);
        }
        return Array.from(typeCounts.entries())
            .sort((a, b) => b[1] - a[1])  // Sort by count descending
            .slice(0, 6);  // Top 6 types
    }, [data.nodes]);

    const edgeTypes = useMemo(() => {
        const typeCounts = new Map();
        for (const l of data.links || []) {
            const t = l.pz_type || l.type || 'unknown';
            typeCounts.set(t, (typeCounts.get(t) || 0) + 1);
        }
        return Array.from(typeCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4);  // Top 4 edge types
    }, [data.links]);

    const stateColors = vizConfig?.stateColors || DEFAULT_STATE_COLORS;

    if (nodeTypes.length === 0) return null;

    return (
        <div className="absolute bottom-24 left-4 z-10 bg-slate-900/90 backdrop-blur-sm rounded-lg p-3 border border-slate-700/50 text-xs max-w-48">
            <div className="text-slate-400 font-medium mb-2">Legend</div>
            
            {/* Node types */}
            <div className="space-y-1">
                {nodeTypes.map(([type, count]) => (
                    <div key={`node-${type}`} className="flex items-center gap-2">
                        <div 
                            className="w-2.5 h-2.5 rounded-full flex-shrink-0" 
                            style={{ backgroundColor: getNodeColor({ type }, vizConfig) }}
                        />
                        <span className="text-slate-300 truncate" title={`${type} (${count})`}>
                            {type}
                        </span>
                        <span className="text-slate-500 ml-auto text-[10px]">{count}</span>
                    </div>
                ))}
            </div>
            
            {/* Edge types */}
            {edgeTypes.length > 0 && (
                <div className="mt-2 pt-2 border-t border-slate-700/50 space-y-1">
                    {edgeTypes.map(([type, count]) => (
                        <div key={`edge-${type}`} className="flex items-center gap-2">
                            <div 
                                className="w-4 h-0.5 flex-shrink-0" 
                                style={{ backgroundColor: getEdgeColor({ type }, vizConfig) }}
                            />
                            <span className="text-slate-400 truncate text-[11px]" title={`${type} (${count})`}>
                                {type}
                            </span>
                            <span className="text-slate-500 ml-auto text-[10px]">{count}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* State indicators */}
            <div className="mt-2 pt-2 border-t border-slate-700/50 space-y-1">
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: stateColors.evidence }} />
                    <span className="text-slate-400 text-[11px]">Evidence</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: stateColors.onPath }} />
                    <span className="text-slate-400 text-[11px]">On Path</span>
                </div>
            </div>
        </div>
    );
};

function GraphVisualization({ graphData, onNodeClick, onNodeHover, vizConfig, refreshNonce }) {
    const containerRef = useRef(null);
    const fgRef = useRef();
    const positionCacheRef = useRef(new Map()); // Cache node positions to preserve across config changes
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [hoveredNode, setHoveredNode] = useState(null);
    const hoveredNodeRef = useRef(null);
    const [perfStats, setPerfStats] = useState(null);
    const perfRef = useRef({
        enabled: false,
        // Sampling (to avoid making perf HUD the bottleneck)
        nodeSampleStride: 10,
        linkSampleStride: 25,

        frameStart: 0,
        lastReport: 0,
        frames: 0,
        frameMsSum: 0,
        frameMsMax: 0,
        runningFrames: 0,
        runningMsSum: 0,
        idleFrames: 0,
        idleMsSum: 0,
        lastScale: 1,
        tickCount: 0,
        lastTickReport: 0,
        tickHz: 0,

        // Force simulation timing
        forceTickMs: 0,
        forceTickMsSum: 0,
        lastForceTickEnd: 0,

        // NEW: Detailed phase timing
        phaseViewportMs: 0,
        phaseViewportMsSum: 0,
        phaseLinkCullMs: 0,
        phaseLinkCullMsSum: 0,
        phaseLinkBatchMs: 0,
        phaseLinkBatchMsSum: 0,
        phaseLinkDrawMs: 0,
        phaseLinkDrawMsSum: 0,
        phaseNodeIterMs: 0,
        phaseNodeIterMsSum: 0,
        phaseNodeDrawMs: 0,
        phaseNodeDrawMsSum: 0,
        phaseNodeCullCount: 0,
        phaseNodeCullCountSum: 0,
        phaseNodeDrawCount: 0,
        phaseNodeDrawCountSum: 0,
        phaseLinkCullCount: 0,
        phaseLinkCullCountSum: 0,
        phaseLinkDrawCount: 0,
        phaseLinkDrawCountSum: 0,

        // Per-frame breakdown accumulators (reset in onRenderFramePre)
        frameNodeCalls: 0,
        frameNodeMsEst: 0,
        frameNodeGlowMsEst: 0,
        frameNodeBorderMsEst: 0,
        frameNodeLabelMsEst: 0,

        frameLinkCbCalls: 0,
        frameLinkCbMsEst: 0,
        frameLinkColorMsEst: 0,
        frameLinkWidthMsEst: 0,
        frameLinkDashMsEst: 0,
        frameLinkArrowMsEst: 0,
        framePointerCalls: 0,
        framePointerMsEst: 0,

        // Custom renderer timing (when useCustomLinkRender=true)
        frameCustomLinkDrawMs: 0,

        // Window sums (accumulated across frames, reset on report)
        nodeCallsSum: 0,
        pointerCallsSum: 0,
        nodeMsSum: 0,
        nodeGlowMsSum: 0,
        nodeBorderMsSum: 0,
        nodeLabelMsSum: 0,
        pointerMsSum: 0,
        linkCbMsSum: 0,
        linkColorMsSum: 0,
        linkWidthMsSum: 0,
        linkDashMsSum: 0,
        linkArrowMsSum: 0,
        customLinkDrawMsSum: 0,
        inferredLinkDrawMsSum: 0,
    });

    // Resize observer to handle container sizing
    useEffect(() => {
        if (!containerRef.current) return;
        
        const ro = new ResizeObserver((entries) => {
            const rect = entries[0]?.contentRect;
            if (rect) {
                setDimensions({ width: rect.width, height: rect.height });
            }
        });
        
        ro.observe(containerRef.current);
        return () => ro.disconnect();
    }, []);

    // Save current positions before data recalculation
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg || typeof fg.graphData !== 'function') return;
        try {
            const currentData = fg.graphData();
            if (currentData?.nodes) {
                currentData.nodes.forEach(n => {
                    if (typeof n.x === 'number' && typeof n.y === 'number') {
                        positionCacheRef.current.set(n.id, { x: n.x, y: n.y, vx: n.vx || 0, vy: n.vy || 0 });
                    }
                });
            }
        } catch (e) {
            // Ignore errors during initialization
        }
    }, [graphData, vizConfig?.maxEdges]);

    // Prepare STRUCTURAL data — only recompute when graphData or maxEdges changes.
    // This ensures hiding/unhiding types doesn't create new node/link arrays (which would restart the sim).
    const structuralData = useMemo(() => {
        const rawNodes = Array.isArray(graphData?.nodes) ? graphData.nodes : [];
        const rawLinks = Array.isArray(graphData?.links)
            ? graphData.links
            : (Array.isArray(graphData?.edges) ? graphData.edges : []);

        // Build node map for quick lookup
        const nodeMap = new Map(rawNodes.map(n => [n.id, n]));
        
        // Compute hierarchy depth via BFS from root documents
        const depthMap = new Map();
        const hierarchyLinks = rawLinks.filter(l => {
            const t = l.type || l.pz_type || '';
            return t.includes('hierarchy') || t.includes('child');
        });
        
        // Find roots (documents or nodes with no incoming hierarchy edges)
        const hasParent = new Set(
            hierarchyLinks.map(l => {
                const dst = l.target ?? l.dst;
                return (typeof dst === 'object' && dst !== null) ? dst.id : dst;
            })
        );
        const roots = rawNodes.filter(n => {
            const isDoc = (n.type || n.pz_type) === 'document';
            const noParent = !hasParent.has(n.id);
            return isDoc || noParent;
        });
        
        // BFS to compute depth
        const queue = roots.map(n => ({ id: n.id, depth: 0 }));
        while (queue.length > 0) {
            const { id, depth } = queue.shift();
            if (depthMap.has(id)) continue;
            depthMap.set(id, depth);
            
            // Find children
            for (const link of hierarchyLinks) {
                const src = link.source ?? link.src;
                const dst = link.target ?? link.dst;
                const srcId = (typeof src === 'object' && src !== null) ? src.id : src;
                const dstId = (typeof dst === 'object' && dst !== null) ? dst.id : dst;
                if (srcId === id && dstId != null && !depthMap.has(dstId)) {
                    queue.push({ id: dstId, depth: depth + 1 });
                }
            }
        }

        // Create nodes (all of them — visibility is handled at render time)
        const nodes = rawNodes.map(n => {
            const depth = depthMap.get(n.id) ?? 0;
            const cachedPos = positionCacheRef.current.get(n.id);
            return {
                ...n,
                _depth: depth,
                label: n.summary || n.label || n.id,
                ...(cachedPos && { x: cachedPos.x, y: cachedPos.y, vx: cachedPos.vx, vy: cachedPos.vy })
            };
        });

        const nodeIds = new Set(nodes.map(n => n.id));

        // Filter links to those with valid endpoints, then apply maxEdges
        const filteredLinks = rawLinks.filter(l => {
            const s = l.source || l.src;
            const t = l.target || l.dst;
            const sId = typeof s === 'object' ? s.id : s;
            const tId = typeof t === 'object' ? t.id : t;
            return nodeIds.has(sId) && nodeIds.has(tId);
        });

        const maxEdges = vizConfig?.maxEdges ?? 10000;
        const limitedLinks = (maxEdges && maxEdges > 0)
            ? filteredLinks.slice(0, maxEdges)
            : filteredLinks;

        // Normalize source/target
        const links = limitedLinks.map(l => {
            const s = l.source || l.src;
            const t = l.target || l.dst;
            const sid = typeof s === 'object' && s !== null ? s.id : s;
            const tid = typeof t === 'object' && t !== null ? t.id : t;
            return {
                ...l,
                // Canonicalize endpoints to node IDs so the force engine and renderer
                // never see stale/foreign node objects.
                source: sid,
                target: tid,
                _sid: sid,
                _tid: tid,
            };
        });

        return { nodes, links, depthMap };
    }, [graphData, vizConfig?.maxEdges]);

    // Compute RENDER properties — updates in-place on the existing nodes/links when vizConfig changes.
    // This does NOT change array identity, so ForceGraph won't restart the simulation.
    const data = useMemo(() => {
        const { nodes, links } = structuralData;

        // Small cache for rgba conversion
        const rgbaCache = new Map();
        const withAlphaCached = (hex, alpha) => {
            const key = `${hex}|${alpha}`;
            const cached = rgbaCache.get(key);
            if (cached) return cached;
            const rgba = withAlpha(hex, alpha);
            rgbaCache.set(key, rgba);
            return rgba;
        };

        const isLowQuality = (links.length >= 30000) || (nodes.length >= 6000);
        const dimOpacity = vizConfig?.dimOpacity ?? 0.02;
        const pathOpacity = vizConfig?.pathOpacity ?? 0.9;
        const baseEdgeWidth = vizConfig?.edgeWidth ?? 0.7;

        // Update node render properties in-place
        for (const n of nodes) {
            const nodeType = n.pz_type || n.type;
            const settings = getNodeTypeSettings(nodeType, vizConfig);
            n._visibleDraw = settings.visible;
            n._shape = settings.shape;
            n.val = getNodeRadius(n, vizConfig);
            n.color = getNodeColor(n, vizConfig);
        }

        // Update link render properties in-place
        for (const l of links) {
            const edgeType = l.pz_type || l.type || '';
            const settings = getEdgeTypeSettings(edgeType, vizConfig);

            const isPath = !!(l.isOnPath || l.isActivePath);
            const typeVisible = settings.visible ?? true;
            l._visibleDraw = typeVisible || isPath;

            const typeStyle = settings.style;
            l._dash = typeStyle === 'dashed' ? [4, 4] : (typeStyle === 'dotted' ? [2, 2] : null);
            l._isReference = edgeType.includes('reference');
            l._typeStrength = settings.strength;
            l._typeDistance = settings.distance;

            const baseColor = getEdgeColor(l, vizConfig);
            const typeOpacity = settings.opacity ?? 0.2;
            const incidentOpacity = Math.max(typeOpacity, 0.5);
            l.color = baseColor;
            l._color = withAlphaCached(baseColor, typeOpacity);
            l._colorPath = withAlphaCached(baseColor, pathOpacity);
            l._colorDim = withAlphaCached(baseColor, dimOpacity);
            l._colorIncident = withAlphaCached(baseColor, incidentOpacity);

            const typeWidth = settings.width ?? 1.0;
            const widthNormal = baseEdgeWidth * typeWidth;
            const widthPath = Math.max(2, widthNormal * 2.5);
            l._widthDraw = isPath ? widthPath : widthNormal;
            l._colorDraw = isPath ? l._colorPath : l._color;
            l._arrowLen = isPath ? 4 : (isLowQuality ? 0 : (l._isReference ? 3 : 0));
        }

        // Rendering should only receive visible links.
        // This avoids expensive per-frame work inside react-force-graph on huge graphs
        // (it checks link arrays for particles/needsRedraw even when links are hidden).
        const renderLinks = links.filter(l => l?._visibleDraw !== false);

        // Return SAME nodes array, but a filtered render link array.
        // Keep `allLinks` for force simulation / counters.
        return { nodes, links: renderLinks, allLinks: links };
    }, [
        structuralData,
        vizConfig?.nodeTypeSettings,
        vizConfig?.edgeTypeSettings,
        vizConfig?.stateColors,
        vizConfig?.dimOpacity,
        vizConfig?.pathOpacity,
        vizConfig?.edgeWidth,
        vizConfig?.nodeBaseSize,
    ]);

    const neighborMap = useMemo(() => {
        if (!vizConfig?.hoverHighlight) return null;
        const map = new Map();
        for (const n of data.nodes) {
            map.set(n.id, new Set([n.id]));
        }
        for (const l of data.links) {
            const s = getId(l.source);
            const t = getId(l.target);
            if (map.has(s)) map.get(s).add(t);
            if (map.has(t)) map.get(t).add(s);
        }
        return map;
    }, [data, vizConfig?.hoverHighlight]);

    // Large-graph rendering: skip the most expensive node effects.
    const lowQuality = useMemo(() => {
        const edges = data.allLinks?.length || data.links?.length || 0;
        const nodes = data.nodes?.length || 0;
        return edges >= 30000 || nodes >= 6000;
    }, [data.allLinks?.length, data.links?.length, data.nodes?.length]);

    // For VERY large graphs, disable expensive interactions that cause full redraws
    const hugeGraph = useMemo(() => {
        const edges = data.allLinks?.length || data.links?.length || 0;
        return edges >= 100000;
    }, [data.allLinks?.length, data.links?.length]);

    // Data to pass to ForceGraph2D: for huge graphs, pass empty links array
    // to eliminate all internal link processing overhead (we draw links ourselves).
    // Note: The threshold here matches useCustomLinkRender (20k edges) so we only
    // strip links when we have a custom renderer to draw them.
    const forceGraphData = useMemo(() => {
        const edges = data.allLinks?.length || data.links?.length || 0;
        const willUseCustomRender = edges >= 20000;
        if (hugeGraph && willUseCustomRender) {
            // Pass nodes + empty links to the library; we draw all links in onRenderFramePre
            return { nodes: data.nodes, links: [] };
        }
        return data;
    }, [data, hugeGraph]);

    // Ensure simulation starts when component mounts with data
    // This effect handles the initial load case
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg || !data.nodes?.length) return;
        
        // Small delay to ensure ForceGraph2D has fully initialized
        const timer = setTimeout(() => {
            fg.d3ReheatSimulation();
        }, 100);
        
        return () => clearTimeout(timer);
    }, [data.nodes?.length]); // Only re-run when node count changes

    // Break the default grid initialization on graph load.
    // react-force-graph initializes nodes in a grid if x/y are missing.
    // For dense graphs, that symmetry + centering yields the "solid disk" look.
    // We use a deterministic, size-aware initializer so large graphs start spread out.
    //
    // IMPORTANT: Only depend on STRUCTURAL data changes and layout-relevant config,
    // NOT color/style settings. Otherwise changing colors restarts the simulation.
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return;
        const { nodes } = data;
        if (!nodes?.length) return;

        const hasAnyPosition = nodes.some(n => typeof n.x === 'number' && typeof n.y === 'number');
        if (hasAnyPosition) return;

        initializeNodePositions({
            nodes,
            width: dimensions.width,
            height: dimensions.height,
            vizConfig,
        });

        if (vizConfig?.runLayout) {
            fg.d3ReheatSimulation();
        } else {
            fg.refresh?.();
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [data.nodes?.length, vizConfig?.runLayout, dimensions.width, dimensions.height]);

    // Layout configuration
    // Removed dagMode/dagLevelDistance as we are focusing on pure force layout per user request.
    
    // Re-heat simulation on refresh
    useEffect(() => {
        if (refreshNonce > 0 && fgRef.current) {
            // Scramble positions to force a visible re-layout
            const { nodes } = data;
            const spacing = computeInitSpacing(nodes.length || 0, vizConfig);
            const rMax = clamp(spacing * Math.sqrt(Math.max(1, nodes.length) / Math.PI), 120, 9000);
            nodes.forEach(n => {
                // Only scramble if not fixed
                if (n.fx === undefined && n.fy === undefined) {
                    // Initialize in a tight circle to force an "explosion" layout
                    const r = rMax * Math.sqrt(Math.random());
                    const theta = Math.random() * 2 * Math.PI;
                    n.x = r * Math.cos(theta);
                    n.y = r * Math.sin(theta);
                    n.vx = 0;
                    n.vy = 0;
                }
            });
            if (vizConfig?.runLayout) {
                fgRef.current.d3ReheatSimulation();
                fgRef.current.zoomToFit(400, 40);
            }
        }
    }, [refreshNonce, data, vizConfig?.runLayout]);

    // Apply force settings
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return;

        // Charge (many-body)
        if (fg.d3Force('charge')) {
            const charge = fg.d3Force('charge');
            charge.strength(vizConfig?.chargeStrength ?? -400);
            if (typeof charge.distanceMax === 'function') {
                const maxD = vizConfig?.chargeDistanceMax;
                charge.distanceMax(maxD == null ? Number.POSITIVE_INFINITY : maxD);
            }
        }

        // Link force: For huge graphs (100k+ edges), disable the link force entirely
        // Link force configuration
        if (fg.d3Force('link')) {
            const link = fg.d3Force('link');
            const baseDistance = vizConfig?.linkDistance ?? 90;

            // IMPORTANT: Render links may be filtered down to visible-only for perf.
            // Keep the full link set in the force simulation when layout is enabled.
            if (vizConfig?.runLayout && typeof link.links === 'function' && Array.isArray(data?.allLinks)) {
                // d3-force will throw if any link endpoint isn't in the node set.
                // Some datasets contain dangling edges; filter them out defensively.
                const nodeIds = new Set((data.nodes || []).map(n => n?.id));
                const allLinks = data.allLinks;
                const validLinks = allLinks.filter(l => {
                    const s = l?.source;
                    const t = l?.target;
                    const sid = (typeof s === 'object' && s !== null) ? s.id : s;
                    const tid = (typeof t === 'object' && t !== null) ? t.id : t;
                    return nodeIds.has(sid) && nodeIds.has(tid);
                });

                try {
                    link.links(validLinks);
                } catch (e) {
                    // Last-resort: if the library still throws, don't crash the UI.
                    // We'll just leave the existing link set in place.
                    // eslint-disable-next-line no-console
                    console.warn('Failed to set full link force links (skipping):', e);
                }
            }

            link.distance((l) => {
                const m = l?._typeDistance;
                const mult = (typeof m === 'number' && Number.isFinite(m) && m > 0) ? m : 1.0;
                return baseDistance * mult;
            });
            if (typeof link.strength === 'function') {
                // Use per-link strength based on edge type
                const baseStrength = vizConfig?.linkStrength ?? 0.7;
                link.strength((l) => {
                    // _typeStrength is set during data preparation from edgeTypeSettings
                    const typeStrength = l._typeStrength ?? 1.0;
                    return baseStrength * typeStrength;
                });
            }
        }

        // Important: react-force-graph includes a default d3.forceCenter().
        // If we add x/y forces, we must disable the default center force or we'll over-constrain to the origin.
        fg.d3Force('center', null);

        // Centering: d3.forceCenter has no "strength".
        // Use x/y forces toward origin instead (react-force-graph's coordinate space is centered).
        const centerStrength = vizConfig?.centerStrength ?? 0.04;
        fg.d3Force('x', d3.forceX(0).strength(centerStrength));
        fg.d3Force('y', d3.forceY(0).strength(centerStrength));

        // Collision - disabled for huge graphs (O(n log n) per tick is expensive at 30k nodes)
        if (hugeGraph) {
            fg.d3Force('collide', null);
        } else {
            const r = vizConfig?.collisionRadius ?? 1.5;
            const radiusScale = vizConfig?.nodeRadiusScale ?? 1.0;
            fg.d3Force('collide', d3.forceCollide(node => ((node.val || 3) * radiusScale) + r));
        }
    }, [
        hugeGraph,
        vizConfig?.chargeStrength, 
        vizConfig?.chargeDistanceMax,
        vizConfig?.linkDistance, 
        vizConfig?.linkStrength,
        vizConfig?.centerStrength, 
        vizConfig?.collisionRadius,
        vizConfig?.nodeRadiusScale,
        vizConfig?.runLayout,
        data?.allLinks,
        data?.nodes,
        // Note: d3VelocityDecay is handled via prop
    ]);

    // Auto-fit on graph load (only when layout is running)
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg || !vizConfig?.runLayout) return;
        // Let a few frames render before fitting
        const t = setTimeout(() => {
            try {
                fg.zoomToFit(500, 60);
            } catch (_) {
                // ignore
            }
        }, 200);
        return () => clearTimeout(t);
    }, [data, vizConfig?.runLayout]);

    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return;

        if (vizConfig?.runLayout) {
            fg.resumeAnimation?.();
            fg.d3ReheatSimulation();
        } else {
            fg.pauseAnimation?.();
        }
    }, [vizConfig?.runLayout]);

    // Shape drawing helpers
    const drawShape = useCallback((ctx, x, y, r, shape) => {
        ctx.beginPath();
        switch (shape) {
            case 'square':
                ctx.rect(x - r, y - r, r * 2, r * 2);
                break;
            case 'diamond':
                ctx.moveTo(x, y - r * 1.2);
                ctx.lineTo(x + r * 1.2, y);
                ctx.lineTo(x, y + r * 1.2);
                ctx.lineTo(x - r * 1.2, y);
                ctx.closePath();
                break;
            case 'triangle':
                ctx.moveTo(x, y - r * 1.15);
                ctx.lineTo(x + r * 1.1, y + r * 0.7);
                ctx.lineTo(x - r * 1.1, y + r * 0.7);
                ctx.closePath();
                break;
            case 'hexagon':
                for (let i = 0; i < 6; i++) {
                    const angle = (Math.PI / 3) * i - Math.PI / 2;
                    const px = x + r * Math.cos(angle);
                    const py = y + r * Math.sin(angle);
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.closePath();
                break;
            case 'circle':
            default:
                ctx.arc(x, y, r, 0, 2 * Math.PI, false);
                break;
        }
    }, []);

    // Node painting - with viewport culling for performance
    const paintNode = useCallback((node, ctx, globalScale) => {
        // Skip hidden nodes
        if (node._visibleDraw === false) return;

        // Viewport culling - skip nodes outside visible area (uses pre-computed bounds)
        const vp = viewportRef.current;
        if (vp.valid && (node.x < vp.minX || node.x > vp.maxX || node.y < vp.minY || node.y > vp.maxY)) {
            return; // Skip off-screen nodes
        }

        const perfEnabled = perfRef.current.enabled;
        const sampleStride = perfRef.current.nodeSampleStride || 10;
        perfRef.current.frameNodeCalls += 1;
        const shouldSample = perfEnabled && (perfRef.current.frameNodeCalls % sampleStride === 0);
        const t0 = shouldSample ? performance.now() : 0;

        const hovered = hoveredNodeRef.current;
        const isHovered = hovered === node;
        const nodeType = node.pz_type || node.type;
        const settings = getNodeTypeSettings(nodeType, vizConfig);
        const shape = node._shape || 'circle';
        
        const baseRadius = node.val * (vizConfig?.nodeRadiusScale ?? 1.0);
        const r = isHovered ? baseRadius * 1.3 : baseRadius;

        const shouldDim = !!(vizConfig?.hoverHighlight && hovered);
        const neighbors = shouldDim ? neighborMap?.get(hovered?.id) : null;
        const isNeighbor = !shouldDim || (neighbors && neighbors.has(node.id));
        const prevAlpha = ctx.globalAlpha;
        if (shouldDim && !isNeighbor) {
            ctx.globalAlpha = 0.08;
        }
        
        const isDocument = nodeType === 'document';
        const isImportant = node.isEvidence || node.isOnPath || node.isActivePath || isDocument;
        
        // Apply per-type node opacity
        const typeOpacity = settings.opacity ?? 1.0;
        if (typeOpacity < 1.0) {
            ctx.globalAlpha = prevAlpha * typeOpacity;
        }
        
        // Draw outer glow for important nodes (skip on large graphs)
        const glowStart = shouldSample ? performance.now() : 0;
        if (!lowQuality && isImportant && !shouldDim) {
            const gradient = ctx.createRadialGradient(node.x, node.y, r * 0.5, node.x, node.y, r * 2.5);
            gradient.addColorStop(0, withAlpha(node.color, 0.4));
            gradient.addColorStop(1, withAlpha(node.color, 0));
            ctx.beginPath();
            ctx.arc(node.x, node.y, r * 2.5, 0, 2 * Math.PI, false);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Draw thinking pulse
        if (node.isThinking) {
             const time = performance.now();
             const pulse = (Math.sin(time / 200) + 1) / 2; // 0 to 1
             const pulseRadius = r * (1.5 + pulse * 0.8); 
             
             ctx.beginPath();
             ctx.arc(node.x, node.y, pulseRadius, 0, 2 * Math.PI, false);
             ctx.fillStyle = withAlpha(node.color || '#3b82f6', 0.5 * (1 - pulse));
             ctx.fill();
        }

        if (shouldSample) {
            const dtGlow = performance.now() - glowStart;
            perfRef.current.frameNodeGlowMsEst += dtGlow * sampleStride;
        }
        
        // Draw node border based on config
        const nodeBorder = vizConfig?.nodeBorder ?? 'none';
        const borderWidth = nodeBorder === 'thick' ? 2 : nodeBorder === 'medium' ? 1.2 : nodeBorder === 'thin' ? 0.6 : 0;
        
        const borderStart = shouldSample ? performance.now() : 0;
        if (!lowQuality && (borderWidth > 0 || isDocument)) {
            drawShape(ctx, node.x, node.y, r + borderWidth/2, shape);
            ctx.strokeStyle = withAlpha('#ffffff', 0.3);
            ctx.lineWidth = (borderWidth || 1.5) / globalScale;
            ctx.stroke();
        }
        if (shouldSample) {
            const dtBorder = performance.now() - borderStart;
            perfRef.current.frameNodeBorderMsEst += dtBorder * sampleStride;
        }

        // Draw main node shape
        drawShape(ctx, node.x, node.y, r, shape);
        
        if (isHovered) {
            // White fill with colored border on hover
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            ctx.strokeStyle = node.color;
            ctx.lineWidth = 2 / globalScale;
            ctx.stroke();
        } else {
            ctx.fillStyle = node.color;
            ctx.fill();
        }

        // Draw label
        const showLabel = vizConfig?.showLabels;
        const labelPolicy = vizConfig?.labelPolicy || 'important';
        
        let shouldDrawLabel = false;
        if (showLabel) {
            if (isHovered) shouldDrawLabel = true;
            else if (labelPolicy === 'all' && globalScale > 1.2) shouldDrawLabel = true;
            else if (labelPolicy === 'important' && globalScale > 1.5) {
                shouldDrawLabel = isImportant || nodeType === 'entry';
            }
        }

        const labelStart = shouldSample ? performance.now() : 0;
        if (shouldDrawLabel) {
            let label = node.label || '';
            const maxLen = vizConfig?.labelMaxLength ?? 20;
            if (label.length > maxLen) {
                label = label.substring(0, maxLen) + '…';
            }
            const baseFontSize = vizConfig?.labelFontSize ?? 10;
            const fontSize = baseFontSize / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = isHovered ? '#ffffff' : '#d0d0d0';
            ctx.fillText(label, node.x, node.y + r + (4 / globalScale));
        }
        if (shouldSample) {
            const dtLabel = performance.now() - labelStart;
            perfRef.current.frameNodeLabelMsEst += dtLabel * sampleStride;
        }

        if (shouldSample) {
            const dtTotal = performance.now() - t0;
            perfRef.current.frameNodeMsEst += dtTotal * sampleStride;
        }

        ctx.globalAlpha = prevAlpha;
    }, [vizConfig, neighborMap, drawShape, lowQuality]);

    const requestRedraw = useCallback(() => {
        const fg = fgRef.current;
        if (!fg) return;

        // Preferred: ask force-graph to redraw without touching the simulation.
        if (typeof fg.refresh === 'function') {
            try {
                fg.refresh();
            } catch (_) {
                // ignore
            }
            return;
        }

        // Fallback: if layout is paused and redraw is auto-paused, briefly resume animation for a frame.
        const engineRunning = typeof fg.isEngineRunning === 'function' ? fg.isEngineRunning() : false;
        if (engineRunning || vizConfig?.runLayout) return;

        if (typeof fg.resumeAnimation === 'function' && typeof fg.pauseAnimation === 'function') {
            try {
                fg.resumeAnimation();
                if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
                    window.requestAnimationFrame(() => {
                        try {
                            fg.pauseAnimation();
                        } catch (_) {
                            // ignore
                        }
                    });
                } else {
                    setTimeout(() => {
                        try {
                            fg.pauseAnimation();
                        } catch (_) {
                            // ignore
                        }
                    }, 0);
                }
            } catch (_) {
                // ignore
            }
        }
    }, [vizConfig?.runLayout]);

    // Perf instrumentation (render timing + engine tick rate)
    useEffect(() => {
        const now = performance.now();
        perfRef.current.enabled = !!vizConfig?.perfHud;
        if (perfRef.current.enabled && perfRef.current.lastReport === 0) {
            perfRef.current.lastReport = now;
            perfRef.current.lastTickReport = now;
        }
        if (!perfRef.current.enabled) {
            setPerfStats(null);
        }
    }, [vizConfig?.perfHud]);

    // Viewport bounds ref - computed once per frame for culling
    const viewportRef = useRef({ minX: -Infinity, minY: -Infinity, maxX: Infinity, maxY: Infinity, valid: false });

    const onRenderFramePre = useCallback((ctx, globalScale) => {
        const vpStart = performance.now();
        // Compute viewport bounds ONCE per frame for culling (not per-node!)
        try {
            const transform = ctx.getTransform();
            const invScale = 1 / Math.max(transform.a, 0.001);
            const canvasW = ctx.canvas.width / (window.devicePixelRatio || 1);
            const canvasH = ctx.canvas.height / (window.devicePixelRatio || 1);
            const pad = 50 * invScale;
            viewportRef.current = {
                minX: -transform.e * invScale - pad,
                minY: -transform.f * invScale - pad,
                maxX: -transform.e * invScale + canvasW * invScale + pad,
                maxY: -transform.f * invScale + canvasH * invScale + pad,
                valid: true,
            };
        } catch (_) {
            viewportRef.current.valid = false;
        }
        const vpEnd = performance.now();

        if (!perfRef.current.enabled) return;
        perfRef.current.frameStart = performance.now();
        perfRef.current.lastScale = globalScale;
        perfRef.current.frameViewportCalcMs = vpEnd - vpStart;
        perfRef.current.phaseViewportMs = vpEnd - vpStart;

        // Reset per-frame breakdown
        perfRef.current.frameNodeCalls = 0;
        perfRef.current.frameNodeMsEst = 0;
        perfRef.current.frameNodeGlowMsEst = 0;
        perfRef.current.frameNodeBorderMsEst = 0;
        perfRef.current.frameNodeLabelMsEst = 0;
        perfRef.current.frameLinkCbCalls = 0;
        perfRef.current.frameLinkCbMsEst = 0;
        perfRef.current.frameLinkColorMsEst = 0;
        perfRef.current.frameLinkWidthMsEst = 0;
        perfRef.current.frameLinkDashMsEst = 0;
        perfRef.current.frameLinkArrowMsEst = 0;
        perfRef.current.frameCustomLinkDrawMs = 0;
        perfRef.current.framePointerCalls = 0;
        perfRef.current.framePointerMsEst = 0;
        perfRef.current.frameForceTickMs = perfRef.current.forceTickMs;
        perfRef.current.forceTickMs = 0; // Reset for next tick
        
        // Reset phase counters
        perfRef.current.phaseLinkCullMs = 0;
        perfRef.current.phaseLinkBatchMs = 0;
        perfRef.current.phaseLinkDrawMs = 0;
        perfRef.current.phaseNodeIterMs = 0;
        perfRef.current.phaseNodeDrawMs = 0;
        perfRef.current.phaseNodeCullCount = 0;
        perfRef.current.phaseNodeDrawCount = 0;
        perfRef.current.phaseLinkCullCount = 0;
        perfRef.current.phaseLinkDrawCount = 0;
    }, []);

    // Batched link renderer for large graphs.
    // The default force-graph renderer draws links one-by-one; for ~100k edges this is prohibitively slow.
    // We disable built-in link rendering and draw links once per frame using canvas path batching.
    const useCustomLinkRender = useMemo(() => {
        const edges = data.allLinks?.length || data.links?.length || 0;
        // Threshold chosen to avoid over-optimizing small graphs.
        return edges >= 20000;
    }, [data.allLinks?.length, data.links?.length]);

    const customLinkRenderData = useMemo(() => {
        if (!useCustomLinkRender) return null;

        const mkKey = (color, width, dash) => {
            const dashKey = Array.isArray(dash) ? dash.join(',') : '';
            return `${color}|${width}|${dashKey}`;
        };

        const pushBucket = (buckets, key, color, width, dash, link) => {
            let b = buckets.get(key);
            if (!b) {
                b = { color, width, dash, links: [] };
                buckets.set(key, b);
            }
            b.links.push(link);
        };

        const bucketsNormal = new Map();
        const bucketsDim = new Map();
        const pathLinks = [];
        const incidentLinksByNode = new Map();

        const addIncident = (nodeId, link) => {
            if (nodeId === undefined || nodeId === null) return;
            let arr = incidentLinksByNode.get(nodeId);
            if (!arr) {
                arr = [];
                incidentLinksByNode.set(nodeId, arr);
            }
            arr.push(link);
        };

        // Use allLinks (the full edge set) for rendering, not the filtered links
        // passed to ForceGraph2D (which may be empty on huge graphs).
        const linksToRender = data.allLinks || data.links || [];

        for (const link of linksToRender) {
            if (link._visibleDraw === false) continue;
            const isPath = !!(link.isOnPath || link.isActivePath);
            if (isPath) {
                pathLinks.push(link);
            } else {
                const keyN = mkKey(link._colorDraw, link._widthDraw, link._dash);
                pushBucket(bucketsNormal, keyN, link._colorDraw, link._widthDraw, link._dash, link);

                const keyD = mkKey(link._colorDim, link._widthDraw, link._dash);
                pushBucket(bucketsDim, keyD, link._colorDim, link._widthDraw, link._dash, link);
            }

            // Precompute incident edge lists for hover highlighting.
            addIncident(link._sid, link);
            addIncident(link._tid, link);
        }

        return { bucketsNormal, bucketsDim, pathLinks, incidentLinksByNode };
        // IMPORTANT: `data.links` array identity is intentionally stable (we mutate links in-place
        // when vizConfig changes to avoid restarting the simulation). That means we *must* also
        // depend on vizConfig fields that can change link render props (visibility/color/width/etc),
        // otherwise the cached buckets will not update and edge hide/show will appear broken.
    }, [
        useCustomLinkRender,
        data.allLinks,
        data.links,
        data.nodes,
        vizConfig?.edgeTypeSettings,
        vizConfig?.edgeWidth,
        vizConfig?.dimOpacity,
        vizConfig?.pathOpacity,
        vizConfig?.stateColors,
    ]);

    const drawLinksBatched = useCallback((ctx, globalScale) => {
        if (!useCustomLinkRender || !customLinkRenderData) return;
        const fg = fgRef.current;
        if (!fg) return;

        const perfEnabled = perfRef.current.enabled;
        const t0 = perfEnabled ? performance.now() : 0;

        // Important: the canvas context provided by ForceGraph2D is already
        // transformed (pan/zoom) into graph coordinates. Draw using node.x/y
        // directly. Converting to screen coords here would double-apply the
        // transform and misalign edges.

        // Use pre-computed viewport bounds from onRenderFramePre
        const vp = viewportRef.current;
        const doCulling = vp.valid;

        // Quick check if a line segment intersects the viewport (AABB test)
        const isLineVisible = (x1, y1, x2, y2) => {
            if (!doCulling) return true;
            // Line AABB
            const minX = Math.min(x1, x2);
            const maxX = Math.max(x1, x2);
            const minY = Math.min(y1, y2);
            const maxY = Math.max(y1, y2);
            // AABB intersection test
            return maxX >= vp.minX && minX <= vp.maxX && maxY >= vp.minY && minY <= vp.maxY;
        };

        const nodeById = new Map(data.nodes.map(n => [n.id, n]));
        const resolveNode = (v) => {
            if (typeof v === 'object' && v !== null) return v;
            return nodeById.get(v);
        };

        const setDash = (dash) => {
            if (Array.isArray(dash) && dash.length) {
                ctx.setLineDash(dash.map(d => d / globalScale));
            } else {
                ctx.setLineDash([]);
            }
        };

        // Track culled links for perf HUD
        let culledCount = 0;
        let drawnCount = 0;
        
        // Detailed timing
        let cullTimeMs = 0;
        let drawTimeMs = 0;

        const strokeBucket = (bucket) => {
            ctx.strokeStyle = bucket.color;
            ctx.lineWidth = bucket.width / globalScale;
            setDash(bucket.dash);
            ctx.beginPath();
            
            const tCullStart = perfEnabled ? performance.now() : 0;
            const visibleLinks = [];
            for (const link of bucket.links) {
                const sN = resolveNode(link.source);
                const tN = resolveNode(link.target);
                if (!sN || !tN) continue;
                const x1 = sN.x || 0;
                const y1 = sN.y || 0;
                const x2 = tN.x || 0;
                const y2 = tN.y || 0;
                // Viewport culling
                if (!isLineVisible(x1, y1, x2, y2)) {
                    culledCount++;
                    continue;
                }
                drawnCount++;
                visibleLinks.push({ x1, y1, x2, y2 });
            }
            if (perfEnabled) cullTimeMs += performance.now() - tCullStart;
            
            const tDrawStart = perfEnabled ? performance.now() : 0;
            for (const { x1, y1, x2, y2 } of visibleLinks) {
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
            }
            ctx.stroke();
            if (perfEnabled) drawTimeMs += performance.now() - tDrawStart;
        };

        // Path edges: draw last for clarity.
        const drawPathLinks = () => {
            if (!customLinkRenderData.pathLinks.length) return;
            const buckets = new Map();
            const mkKey = (color, width, dash) => {
                const dashKey = Array.isArray(dash) ? dash.join(',') : '';
                return `${color}|${width}|${dashKey}`;
            };
            for (const link of customLinkRenderData.pathLinks) {
                const key = mkKey(link._colorPath, link._widthDraw, link._dash);
                let b = buckets.get(key);
                if (!b) {
                    b = { color: link._colorPath, width: link._widthDraw, dash: link._dash, links: [] };
                    buckets.set(key, b);
                }
                b.links.push(link);
            }
            for (const b of buckets.values()) {
                strokeBucket(b);
            }
        };

        ctx.save();
        ctx.globalAlpha = 1;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const hovered = hoveredNodeRef.current;
        const doHoverDim = !!(vizConfig?.hoverHighlight && hovered);
        if (doHoverDim) {
            // Dimmed base layer
            for (const b of customLinkRenderData.bucketsDim.values()) {
                strokeBucket(b);
            }

            // Re-highlight incident edges
            const hid = hovered.id;
            const incident = customLinkRenderData.incidentLinksByNode.get(hid) || [];
            if (incident.length) {
                const buckets = new Map();
                const mkKey = (color, width, dash) => {
                    const dashKey = Array.isArray(dash) ? dash.join(',') : '';
                    return `${color}|${width}|${dashKey}`;
                };
                for (const link of incident) {
                    if (link.isOnPath || link.isActivePath) continue; // will be drawn in path pass
                    const key = mkKey(link._colorIncident, link._widthDraw, link._dash);
                    let b = buckets.get(key);
                    if (!b) {
                        b = { color: link._colorIncident, width: link._widthDraw, dash: link._dash, links: [] };
                        buckets.set(key, b);
                    }
                    b.links.push(link);
                }
                for (const b of buckets.values()) {
                    strokeBucket(b);
                }
            }
        } else {
            for (const b of customLinkRenderData.bucketsNormal.values()) {
                strokeBucket(b);
            }
        }

        drawPathLinks();

        // Store culling stats for perf HUD
        if (perfEnabled) {
            perfRef.current.linksCulled = culledCount;
            perfRef.current.linksDrawn = drawnCount;
            perfRef.current.phaseLinkCullMs = cullTimeMs;
            perfRef.current.phaseLinkDrawMs = drawTimeMs;
            perfRef.current.phaseLinkCullCount = culledCount;
            perfRef.current.phaseLinkDrawCount = drawnCount;
            perfRef.current.frameCustomLinkDrawMs = performance.now() - t0;
        }

        // Reset dash to avoid affecting node drawing
        ctx.setLineDash([]);
        ctx.restore();
    }, [
        useCustomLinkRender,
        customLinkRenderData,
        data.nodes,
        vizConfig?.hoverHighlight,
    ]);

    const onRenderFramePost = useCallback((_ctx, globalScale) => {
        if (!perfRef.current.enabled) return;
        const now = performance.now();
        const dt = now - (perfRef.current.frameStart || now);
        perfRef.current.frames += 1;
        perfRef.current.frameMsSum += dt;
        perfRef.current.frameMsMax = Math.max(perfRef.current.frameMsMax, dt);

        // Accumulate breakdown window sums
        perfRef.current.nodeCallsSum += perfRef.current.frameNodeCalls;
        perfRef.current.pointerCallsSum += perfRef.current.framePointerCalls;
        perfRef.current.nodeMsSum += perfRef.current.frameNodeMsEst;
        perfRef.current.nodeGlowMsSum += perfRef.current.frameNodeGlowMsEst;
        perfRef.current.nodeBorderMsSum += perfRef.current.frameNodeBorderMsEst;
        perfRef.current.nodeLabelMsSum += perfRef.current.frameNodeLabelMsEst;
        perfRef.current.pointerMsSum += perfRef.current.framePointerMsEst;
        perfRef.current.linkCbMsSum += perfRef.current.frameLinkCbMsEst;
        perfRef.current.linkColorMsSum += perfRef.current.frameLinkColorMsEst;
        perfRef.current.linkWidthMsSum += perfRef.current.frameLinkWidthMsEst;
        perfRef.current.linkDashMsSum += perfRef.current.frameLinkDashMsEst;
        perfRef.current.linkArrowMsSum += perfRef.current.frameLinkArrowMsEst;
        perfRef.current.customLinkDrawMsSum += perfRef.current.frameCustomLinkDrawMs;
        perfRef.current.forceTickMsSum += perfRef.current.frameForceTickMs || 0;
        
        // Accumulate phase sums
        perfRef.current.phaseViewportMsSum += perfRef.current.phaseViewportMs || 0;
        perfRef.current.phaseLinkCullMsSum += perfRef.current.phaseLinkCullMs || 0;
        perfRef.current.phaseLinkDrawMsSum += perfRef.current.phaseLinkDrawMs || 0;
        perfRef.current.phaseLinkCullCountSum += perfRef.current.phaseLinkCullCount || 0;
        perfRef.current.phaseLinkDrawCountSum += perfRef.current.phaseLinkDrawCount || 0;
        perfRef.current.phaseNodeCullCountSum += perfRef.current.phaseNodeCullCount || 0;
        perfRef.current.phaseNodeDrawCountSum += perfRef.current.phaseNodeDrawCount || 0;

        // Track what we've measured vs total frame time
        const measuredMs = 
            perfRef.current.frameNodeMsEst +
            perfRef.current.frameLinkCbMsEst +
            (perfRef.current.frameCustomLinkDrawMs || 0) +
            (perfRef.current.framePointerMsEst || 0) +
            (perfRef.current.frameViewportCalcMs || 0);
        
        // "Library overhead" is time spent inside react-force-graph that we can't instrument
        // This includes: iterating nodes/links arrays, internal state updates, canvas setup
        const libraryOverhead = Math.max(0, dt - measuredMs);
        perfRef.current.libraryOverheadSum = (perfRef.current.libraryOverheadSum || 0) + libraryOverhead;

        // "Other" is time spent outside our measured node/link callbacks.
        // IMPORTANT: when `useCustomLinkRender` is enabled we disable the library's built-in
        // link renderer, so any remaining time is NOT "link draw".
        const inferredOther = Math.max(
            0,
            dt
                - perfRef.current.frameNodeMsEst
                - perfRef.current.frameLinkCbMsEst
                - (perfRef.current.frameCustomLinkDrawMs || 0)
                - (perfRef.current.framePointerMsEst || 0)
        );
        perfRef.current.inferredLinkDrawMsSum += inferredOther;

        const fg = fgRef.current;
        const engineRunning = !!(fg && typeof fg.isEngineRunning === 'function' ? fg.isEngineRunning() : false);
        if (engineRunning) {
            perfRef.current.runningFrames += 1;
            perfRef.current.runningMsSum += dt;
        } else {
            perfRef.current.idleFrames += 1;
            perfRef.current.idleMsSum += dt;
        }

        // Update HUD ~2x/second to keep React overhead low
        const elapsed = now - (perfRef.current.lastReport || now);
        if (elapsed < 500) return;

        const fps = perfRef.current.frames > 0 ? (perfRef.current.frames * 1000) / elapsed : 0;
        const frameMsAvg = perfRef.current.frames > 0 ? (perfRef.current.frameMsSum / perfRef.current.frames) : 0;
        const runningMsAvg = perfRef.current.runningFrames > 0 ? (perfRef.current.runningMsSum / perfRef.current.runningFrames) : null;
        const idleMsAvg = perfRef.current.idleFrames > 0 ? (perfRef.current.idleMsSum / perfRef.current.idleFrames) : null;
        const frameMsMax = perfRef.current.frameMsMax;

        const nodeMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.nodeMsSum / perfRef.current.frames) : 0;
        const nodeGlowMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.nodeGlowMsSum / perfRef.current.frames) : 0;
        const nodeBorderMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.nodeBorderMsSum / perfRef.current.frames) : 0;
        const nodeLabelMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.nodeLabelMsSum / perfRef.current.frames) : 0;
        const pointerMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.pointerMsSum / perfRef.current.frames) : 0;
        const inferredOtherMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.inferredLinkDrawMsSum / perfRef.current.frames) : 0;

        const nodeCallsAvg = perfRef.current.frames > 0 ? (perfRef.current.nodeCallsSum / perfRef.current.frames) : 0;
        const pointerCallsAvg = perfRef.current.frames > 0 ? (perfRef.current.pointerCallsSum / perfRef.current.frames) : 0;

        const linkCbMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.linkCbMsSum / perfRef.current.frames) : 0;
        const linkColorMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.linkColorMsSum / perfRef.current.frames) : 0;
        const linkWidthMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.linkWidthMsSum / perfRef.current.frames) : 0;
        const linkDashMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.linkDashMsSum / perfRef.current.frames) : 0;
        const linkArrowMsAvgEst = perfRef.current.frames > 0 ? (perfRef.current.linkArrowMsSum / perfRef.current.frames) : 0;
        const customLinkDrawMsAvg = perfRef.current.frames > 0 ? (perfRef.current.customLinkDrawMsSum / perfRef.current.frames) : 0;

        // Render-visible link count (separate from total links, which still affect force)
        let visibleLinks = 0;
        for (const l of data.links || []) {
            if (l?._visibleDraw === false) continue;
            visibleLinks += 1;
        }

        // Force tick timing
        const forceTickMsAvg = perfRef.current.frames > 0 ? (perfRef.current.forceTickMsSum / perfRef.current.frames) : 0;
        const libraryOverheadMsAvg = perfRef.current.frames > 0 ? ((perfRef.current.libraryOverheadSum || 0) / perfRef.current.frames) : 0;
        
        // Phase timing averages
        const phaseViewportMsAvg = perfRef.current.frames > 0 ? (perfRef.current.phaseViewportMsSum / perfRef.current.frames) : 0;
        const phaseLinkCullMsAvg = perfRef.current.frames > 0 ? (perfRef.current.phaseLinkCullMsSum / perfRef.current.frames) : 0;
        const phaseLinkDrawMsAvg = perfRef.current.frames > 0 ? (perfRef.current.phaseLinkDrawMsSum / perfRef.current.frames) : 0;
        const phaseLinkCullCountAvg = perfRef.current.frames > 0 ? Math.round(perfRef.current.phaseLinkCullCountSum / perfRef.current.frames) : 0;
        const phaseLinkDrawCountAvg = perfRef.current.frames > 0 ? Math.round(perfRef.current.phaseLinkDrawCountSum / perfRef.current.frames) : 0;
        const phaseNodeCullCountAvg = perfRef.current.frames > 0 ? Math.round(perfRef.current.phaseNodeCullCountSum / perfRef.current.frames) : 0;
        const phaseNodeDrawCountAvg = perfRef.current.frames > 0 ? Math.round(perfRef.current.phaseNodeDrawCountSum / perfRef.current.frames) : 0;

        // Heuristic: compare idle vs running averages
        let likelyBottleneck = 'unknown';
        if (idleMsAvg !== null && idleMsAvg > 18) likelyBottleneck = 'render';
        if (runningMsAvg !== null && idleMsAvg !== null && (runningMsAvg - idleMsAvg) > 10) likelyBottleneck = 'force';

        // If render-bound, guess whether nodes, custom link draw, or other dominates.
        if (likelyBottleneck === 'render') {
            const linkDrawMs = customLinkDrawMsAvg;
            if (linkDrawMs > Math.max(nodeMsAvgEst, inferredOtherMsAvgEst) * 1.2) likelyBottleneck = 'render:links';
            else if (nodeMsAvgEst > Math.max(linkDrawMs, inferredOtherMsAvgEst) * 1.2) likelyBottleneck = 'render:nodes';
            else if (inferredOtherMsAvgEst > Math.max(nodeMsAvgEst, linkDrawMs) * 1.2) likelyBottleneck = 'render:lib';
        }

        const totalLinksCount = (data.allLinks?.length ?? data.links?.length ?? 0);
        
        // Calculate "unaccounted" time - time not attributed to any measured phase
        const accountedMs = nodeMsAvgEst + customLinkDrawMsAvg + pointerMsAvgEst + phaseViewportMsAvg;
        const unaccountedMs = Math.max(0, frameMsAvg - accountedMs);

        setPerfStats({
            fps,
            frameMsAvg,
            frameMsMax,
            tickHz: perfRef.current.tickHz,
            engineRunning,
            runningMsAvg,
            idleMsAvg,
            likelyBottleneck,
            zoom: globalScale,
            nodes: data.nodes?.length || 0,
            links: totalLinksCount,
            visibleLinks,

            // Breakdown (estimated; sampling-based)
            nodeMsAvgEst,
            nodeGlowMsAvgEst,
            nodeBorderMsAvgEst,
            nodeLabelMsAvgEst,
            pointerMsAvgEst,
            inferredOtherMsAvgEst,
            libraryOverheadMsAvg,
            forceTickMsAvg,
            nodeCallsAvg,
            pointerCallsAvg,
            linkCbMsAvgEst,
            linkColorMsAvgEst,
            linkWidthMsAvgEst,
            linkDashMsAvgEst,
            linkArrowMsAvgEst,
            customLinkDrawMsAvg,
            nodeSampleStride: perfRef.current.nodeSampleStride,
            linkSampleStride: perfRef.current.linkSampleStride,
            linksCulled: perfRef.current.linksCulled || 0,
            linksDrawn: perfRef.current.linksDrawn || 0,
            
            // Detailed phase breakdown
            phaseViewportMsAvg,
            phaseLinkCullMsAvg,
            phaseLinkDrawMsAvg,
            phaseLinkCullCountAvg,
            phaseLinkDrawCountAvg,
            phaseNodeCullCountAvg,
            phaseNodeDrawCountAvg,
            unaccountedMs,
            accountedMs,
        });

        // Reset accumulators
        perfRef.current.lastReport = now;
        perfRef.current.frames = 0;
        perfRef.current.frameMsSum = 0;
        perfRef.current.frameMsMax = 0;
        perfRef.current.runningFrames = 0;
        perfRef.current.runningMsSum = 0;
        perfRef.current.idleFrames = 0;
        perfRef.current.idleMsSum = 0;
        perfRef.current.lastScale = globalScale;

        // Reset breakdown window sums
        perfRef.current.nodeCallsSum = 0;
        perfRef.current.pointerCallsSum = 0;
        perfRef.current.nodeMsSum = 0;
        perfRef.current.nodeGlowMsSum = 0;
        perfRef.current.nodeBorderMsSum = 0;
        perfRef.current.nodeLabelMsSum = 0;
        perfRef.current.pointerMsSum = 0;
        perfRef.current.linkCbMsSum = 0;
        perfRef.current.linkColorMsSum = 0;
        perfRef.current.linkWidthMsSum = 0;
        perfRef.current.linkDashMsSum = 0;
        perfRef.current.linkArrowMsSum = 0;
        perfRef.current.customLinkDrawMsSum = 0;
        perfRef.current.inferredLinkDrawMsSum = 0;
        perfRef.current.forceTickMsSum = 0;
        perfRef.current.libraryOverheadSum = 0;
        
        // Reset phase sums
        perfRef.current.phaseViewportMsSum = 0;
        perfRef.current.phaseLinkCullMsSum = 0;
        perfRef.current.phaseLinkDrawMsSum = 0;
        perfRef.current.phaseLinkCullCountSum = 0;
        perfRef.current.phaseLinkDrawCountSum = 0;
        perfRef.current.phaseNodeCullCountSum = 0;
        perfRef.current.phaseNodeDrawCountSum = 0;
    }, [data.nodes?.length, data.links?.length, data.allLinks?.length]);

    const onRenderFramePreWithLinks = useCallback((ctx, globalScale) => {
        onRenderFramePre(ctx, globalScale);
        // Draw links before the internal tickFrame draws nodes.
        if (useCustomLinkRender) {
            if (perfRef.current.enabled) {
                const t0 = performance.now();
                drawLinksBatched(ctx, globalScale);
                perfRef.current.frameCustomLinkDrawMs = performance.now() - t0;
            } else {
                drawLinksBatched(ctx, globalScale);
            }
        }
    }, [onRenderFramePre, useCustomLinkRender, drawLinksBatched]);

    const paintNodePointerArea = useCallback((node, color, ctx) => {
        if (node._visibleDraw === false) return;

        const perfEnabled = perfRef.current.enabled;
        const sampleStride = perfRef.current.nodeSampleStride || 10;
        perfRef.current.framePointerCalls += 1;
        const shouldSample = perfEnabled && (perfRef.current.framePointerCalls % sampleStride === 0);
        const t0 = shouldSample ? performance.now() : 0;

        const r = (node.val || 3) * (vizConfig?.nodeRadiusScale ?? 1.0);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 2, 0, 2 * Math.PI, false);
        ctx.fill();

        if (shouldSample) {
            const dt = performance.now() - t0;
            perfRef.current.framePointerMsEst += dt * sampleStride;
        }
    }, [vizConfig?.nodeRadiusScale]);

    const onEngineTick = useCallback(() => {
        const now = performance.now();
        // Track time since last tick - this gives us per-tick simulation time
        if (perfRef.current.lastForceTickEnd > 0) {
            const sinceLastTick = now - perfRef.current.lastForceTickEnd;
            // If this tick came quickly after the last one, we can infer tick duration
            if (sinceLastTick < 50) {
                perfRef.current.forceTickMs = sinceLastTick;
            }
        }
        perfRef.current.lastForceTickEnd = now;

        if (!perfRef.current.enabled) return;
        perfRef.current.tickCount += 1;
        const elapsed = now - (perfRef.current.lastTickReport || now);
        if (elapsed < 500) return;
        perfRef.current.tickHz = (perfRef.current.tickCount * 1000) / elapsed;
        perfRef.current.tickCount = 0;
        perfRef.current.lastTickReport = now;
    }, []);

    // Interaction handlers
    const handleNodeHover = useCallback((node) => {
        hoveredNodeRef.current = node || null;
        setHoveredNode(node || null);
        if (node && fgRef.current && containerRef.current) {
            const { x, y } = fgRef.current.graph2ScreenCoords(node.x, node.y);
            const rect = containerRef.current.getBoundingClientRect();
            onNodeHover?.({ pageX: rect.left + x, pageY: rect.top + y }, node);
        } else {
            onNodeHover?.(null, null);
        }
        if (containerRef.current) {
            containerRef.current.style.cursor = node ? 'pointer' : null;
        }
        requestRedraw();
    }, [onNodeHover, requestRedraw]);

    // When layout is paused, ForceGraph may auto-pause redraws. Ensure style/data changes repaint.
    // We use structuralData here (not vizConfig) to avoid triggering on every setting change.
    useEffect(() => {
        if (!fgRef.current) return;
        if (!vizConfig?.runLayout) {
            requestRedraw();
        }
    }, [requestRedraw, vizConfig?.runLayout, structuralData, refreshNonce]);

    const handleZoom = (factor) => {
        if (!fgRef.current) return;
        const currentZoom = fgRef.current.zoom();
        fgRef.current.zoom(currentZoom * factor, 400);
    };

    const handleReset = () => {
        if (!fgRef.current) return;
        fgRef.current.zoomToFit(400, 40);
    };

    const handleSearchSelect = (node) => {
        onNodeClick?.(node);
        if (!node || !fgRef.current) return;
        
        // Find the node in the current data to get its x/y
        // Note: react-force-graph mutates the data prop with x/y
        const graphNode = data.nodes.find(n => n.id === node.id);
        if (graphNode && typeof graphNode.x === 'number') {
            fgRef.current.centerAt(graphNode.x, graphNode.y, 1000);
            fgRef.current.zoom(2.5, 1000);
        }
    };

    // Rendering performance: on HiDPI displays the canvas can be 4x the pixels.
    // For very large graphs this becomes the dominant bottleneck, so we clamp DPR.
    const pixelRatio = useMemo(() => {
        const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1;
        const edgeCount = data.links?.length || 0;
        // Clamp harder for very large edge counts.
        if (edgeCount >= 50000) return 1;
        return Math.min(2, dpr);
    }, [data.links?.length]);

    // Avoid per-link function overhead when curvature is constant.
    const linkCurvature = vizConfig?.edgeCurvature ?? 0;

    // Link accessors: prefer string property accessors (fastest) unless we need dynamic dimming.
    const linkColorAccessor = useMemo(() => {
        if (useCustomLinkRender) return '_colorDraw';
        if (vizConfig?.hoverHighlight && hoveredNode) {
            const hid = hoveredNode.id;
            return (link) => {
                if (link.isOnPath || link.isActivePath) return link._colorPath;
                return (link._sid === hid || link._tid === hid) ? link._colorIncident : link._colorDim;
            };
        }
        return '_colorDraw';
    }, [useCustomLinkRender, vizConfig?.hoverHighlight, hoveredNode?.id]);

    const linkVisibilityAccessor = useMemo(() => {
        if (useCustomLinkRender) return false;
        return (link) => link._visibleDraw !== false;
    }, [useCustomLinkRender]);

    return (
        <div ref={containerRef} className="w-full h-full bg-slate-950 relative">
            {dimensions.width > 0 && (
                <ForceGraph2D
                    ref={fgRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    graphData={forceGraphData}
                    pixelRatio={pixelRatio}
                    
                    // Layout
                    dagMode={undefined} // Force only
                    d3VelocityDecay={vizConfig?.d3VelocityDecay ?? 0.3}
                    warmupTicks={0} // Show animation immediately
                    // When layout is paused, stop the engine so we aren't continuously repainting
                    // an expensive static scene.
                    cooldownTicks={vizConfig?.runLayout ? Infinity : 0}
                    cooldownTime={vizConfig?.runLayout ? Infinity : 0}
                    autoPauseRedraw={true}

                    // Perf
                    onRenderFramePre={onRenderFramePreWithLinks}
                    onRenderFramePost={onRenderFramePost}
                    onEngineTick={onEngineTick}

                    
                    // Rendering
                    nodeCanvasObject={paintNode}
                    nodeCanvasObjectMode={() => 'replace'}
                    nodePointerAreaPaint={hugeGraph ? undefined : paintNodePointerArea}
                    enablePointerInteraction={!hugeGraph}
                    enableNodeDrag={!hugeGraph}
                    
                    // Links - use per-type settings from vizConfig
                    linkColor={linkColorAccessor}
                    linkWidth={'_widthDraw'}
                    linkLineDash={'_dash'}
                    linkCurvature={linkCurvature}
                    linkDirectionalArrowLength={'_arrowLen'}
                    linkDirectionalArrowRelPos={1}
                    linkVisibility={linkVisibilityAccessor}
                    
                    // Interaction
                    onNodeClick={onNodeClick}
                    onNodeHover={handleNodeHover}
                    onNodeDrag={(node, translate) => {
                        if (fgRef.current) {
                            // Ensure simulation is active during drag
                            fgRef.current.d3ReheatSimulation();
                        }
                    }}
                    onNodeDragEnd={node => {
                        if (node) {
                            node.fx = node.x;
                            node.fy = node.y;
                        }
                    }}
                    
                    // Config
                    backgroundColor="#0f172a"
                />
            )}

            {vizConfig?.perfHud && perfStats && (
                <div className="absolute top-4 left-4 bg-gray-900/95 p-3 rounded border border-gray-700 backdrop-blur pointer-events-none z-10 max-w-md select-none">
                    <div className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Performance Debug</div>
                    <div className="text-xs text-gray-200 font-mono leading-relaxed">
                        {/* Summary */}
                        <div className="text-yellow-400 font-bold text-sm">
                            {perfStats.fps.toFixed(0)} fps • {perfStats.frameMsAvg.toFixed(1)}ms/frame
                        </div>
                        <div className="text-[10px] text-gray-400 mb-2">
                            max: {perfStats.frameMsMax.toFixed(1)}ms | target: 16.7ms (60fps)
                        </div>
                        
                        {/* Frame Breakdown Bar */}
                        <div className="text-[10px] text-gray-400 uppercase mb-1">Frame Budget</div>
                        <div className="h-4 bg-gray-800 rounded overflow-hidden flex mb-1">
                            {perfStats.nodeMsAvgEst > 0 && (
                                <div 
                                    className="bg-blue-500 h-full" 
                                    style={{ width: `${Math.min(100, (perfStats.nodeMsAvgEst / perfStats.frameMsAvg) * 100)}%` }}
                                    title={`Nodes: ${perfStats.nodeMsAvgEst.toFixed(1)}ms`}
                                />
                            )}
                            {perfStats.customLinkDrawMsAvg > 0 && (
                                <div 
                                    className="bg-green-500 h-full" 
                                    style={{ width: `${Math.min(100, (perfStats.customLinkDrawMsAvg / perfStats.frameMsAvg) * 100)}%` }}
                                    title={`Links: ${perfStats.customLinkDrawMsAvg.toFixed(1)}ms`}
                                />
                            )}
                            {perfStats.libraryOverheadMsAvg > 0 && (
                                <div 
                                    className="bg-red-500 h-full" 
                                    style={{ width: `${Math.min(100, (perfStats.libraryOverheadMsAvg / perfStats.frameMsAvg) * 100)}%` }}
                                    title={`Library: ${perfStats.libraryOverheadMsAvg.toFixed(1)}ms`}
                                />
                            )}
                        </div>
                        <div className="flex text-[9px] text-gray-500 mb-2 gap-3">
                            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-blue-500 rounded-sm"></span>nodes</span>
                            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-green-500 rounded-sm"></span>links</span>
                            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-red-500 rounded-sm"></span>library</span>
                        </div>

                        {/* Detailed Timing */}
                        <div className="border-t border-gray-700 pt-2 mt-1">
                            <div className="text-[10px] text-gray-400 uppercase mb-1">Timing Breakdown (ms)</div>
                            <table className="w-full text-[10px]">
                                <tbody>
                                    <tr className={perfStats.nodeMsAvgEst > 30 ? 'text-orange-400' : ''}>
                                        <td className="pr-2">Nodes</td>
                                        <td className="text-right font-bold">{perfStats.nodeMsAvgEst.toFixed(2)}</td>
                                        <td className="text-gray-500 pl-2">({perfStats.nodeCallsAvg?.toFixed(0) || 0} drawn)</td>
                                    </tr>
                                    <tr className="text-[9px] text-gray-500">
                                        <td className="pl-2">├ glow</td>
                                        <td className="text-right">{perfStats.nodeGlowMsAvgEst.toFixed(2)}</td>
                                        <td></td>
                                    </tr>
                                    <tr className="text-[9px] text-gray-500">
                                        <td className="pl-2">├ border</td>
                                        <td className="text-right">{perfStats.nodeBorderMsAvgEst.toFixed(2)}</td>
                                        <td></td>
                                    </tr>
                                    <tr className="text-[9px] text-gray-500">
                                        <td className="pl-2">└ label</td>
                                        <td className="text-right">{perfStats.nodeLabelMsAvgEst.toFixed(2)}</td>
                                        <td></td>
                                    </tr>
                                    <tr className={perfStats.customLinkDrawMsAvg > 30 ? 'text-orange-400' : ''}>
                                        <td className="pr-2">Links (total)</td>
                                        <td className="text-right font-bold">{perfStats.customLinkDrawMsAvg?.toFixed(2) || '0.00'}</td>
                                        <td className="text-gray-500 pl-2">({perfStats.linksDrawn || 0} drawn)</td>
                                    </tr>
                                    <tr className="text-[9px] text-gray-500">
                                        <td className="pl-2">├ culling</td>
                                        <td className="text-right">{perfStats.phaseLinkCullMsAvg?.toFixed(2) || '0.00'}</td>
                                        <td className="pl-2">({perfStats.linksCulled || 0} culled)</td>
                                    </tr>
                                    <tr className="text-[9px] text-gray-500">
                                        <td className="pl-2">└ drawing</td>
                                        <td className="text-right">{perfStats.phaseLinkDrawMsAvg?.toFixed(2) || '0.00'}</td>
                                        <td></td>
                                    </tr>
                                    <tr className={perfStats.libraryOverheadMsAvg > 50 ? 'text-red-400' : ''}>
                                        <td className="pr-2">Library overhead</td>
                                        <td className="text-right font-bold">{perfStats.libraryOverheadMsAvg?.toFixed(2) || '0.00'}</td>
                                        <td className="text-gray-500 pl-2">(unaccounted)</td>
                                    </tr>
                                    <tr className="text-gray-500">
                                        <td className="pr-2">Viewport calc</td>
                                        <td className="text-right">{perfStats.phaseViewportMsAvg?.toFixed(2) || '0.00'}</td>
                                        <td></td>
                                    </tr>
                                    <tr className="text-gray-500">
                                        <td className="pr-2">Pointer areas</td>
                                        <td className="text-right">{perfStats.pointerMsAvgEst?.toFixed(2) || '0.00'}</td>
                                        <td></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        {/* Physics Engine */}
                        <div className="border-t border-gray-700 pt-2 mt-2">
                            <div className="text-[10px] text-gray-400 uppercase mb-1">Physics Engine</div>
                            <div className="flex justify-between">
                                <span>Tick rate</span>
                                <span className="font-bold">{perfStats.tickHz.toFixed(1)} hz {perfStats.engineRunning ? '🟢' : '⏸️'}</span>
                            </div>
                            {perfStats.runningMsAvg !== null && (
                                <div className="flex justify-between text-[10px] text-gray-400">
                                    <span>While running</span>
                                    <span>{perfStats.runningMsAvg.toFixed(1)}ms/frame</span>
                                </div>
                            )}
                            {perfStats.idleMsAvg !== null && (
                                <div className="flex justify-between text-[10px] text-gray-400">
                                    <span>While idle</span>
                                    <span>{perfStats.idleMsAvg.toFixed(1)}ms/frame</span>
                                </div>
                            )}
                        </div>

                        {/* Graph Stats */}
                        <div className="border-t border-gray-700 pt-2 mt-2">
                            <div className="text-[10px] text-gray-400 uppercase mb-1">Graph Stats</div>
                            <div className="flex justify-between">
                                <span>Nodes</span>
                                <span className="font-bold">{perfStats.nodes.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Edges (total)</span>
                                <span className="font-bold">{perfStats.links.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between text-[10px] text-gray-400">
                                <span>Zoom</span>
                                <span>{perfStats.zoom?.toFixed(2) || '1.00'}x</span>
                            </div>
                        </div>

                        {/* Bottleneck indicator */}
                        <div className="border-t border-gray-700 pt-2 mt-2 flex items-center gap-2">
                            <span className="text-[10px] text-gray-400">BOTTLENECK:</span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                perfStats.likelyBottleneck.includes('lib') ? 'bg-red-900 text-red-200' :
                                perfStats.likelyBottleneck.includes('links') ? 'bg-green-900 text-green-200' :
                                perfStats.likelyBottleneck.includes('nodes') ? 'bg-blue-900 text-blue-200' :
                                perfStats.likelyBottleneck.includes('force') ? 'bg-yellow-900 text-yellow-200' :
                                'bg-gray-700 text-gray-300'
                            }`}>
                                {perfStats.likelyBottleneck}
                            </span>
                        </div>
                        
                        {hugeGraph && (
                            <div className="text-[10px] text-cyan-400 mt-2">
                                ⚡ Interaction disabled (100k+ edges) - use GPU mode for better perf
                            </div>
                        )}
                    </div>
                </div>
            )}

            <div className="absolute top-4 right-4 z-10">
                <GraphSearch nodes={data.nodes} onSelectNode={handleSearchSelect} />
            </div>

            {/* Dynamic Legend based on actual types in the graph */}
            <DynamicLegend data={data} vizConfig={vizConfig} />

            <div className="absolute bottom-20 right-4 flex flex-col gap-2 z-10">
                <button onClick={() => handleZoom(1.2)} className="p-2 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 hover:text-white shadow-lg border border-slate-700">
                    <ZoomIn size={16} />
                </button>
                <button onClick={() => handleZoom(0.8)} className="p-2 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 hover:text-white shadow-lg border border-slate-700">
                    <ZoomOut size={16} />
                </button>
                <button onClick={handleReset} className="p-2 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 hover:text-white shadow-lg border border-slate-700">
                    <Maximize size={16} />
                </button>
            </div>
        </div>
    );
}

export default GraphVisualization;
