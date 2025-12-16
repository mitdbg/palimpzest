/**
 * GraphVisualization3D - GPU-accelerated 2D graph visualization using Three.js/WebGL
 * 
 * Uses react-force-graph-3d with numDimensions=2 for:
 * - WebGL rendering (GPU accelerated)
 * - 2D force layout (nodes stay on z=0 plane)
 * - Top-down orthographic-like view
 */
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as d3 from 'd3';
import * as THREE from 'three';
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react';
import GraphSearch from './GraphSearch';
import { 
    DEFAULT_NODE_SETTINGS, 
    DEFAULT_EDGE_SETTINGS, 
    DEFAULT_NODE_FALLBACK, 
    DEFAULT_EDGE_FALLBACK 
} from './TypeColorEditor';

// ==================== WEBGL DETECTION ====================
/**
 * Check if WebGL is available and not exhausted.
 * Returns { supported: boolean, reason?: string }
 */
function checkWebGLAvailability() {
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (!gl) {
            // Try to determine why
            const testCanvas = document.createElement('canvas');
            // Check if WebGL is supported at all
            if (!window.WebGLRenderingContext) {
                return { supported: false, reason: 'WebGL is not supported by your browser.' };
            }
            return { supported: false, reason: 'WebGL contexts exhausted. Please close this browser tab completely (Cmd+W / Ctrl+W) and reopen.' };
        }
        // Clean up the test context
        const loseContext = gl.getExtension('WEBGL_lose_context');
        if (loseContext) {
            loseContext.loseContext();
        }
        return { supported: true };
    } catch (e) {
        return { supported: false, reason: `WebGL check failed: ${e.message}` };
    }
}

// ==================== HELPERS ====================
const DEFAULT_STATE_COLORS = {
    evidence: '#f59e0b',
    onPath: '#10b981',
    activePath: '#06b6d4',
    visited: '#3b82f6',
    entry: '#ec4899',
};

const getNodeTypeSettings = (nodeType, vizConfig) => {
    const userSettings = vizConfig?.nodeTypeSettings?.[nodeType] || {};
    const defaults = DEFAULT_NODE_SETTINGS[nodeType] || DEFAULT_NODE_FALLBACK;
    return {
        color: userSettings.color ?? defaults.color ?? DEFAULT_NODE_FALLBACK.color,
        size: userSettings.size ?? defaults.size ?? DEFAULT_NODE_FALLBACK.size,
        visible: userSettings.visible ?? true,
    };
};

const getEdgeTypeSettings = (edgeType, vizConfig) => {
    const userSettings = vizConfig?.edgeTypeSettings?.[edgeType] || {};
    const defaults = DEFAULT_EDGE_SETTINGS[edgeType] || DEFAULT_EDGE_FALLBACK;
    return {
        color: userSettings.color ?? defaults.color ?? DEFAULT_EDGE_FALLBACK.color,
        width: userSettings.width ?? defaults.width ?? DEFAULT_EDGE_FALLBACK.width,
        visible: userSettings.visible ?? true,
        strength: userSettings.strength ?? defaults.strength ?? DEFAULT_EDGE_FALLBACK.strength ?? 1.0,
        distance: userSettings.distance ?? defaults.distance ?? DEFAULT_EDGE_FALLBACK.distance ?? 100,
    };
};

const getNodeColor = (node, vizConfig) => {
    const stateColors = vizConfig?.stateColors || DEFAULT_STATE_COLORS;
    // Evidence nodes: yellow highlight (highest priority)
    if (node.isEvidence) return '#eab308';  // Yellow - evidence found
    // Active path nodes: current step highlight
    if (node.isActivePath) return stateColors.activePath;
    // On path nodes: part of trace path
    if (node.isOnPath) return stateColors.onPath;
    // Visited nodes: green highlight (query traversal visited this node)
    if (node.visited) return '#22c55e';  // Green - visited
    const nodeType = node.pz_type || node.type;
    if (nodeType === 'entry') return stateColors.entry;
    return getNodeTypeSettings(nodeType, vizConfig).color;
};

const getEdgeColor = (edge, vizConfig) => {
    const stateColors = vizConfig?.stateColors || DEFAULT_STATE_COLORS;
    if (edge.isOnPath) return stateColors.onPath;
    if (edge.isActivePath) return stateColors.activePath;
    const edgeType = edge.pz_type || edge.type || '';
    return getEdgeTypeSettings(edgeType, vizConfig).color;
};

const getNodeRadius = (node, vizConfig) => {
    const nodeType = node.pz_type || node.type;
    const settings = getNodeTypeSettings(nodeType, vizConfig);
    const baseSize = settings.size || vizConfig?.nodeBaseSize || 3;
    return Math.max(2, Math.min(baseSize * 4 * (vizConfig?.nodeRadiusScale ?? 1.0), 24));
};

// ==================== LEGEND ====================
function DynamicLegend({ data, vizConfig, show, onToggle }) {
    const nodeTypes = useMemo(() => {
        const types = new Set();
        data.nodes?.forEach(n => {
            const t = n.pz_type || n.type;
            if (t) types.add(t);
        });
        return Array.from(types).sort();
    }, [data.nodes]);

    if (nodeTypes.length === 0) return null;

    return (
        <div className="absolute bottom-24 left-4 z-10">
            {show ? (
                <div className="bg-slate-900/90 p-3 rounded border border-slate-700 max-h-48 overflow-y-auto">
                    <div className="flex items-center justify-between mb-2">
                        <div className="text-xs text-slate-400 uppercase tracking-wider">Node Types</div>
                        <button 
                            onClick={onToggle}
                            className="text-xs text-slate-500 hover:text-slate-300 ml-2"
                            title="Hide legend"
                        >
                            ✕
                        </button>
                    </div>
                    <div className="space-y-1">
                        {nodeTypes.map(type => {
                            const settings = getNodeTypeSettings(type, vizConfig);
                            return (
                                <div key={type} className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: settings.color }} />
                                    <span className="text-xs text-slate-300">{type}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            ) : (
                <button 
                    onClick={onToggle}
                    className="bg-slate-900/90 px-2 py-1 rounded border border-slate-700 text-xs text-slate-400 hover:text-slate-200"
                    title="Show legend"
                >
                    Legend
                </button>
            )}
        </div>
    );
}

// ==================== MAIN COMPONENT ====================
function GraphVisualization3D({ 
    data, 
    onNodeClick, 
    onNodeHover,
    vizConfig = {},
    graphId,
    showLegend = true,
    onToggleLegend,
}) {
    const fgRef = useRef();
    const containerRef = useRef();
    
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [hoveredNode, setHoveredNode] = useState(null);
    const [webglStatus, setWebglStatus] = useState({ supported: true, checked: false });
    const [perfStats, setPerfStats] = useState(null);
    
    // Perf tracking ref
    const perfRef = useRef({
        enabled: false,
        lastReport: 0,
        frames: 0,
        frameTimes: [],
        lastFrameTime: 0,
        engineRunning: false,
    });

    // ==================== CHECK WEBGL AVAILABILITY ====================
    useEffect(() => {
        const status = checkWebGLAvailability();
        setWebglStatus({ ...status, checked: true });
    }, []);

    // ==================== PERF TRACKING ====================
    useEffect(() => {
        perfRef.current.enabled = vizConfig?.perfHud || false;
    }, [vizConfig?.perfHud]);

    // Frame timing using requestAnimationFrame
    useEffect(() => {
        if (!vizConfig?.perfHud) {
            setPerfStats(null);
            return;
        }

        let animId;
        let lastTime = performance.now();
        const frameTimes = [];
        const MAX_SAMPLES = 60;

        const tick = () => {
            const now = performance.now();
            const dt = now - lastTime;
            lastTime = now;
            
            frameTimes.push(dt);
            if (frameTimes.length > MAX_SAMPLES) frameTimes.shift();

            // Update stats every 500ms
            if (now - perfRef.current.lastReport > 500 && frameTimes.length > 0) {
                perfRef.current.lastReport = now;

                const sum = frameTimes.reduce((a, b) => a + b, 0);
                const avg = sum / frameTimes.length;
                const max = Math.max(...frameTimes);
                const min = Math.min(...frameTimes);
                const fps = 1000 / avg;

                // Calculate frame time variance (indicates GPU pressure)
                const variance = frameTimes.reduce((acc, t) => acc + Math.pow(t - avg, 2), 0) / frameTimes.length;
                const stdDev = Math.sqrt(variance);
                const jitter = stdDev / avg; // Normalized jitter

                // Memory (Chrome only)
                let memoryMB = null;
                let memoryUsedMB = null;
                if (performance.memory) {
                    memoryMB = Math.round(performance.memory.totalJSHeapSize / 1024 / 1024);
                    memoryUsedMB = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
                }

                // GPU info from renderer
                let gpuInfo = null;
                let drawCalls = null;
                let triangles = null;
                const fg = fgRef.current;
                if (fg) {
                    try {
                        const renderer = fg.renderer();
                        if (renderer) {
                            const info = renderer.info;
                            if (info) {
                                drawCalls = info.render?.calls;
                                triangles = info.render?.triangles;
                            }
                            const gl = renderer.getContext();
                            if (gl) {
                                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                                if (debugInfo) {
                                    gpuInfo = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                                }
                            }
                        }
                    } catch (e) { /* ignore */ }
                }

                // Engine state
                let engineRunning = false;
                if (fg && typeof fg.isEngineRunning === 'function') {
                    try {
                        engineRunning = fg.isEngineRunning();
                    } catch (e) { /* ignore */ }
                }

                // Estimate load based on frame budget (16.67ms = 60fps)
                const loadPercent = Math.round((avg / 16.67) * 100);
                
                // Bottleneck estimation
                let bottleneck = 'unknown';
                if (loadPercent < 50) bottleneck = 'none (smooth)';
                else if (loadPercent < 100) bottleneck = 'moderate';
                else if (jitter > 0.3) bottleneck = 'GPU (high jitter)';
                else bottleneck = 'CPU/GPU saturated';

                setPerfStats({
                    fps,
                    frameMsAvg: avg,
                    frameMsMax: max,
                    frameMsMin: min,
                    frameJitter: jitter,
                    loadPercent,
                    memoryMB,
                    memoryUsedMB,
                    gpuInfo,
                    drawCalls,
                    triangles,
                    engineRunning,
                    bottleneck,
                    nodes: data.nodes?.length || 0,
                    links: (data.allLinks?.length || data.links?.length || 0),
                });
            }

            animId = requestAnimationFrame(tick);
        };

        animId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(animId);
    }, [vizConfig?.perfHud, data.nodes?.length, data.allLinks?.length, data.links?.length]);

    // ==================== RESIZE ====================
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                setDimensions({ width: entry.contentRect.width, height: entry.contentRect.height });
            }
        });
        observer.observe(container);
        const rect = container.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
        return () => observer.disconnect();
    }, []);

    // ==================== CLEANUP WEBGL ON UNMOUNT ====================
    useEffect(() => {
        return () => {
            const fg = fgRef.current;
            if (fg) {
                try {
                    const renderer = fg.renderer();
                    if (renderer) {
                        renderer.dispose();
                        renderer.forceContextLoss();
                    }
                } catch (e) {
                    // ignore
                }
            }
        };
    }, []);

    // ==================== GRAPH DATA ====================
    // STABLE node storage - maintains object identity across renders to preserve simulation positions
    const stableNodeMapRef = useRef(new Map());  // id -> stable node object (positions preserved)
    const stableLinkMapRef = useRef(new Map());  // "source->target" -> stable link object
    const lastGraphIdRef = useRef(graphId);

    // Clear stable maps when switching to a different graph
    useEffect(() => {
        if (graphId !== lastGraphIdRef.current) {
            stableNodeMapRef.current.clear();
            stableLinkMapRef.current.clear();
            lastGraphIdRef.current = graphId;
        }
    }, [graphId]);

    // Compute a hash of node states (visited, isEvidence, isOnPath, isActivePath) to detect changes
    // This is needed because stable node objects don't trigger React change detection
    const nodeStateHash = useMemo(() => {
        if (!data.nodes?.length) return '';
        const parts = data.nodes.map(n => 
            `${n.id}:${n.visited ? 1 : 0}${n.isEvidence ? 1 : 0}${n.isOnPath ? 1 : 0}${n.isActivePath ? 1 : 0}`
        );
        return parts.join(',');
    }, [data.nodes]);

    // Hash of visibility settings only - changing colors shouldn't reset the simulation
    const visibilityHash = useMemo(() => {
        const nodeTypes = vizConfig?.nodeTypeSettings || {};
        const edgeTypes = vizConfig?.edgeTypeSettings || {};
        const nodeParts = Object.entries(nodeTypes).map(([k, v]) => `${k}:${v?.visible ?? true}`);
        const edgeParts = Object.entries(edgeTypes).map(([k, v]) => `${k}:${v?.visible ?? true}`);
        return [...nodeParts, ...edgeParts].join('|');
    }, [vizConfig?.nodeTypeSettings, vizConfig?.edgeTypeSettings]);

    // Build stable graph data - reuses same node/link objects across renders
    // The simulation mutates node.x, node.y, node.z directly on these objects
    const graphDataMemo = useMemo(() => {
        if (!data.nodes?.length) return { nodes: [], links: [] };

        const stableNodeMap = stableNodeMapRef.current;
        const stableLinkMap = stableLinkMapRef.current;

        // Build set of current node IDs
        const currentNodeIds = new Set();
        const visibleNodeIds = new Set();

        // Process nodes - reuse existing objects, create new ones only for new nodes
        for (const rawNode of data.nodes) {
            const id = rawNode.id;
            currentNodeIds.add(id);
            
            const settings = getNodeTypeSettings(rawNode.pz_type || rawNode.type, vizConfig);
            if (settings.visible === false) continue;
            
            visibleNodeIds.add(id);
            
            let stableNode = stableNodeMap.get(id);
            if (stableNode) {
                // UPDATE existing node in-place - preserve x, y, z, vx, vy, vz!
                stableNode.label = rawNode.label;
                stableNode.pz_type = rawNode.pz_type;
                stableNode.type = rawNode.type;
                stableNode.isEvidence = rawNode.isEvidence;
                stableNode.isOnPath = rawNode.isOnPath;
                stableNode.isActivePath = rawNode.isActivePath;
                stableNode.visited = rawNode.visited;
                stableNode.val = Math.pow(getNodeRadius(rawNode, vizConfig) / 4, 3);
                // Don't touch x, y, z, vx, vy, vz - let simulation manage them
            } else {
                // CREATE new node
                stableNode = {
                    id,
                    label: rawNode.label,
                    pz_type: rawNode.pz_type,
                    type: rawNode.type,
                    isEvidence: rawNode.isEvidence,
                    isOnPath: rawNode.isOnPath,
                    isActivePath: rawNode.isActivePath,
                    visited: rawNode.visited,
                    val: Math.pow(getNodeRadius(rawNode, vizConfig) / 4, 3),
                    // New nodes start without positions - simulation will initialize them
                };
                stableNodeMap.set(id, stableNode);
            }
        }

        // Build nodes array - create NEW array but with STABLE node objects inside
        // This way ForceGraph detects the data change, but node positions are preserved
        const nodes = [];
        for (const id of visibleNodeIds) {
            const node = stableNodeMap.get(id);
            if (node) nodes.push(node);
        }

        // Process links
        const allLinks = data.allLinks || data.links || [];
        const currentLinkKeys = new Set();

        for (const rawLink of allLinks) {
            const sid = rawLink.source?.id ?? rawLink.source;
            const tid = rawLink.target?.id ?? rawLink.target;
            
            // Skip links with missing endpoints
            if (!visibleNodeIds.has(sid) || !visibleNodeIds.has(tid)) continue;
            
            const linkKey = `${sid}->${tid}`;
            currentLinkKeys.add(linkKey);
            
            const linkType = rawLink.pz_type || rawLink.type;
            const settings = getEdgeTypeSettings(linkType, vizConfig);
            
            let stableLink = stableLinkMap.get(linkKey);
            if (stableLink) {
                // UPDATE existing link in-place
                stableLink.type = linkType;
                stableLink.visible = settings.visible !== false;
                stableLink.isOnPath = rawLink.isOnPath;
                stableLink.isActivePath = rawLink.isActivePath;
            } else {
                // CREATE new link
                stableLink = {
                    source: sid,
                    target: tid,
                    type: linkType,
                    visible: settings.visible !== false,
                    isOnPath: rawLink.isOnPath,
                    isActivePath: rawLink.isActivePath,
                };
                stableLinkMap.set(linkKey, stableLink);
            }
        }

        // Build links array - NEW array with STABLE link objects
        const links = [];
        for (const key of currentLinkKeys) {
            const link = stableLinkMap.get(key);
            if (link) links.push(link);
        }

        // Return NEW object so ForceGraph detects change, but node/link objects are stable
        return { nodes, links };
    }, [data, visibilityHash]);  // Only depend on data and visibility, not colors/forces

    // ==================== CAMERA SETUP (top-down 2D view) ====================
    useEffect(() => {
        if (!dimensions.width) return;
        
        const timer = setTimeout(() => {
            const fg = fgRef.current;
            if (!fg) return;
            
            // Position camera looking down at z=0 plane
            fg.cameraPosition({ x: 0, y: 0, z: 1000 }, { x: 0, y: 0, z: 0 }, 0);
            
            // Configure controls for 2D-style panning
            const controls = fg.controls();
            if (controls) {
                controls.enableRotate = false;  // No rotation - keep top-down
                controls.screenSpacePanning = true;
                controls.mouseButtons = {
                    LEFT: THREE.MOUSE.PAN,
                    MIDDLE: THREE.MOUSE.DOLLY,
                    RIGHT: THREE.MOUSE.PAN,
                };
            }
        }, 100);
        
        return () => clearTimeout(timer);
    }, [dimensions.width]);

    // ==================== FORCE CONFIGURATION ====================
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return;

        // Charge force
        const charge = fg.d3Force('charge');
        if (charge) {
            charge.strength(vizConfig?.chargeStrength ?? -200);
            if (typeof charge.distanceMax === 'function') {
                charge.distanceMax(vizConfig?.chargeDistanceMax ?? 500);
            }
        }

        // Link force - use per-link strength based on edge type
        const link = fg.d3Force('link');
        if (link) {
            // Distance can vary per link type
            link.distance((l) => {
                const linkType = l.type || '';
                const settings = getEdgeTypeSettings(linkType, vizConfig);
                const baseDistance = vizConfig?.linkDistance ?? 60;
                return baseDistance * (settings.distance / 100);  // distance is stored as 0-200 scale
            });
            // Strength varies per link type
            if (typeof link.strength === 'function') {
                link.strength((l) => {
                    const linkType = l.type || '';
                    const settings = getEdgeTypeSettings(linkType, vizConfig);
                    const baseStrength = vizConfig?.linkStrength ?? 0.5;
                    return baseStrength * settings.strength;  // strength is 0-2 multiplier
                });
            }
        }

        // Center forces (replace default center with x/y forces)
        fg.d3Force('center', null);
        fg.d3Force('x', d3.forceX(0).strength(vizConfig?.centerStrength ?? 0.03));
        fg.d3Force('y', d3.forceY(0).strength(vizConfig?.centerStrength ?? 0.03));
        
        // No collision for performance on large graphs
        fg.d3Force('collide', null);

        // Only reheat if charge/center settings changed (not link settings)
        // Link strength/distance apply immediately without reheat
    }, [vizConfig?.chargeStrength, vizConfig?.chargeDistanceMax, vizConfig?.linkDistance, vizConfig?.linkStrength, vizConfig?.centerStrength, vizConfig?.edgeTypeSettings]);

    // ==================== PAUSE/RESUME ====================
    // Only resume animation - don't reheat on every toggle (causes jitter)
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return;
        
        if (vizConfig?.runLayout) {
            fg.resumeAnimation?.();
            // Don't call d3ReheatSimulation() - let the existing simulation continue
        } else {
            fg.pauseAnimation?.();
        }
    }, [vizConfig?.runLayout]);

    // ==================== UPDATE NODE COLORS ON STATE CHANGE ====================
    // When node states (visited, isEvidence, etc.) change, update their mesh colors directly
    // Use nodeStateHash to detect changes in stable node objects
    useEffect(() => {
        // Count visited nodes from RAW input data for debugging
        const rawVisitedCount = data.nodes?.filter(n => n.visited).length || 0;
        const rawEvidenceCount = data.nodes?.filter(n => n.isEvidence).length || 0;
        // Also count from stable nodes (these are what we render)
        const stableVisitedCount = graphDataMemo.nodes?.filter(n => n.visited).length || 0;
        const stableEvidenceCount = graphDataMemo.nodes?.filter(n => n.isEvidence).length || 0;
        console.log(`[3D] Node state update - raw: visited=${rawVisitedCount}, evidence=${rawEvidenceCount} | stable: visited=${stableVisitedCount}, evidence=${stableEvidenceCount}`);
        
        const timer = setTimeout(() => {
            const fg = fgRef.current;
            if (!fg || typeof fg.graphData !== 'function') return;
            
            // Get the current graph data from the scene
            let gData;
            try {
                gData = fg.graphData();
            } catch (e) {
                return; // Graph not ready yet
            }
            if (!gData?.nodes) return;
            
            let updatedCount = 0;
            // Update each node's mesh color
            for (const node of gData.nodes) {
                if (node.__threeObj && node.__threeObj.material) {
                    const newColor = getNodeColor(node, vizConfig);
                    node.__threeObj.material.color.set(newColor);
                    if (node.visited || node.isEvidence) updatedCount++;
                }
            }
            console.log(`[3D] Updated ${updatedCount} special nodes, total with __threeObj: ${gData.nodes.filter(n => n.__threeObj).length}`);
        }, 100);  // Delay to let ForceGraph create the objects first
        
        return () => clearTimeout(timer);
    }, [nodeStateHash, vizConfig, data.nodes, graphDataMemo.nodes]);  // Use nodeStateHash to trigger on state changes

    // ==================== NODE RENDERING ====================
    // Color callback for default sphere rendering (when nodeThreeObject returns undefined)
    const nodeColor = useCallback((node) => {
        return getNodeColor(node, vizConfig);
    }, [vizConfig]);

    // For large graphs, use simpler rendering
    const isLargeGraph = graphDataMemo.nodes.length > 1000;
    const useBatchedLinks = graphDataMemo.links.length > 5000;  // Use batched rendering for many links
    
    // Shared geometries for node shapes - disposed on unmount
    const geometriesRef = useRef(null);
    const geometries = useMemo(() => {
        // Dispose old geometries if they exist
        if (geometriesRef.current) {
            Object.values(geometriesRef.current).forEach(g => g.dispose());
        }
        const geos = {
            circle: new THREE.SphereGeometry(1, 8, 8), // Low poly sphere
            square: new THREE.BoxGeometry(1.5, 1.5, 1.5),
            triangle: new THREE.TetrahedronGeometry(1.5),
            diamond: new THREE.OctahedronGeometry(1.5),
            hexagon: new THREE.CylinderGeometry(1, 1, 1, 6),
        };
        geometriesRef.current = geos;
        return geos;
    }, []);
    
    // Cleanup geometries on unmount
    useEffect(() => {
        return () => {
            if (geometriesRef.current) {
                Object.values(geometriesRef.current).forEach(g => g.dispose());
            }
        };
    }, []);

    // Custom node renderer to support shapes AND ensure colors update on state changes
    const nodeThreeObject = useCallback((node) => {
        const settings = getNodeTypeSettings(node.pz_type || node.type, vizConfig);
        const shape = settings.shape || 'circle';
        const color = getNodeColor(node, vizConfig);
        
        // Check if we already have a mesh and just need to update the color
        if (node.__threeObj && node.__threeObj.material) {
            const currentColor = node.__threeObj.material.color.getHexString();
            const newColor = color.replace('#', '');
            if (currentColor !== newColor) {
                node.__threeObj.material.color.set(color);
            }
            return node.__threeObj;
        }
        
        // Create new mesh
        const geometry = geometries[shape] || geometries.circle;
        const material = new THREE.MeshLambertMaterial({ 
            color, 
            transparent: true, 
            opacity: settings.opacity ?? 0.9 
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        const size = getNodeRadius(node, vizConfig);
        mesh.scale.set(size, size, size);
        
        return mesh;
    }, [vizConfig, geometries]);  // Don't include graphDataMemo - color updates handled by separate effect

    // ==================== BATCHED LINK RENDERING ====================
    // For large graphs, we bypass the library's per-link rendering and use
    // a single THREE.LineSegments object for ALL edges - ONE draw call.
    const batchedLinksRef = useRef(null);
    const linkPositionsRef = useRef(null);
    
    // Create batched link geometry (only for large graphs)
    useEffect(() => {
        if (!useBatchedLinks) return;
        const fg = fgRef.current;
        if (!fg) return;
        
        const scene = fg.scene();
        if (!scene) return;
        
        // Remove old batched lines if any
        if (batchedLinksRef.current) {
            scene.remove(batchedLinksRef.current);
            batchedLinksRef.current.geometry.dispose();
            batchedLinksRef.current.material.dispose();
            batchedLinksRef.current = null;
        }
        
        const numLinks = graphDataMemo.links.length;
        if (numLinks === 0) return;
        
        // Create position buffer: 2 vertices per line, 3 floats per vertex
        const positions = new Float32Array(numLinks * 2 * 3);
        linkPositionsRef.current = positions;
        
        // Create color buffer: 2 vertices per line, 3 floats (RGB) per vertex
        const colors = new Float32Array(numLinks * 2 * 3);
        const links = graphDataMemo.links;
        
        for (let i = 0; i < links.length; i++) {
            const link = links[i];
            const settings = getEdgeTypeSettings(link.type || '', vizConfig);
            
            let r=0, g=0, b=0;
            if (settings.visible !== false) {
                const colorHex = settings.color;
                const color = new THREE.Color(colorHex);
                r = color.r;
                g = color.g;
                b = color.b;
            }
            
            // Source vertex color
            colors[i * 6 + 0] = r;
            colors[i * 6 + 1] = g;
            colors[i * 6 + 2] = b;
            
            // Target vertex color
            colors[i * 6 + 3] = r;
            colors[i * 6 + 4] = g;
            colors[i * 6 + 5] = b;
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // Mark position as dynamic since we update every frame
        geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
        
        // Simple line material with vertex colors
        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: vizConfig?.edgeOpacity ?? 0.4,
            depthWrite: false,
            linewidth: vizConfig?.edgeWidth ?? 1, // Note: WebGL linewidth is often limited to 1
        });
        
        const lineSegments = new THREE.LineSegments(geometry, material);
        lineSegments.frustumCulled = false;  // Always render
        lineSegments.renderOrder = -1;  // Render before nodes
        scene.add(lineSegments);
        batchedLinksRef.current = lineSegments;
        
        return () => {
            if (batchedLinksRef.current && scene) {
                scene.remove(batchedLinksRef.current);
                batchedLinksRef.current.geometry.dispose();
                batchedLinksRef.current.material.dispose();
                batchedLinksRef.current = null;
            }
        };
    }, [useBatchedLinks, graphDataMemo.links.length, vizConfig]); // Re-run when vizConfig changes to update colors
    
    // Update batched link positions using requestAnimationFrame
    useEffect(() => {
        if (!useBatchedLinks) return;
        
        let animationId;
        
        const updateLinkPositions = () => {
            if (batchedLinksRef.current && linkPositionsRef.current) {
                const positions = linkPositionsRef.current;
                const links = graphDataMemo.links;
                
                for (let i = 0; i < links.length; i++) {
                    const link = links[i];
                    const source = link.source;
                    const target = link.target;
                    
                    if (source && target) {
                        // If hidden, collapse geometry to a single point (0,0,0)
                        if (link.visible === false) {
                            positions[i * 6 + 0] = 0;
                            positions[i * 6 + 1] = 0;
                            positions[i * 6 + 2] = 0;
                            positions[i * 6 + 3] = 0;
                            positions[i * 6 + 4] = 0;
                            positions[i * 6 + 5] = 0;
                            continue;
                        }

                        // Source vertex
                        positions[i * 6 + 0] = source.x || 0;
                        positions[i * 6 + 1] = source.y || 0;
                        positions[i * 6 + 2] = 0;
                        
                        // Target vertex
                        positions[i * 6 + 3] = target.x || 0;
                        positions[i * 6 + 4] = target.y || 0;
                        positions[i * 6 + 5] = 0;
                    }
                }
                
                batchedLinksRef.current.geometry.attributes.position.needsUpdate = true;
            }
            
            animationId = requestAnimationFrame(updateLinkPositions);
        };
        
        // Start the update loop
        animationId = requestAnimationFrame(updateLinkPositions);
        
        return () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        };
    }, [useBatchedLinks, graphDataMemo.links]);

    // ==================== STANDARD LINK RENDERING (for small graphs) ====================
    const linkColor = useCallback((link) => {
        const settings = getEdgeTypeSettings(link.type || '', vizConfig);
        return settings.color;
    }, [vizConfig]);

    // Link visibility function - checks edge type settings
    const linkVisibility = useCallback((link) => {
        const settings = getEdgeTypeSettings(link.type || '', vizConfig);
        return settings.visible !== false;
    }, [vizConfig]);

    // ==================== INTERACTION ====================
    const handleNodeClick = useCallback((node, event) => {
        if (onNodeClick) onNodeClick(node, event);
    }, [onNodeClick]);

    const handleNodeHover = useCallback((node) => {
        setHoveredNode(node);
        if (onNodeHover) onNodeHover(node);
    }, [onNodeHover]);

    // ==================== ZOOM CONTROLS ====================
    const handleZoom = useCallback((factor) => {
        const fg = fgRef.current;
        if (!fg) return;
        const camera = fg.camera();
        if (camera) {
            const newZ = Math.max(100, Math.min(5000, camera.position.z / factor));
            fg.cameraPosition({ x: camera.position.x, y: camera.position.y, z: newZ });
        }
    }, []);

    const handleFit = useCallback(() => {
        const fg = fgRef.current;
        if (fg) fg.zoomToFit(400, 50);
    }, []);

    const handleSearchSelect = useCallback((node) => {
        const fg = fgRef.current;
        if (!fg || !node) return;
        const camera = fg.camera();
        fg.cameraPosition(
            { x: node.x || 0, y: node.y || 0, z: camera?.position.z || 500 },
            { x: node.x || 0, y: node.y || 0, z: 0 },
            1000
        );
    }, []);

    // ==================== WEBGL NOT AVAILABLE - show error ====================
    if (webglStatus.checked && !webglStatus.supported) {
        return (
            <div ref={containerRef} style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
                color: '#e0e0e0',
                padding: '2rem',
                textAlign: 'center',
            }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
                <h2 style={{ margin: '0 0 1rem 0', color: '#f59e0b' }}>WebGL Not Available</h2>
                <p style={{ margin: '0 0 1.5rem 0', maxWidth: '400px', lineHeight: 1.6 }}>
                    {webglStatus.reason}
                </p>
                <button
                    onClick={() => window.location.reload()}
                    style={{
                        padding: '0.75rem 1.5rem',
                        background: '#3b82f6',
                        color: 'white',
                        border: 'none',
                        borderRadius: '0.5rem',
                        cursor: 'pointer',
                        fontSize: '1rem',
                    }}
                >
                    Reload Page
                </button>
                <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#888' }}>
                    If reloading doesn't work, close this tab completely (Cmd+W) and reopen.
                </p>
            </div>
        );
    }

    // ==================== RENDER ====================
    return (
        <div ref={containerRef} className="w-full h-full bg-slate-950 relative">
            {dimensions.width > 0 && (
                <ForceGraph3D
                    ref={fgRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    graphData={graphDataMemo}
                    backgroundColor="#0f172a"
                    
                    // KEY: 2D layout with 3D/WebGL rendering
                    numDimensions={2}
                    
                    // Node appearance
                    nodeThreeObject={nodeThreeObject}
                    nodeThreeObjectExtend={false}
                    nodeColor={nodeColor}
                    nodeVal="val"
                    nodeLabel={node => node.label || node.id}
                    nodeOpacity={0.9}
                    nodeResolution={isLargeGraph ? 4 : 8}
                    
                    // Link appearance - use per-link visibility check when not batched
                    linkVisibility={useBatchedLinks ? false : linkVisibility}
                    linkColor={useBatchedLinks ? () => 'transparent' : linkColor}
                    linkWidth={useBatchedLinks ? 0 : (vizConfig?.edgeWidth ?? 0.3)}
                    linkOpacity={useBatchedLinks ? 0 : (vizConfig?.edgeOpacity ?? 0.4)}
                    linkResolution={2}
                    
                    // Physics
                    d3VelocityDecay={vizConfig?.d3VelocityDecay ?? 0.3}
                    warmupTicks={0}
                    cooldownTicks={vizConfig?.runLayout ? Infinity : 0}
                    
                    // Interaction - disable for large graphs (>5k nodes)
                    onNodeClick={handleNodeClick} 
                    onNodeHover={graphDataMemo.nodes.length < 5000 ? handleNodeHover : undefined}
                    enableNodeDrag={graphDataMemo.nodes.length < 5000}
                    enableNavigationControls={true}
                    
                    // Performance optimizations
                    enablePointerInteraction={graphDataMemo.nodes.length < 5000}
                    rendererConfig={{ 
                        antialias: false,  // Disable AA for perf
                        alpha: false,
                        powerPreference: 'high-performance',
                    }}
                />
            )}

            {/* Search */}
            <div className="absolute top-4 right-4 z-10">
                <GraphSearch nodes={data.nodes} onSelectNode={handleSearchSelect} />
            </div>

            {/* Legend */}
            <DynamicLegend data={data} vizConfig={vizConfig} show={showLegend} onToggle={onToggleLegend} />

            {/* Zoom Controls */}
            <div className="absolute bottom-4 right-4 flex flex-col gap-2 z-10">
                <button onClick={() => handleZoom(1.3)} className="p-2 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 shadow-lg border border-slate-700">
                    <ZoomIn size={16} />
                </button>
                <button onClick={() => handleZoom(0.7)} className="p-2 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 shadow-lg border border-slate-700">
                    <ZoomOut size={16} />
                </button>
                <button onClick={handleFit} className="p-2 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 shadow-lg border border-slate-700">
                    <Maximize size={16} />
                </button>
            </div>

            {/* Hover Info */}
            {hoveredNode && (
                <div className="absolute bottom-24 right-4 bg-slate-900/95 p-3 rounded border border-slate-700 z-10 max-w-sm">
                    <div className="text-sm font-medium text-white">{hoveredNode.label || hoveredNode.id}</div>
                    <div className="text-xs text-slate-400 mt-1">{hoveredNode.pz_type || hoveredNode.type}</div>
                </div>
            )}
            
            {/* Performance HUD */}
            {vizConfig?.perfHud && perfStats && (
                <div className="absolute top-4 left-4 bg-gray-900/95 p-3 rounded border border-gray-700 backdrop-blur pointer-events-none z-10 max-w-md select-none">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="px-2 py-0.5 bg-cyan-900/80 text-cyan-300 text-[10px] rounded border border-cyan-700">GPU 🚀</span>
                        <span className="text-[10px] text-gray-400 uppercase tracking-wider">WebGL Performance</span>
                    </div>
                    <div className="text-xs text-gray-200 font-mono leading-relaxed">
                        {/* Summary */}
                        <div className="text-yellow-400 font-bold text-sm">
                            {perfStats.fps.toFixed(0)} fps • {perfStats.frameMsAvg.toFixed(1)}ms/frame
                        </div>
                        <div className="text-[10px] text-gray-400 mb-2">
                            min: {perfStats.frameMsMin.toFixed(1)}ms | max: {perfStats.frameMsMax.toFixed(1)}ms | target: 16.7ms
                        </div>
                        
                        {/* Load Bar */}
                        <div className="text-[10px] text-gray-400 uppercase mb-1">GPU Load Estimate</div>
                        <div className="h-3 bg-gray-800 rounded overflow-hidden mb-1">
                            <div 
                                className={`h-full transition-all ${
                                    perfStats.loadPercent < 50 ? 'bg-green-500' :
                                    perfStats.loadPercent < 80 ? 'bg-yellow-500' :
                                    perfStats.loadPercent < 100 ? 'bg-orange-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${Math.min(100, perfStats.loadPercent)}%` }}
                            />
                        </div>
                        <div className="text-[10px] text-gray-400 mb-2">
                            {perfStats.loadPercent}% of 60fps budget
                        </div>

                        {/* Render Stats */}
                        <div className="border-t border-gray-700 pt-2 mt-1">
                            <div className="text-[10px] text-gray-400 uppercase mb-1">WebGL Stats</div>
                            <table className="w-full text-[10px]">
                                <tbody>
                                    {perfStats.drawCalls !== null && (
                                        <tr>
                                            <td className="pr-2">Draw calls</td>
                                            <td className="text-right font-bold">{perfStats.drawCalls?.toLocaleString()}</td>
                                        </tr>
                                    )}
                                    {perfStats.triangles !== null && (
                                        <tr>
                                            <td className="pr-2">Triangles</td>
                                            <td className="text-right font-bold">{perfStats.triangles?.toLocaleString()}</td>
                                        </tr>
                                    )}
                                    <tr>
                                        <td className="pr-2">Frame jitter</td>
                                        <td className={`text-right font-bold ${perfStats.frameJitter > 0.3 ? 'text-orange-400' : ''}`}>
                                            {(perfStats.frameJitter * 100).toFixed(0)}%
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        {/* Memory (Chrome only) */}
                        {perfStats.memoryMB !== null && (
                            <div className="border-t border-gray-700 pt-2 mt-2">
                                <div className="text-[10px] text-gray-400 uppercase mb-1">Memory (JS Heap)</div>
                                <div className="flex justify-between">
                                    <span>Used / Total</span>
                                    <span className="font-bold">{perfStats.memoryUsedMB}MB / {perfStats.memoryMB}MB</span>
                                </div>
                            </div>
                        )}

                        {/* Graph Stats */}
                        <div className="border-t border-gray-700 pt-2 mt-2">
                            <div className="text-[10px] text-gray-400 uppercase mb-1">Graph</div>
                            <div className="flex justify-between">
                                <span>Nodes</span>
                                <span className="font-bold">{perfStats.nodes.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Edges</span>
                                <span className="font-bold">{perfStats.links.toLocaleString()}</span>
                            </div>
                            {useBatchedLinks && (
                                <div className="text-[9px] text-green-400 mt-1">
                                    ✓ Batched rendering (1 draw call)
                                </div>
                            )}
                            <div className="flex justify-between text-[10px] text-gray-400">
                                <span>Physics</span>
                                <span>{perfStats.engineRunning ? '🟢 running' : '⏸️ paused'}</span>
                            </div>
                        </div>

                        {/* GPU Info */}
                        {perfStats.gpuInfo && (
                            <div className="border-t border-gray-700 pt-2 mt-2">
                                <div className="text-[10px] text-gray-400 uppercase mb-1">GPU</div>
                                <div className="text-[9px] text-gray-300 break-all">{perfStats.gpuInfo}</div>
                            </div>
                        )}

                        {/* Bottleneck */}
                        <div className="border-t border-gray-700 pt-2 mt-2 flex items-center gap-2">
                            <span className="text-[10px] text-gray-400">STATUS:</span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                perfStats.bottleneck.includes('none') ? 'bg-green-900 text-green-200' :
                                perfStats.bottleneck.includes('moderate') ? 'bg-yellow-900 text-yellow-200' :
                                perfStats.bottleneck.includes('GPU') ? 'bg-orange-900 text-orange-200' :
                                'bg-red-900 text-red-200'
                            }`}>
                                {perfStats.bottleneck}
                            </span>
                        </div>
                        
                        {/* Recommendation for large graphs */}
                        {perfStats.links > 20000 && perfStats.fps < 30 && (
                            <div className="border-t border-gray-700 pt-2 mt-2 bg-blue-900/50 -mx-3 px-3 py-2 rounded">
                                <div className="text-[10px] text-blue-300 font-bold uppercase mb-1">💡 Recommendation</div>
                                <div className="text-[10px] text-blue-200">
                                    With {perfStats.links.toLocaleString()} edges, <strong>Canvas mode</strong> may perform better.
                                    WebGL creates 1 draw call per edge.
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
            
            {/* GPU indicator (only show if perf HUD is off) */}
            {!vizConfig?.perfHud && (
                <div className="absolute top-4 left-4 px-2 py-1 bg-cyan-900/80 text-cyan-300 text-xs rounded border border-cyan-700">
                    GPU 🚀
                </div>
            )}
        </div>
    );
}

export default GraphVisualization3D;
