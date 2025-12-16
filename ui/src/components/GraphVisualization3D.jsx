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
    };
};

const getNodeColor = (node, vizConfig) => {
    const stateColors = vizConfig?.stateColors || DEFAULT_STATE_COLORS;
    if (node.isEvidence) return stateColors.evidence;
    if (node.isOnPath) return stateColors.onPath;
    if (node.isActivePath) return stateColors.activePath;
    if (node.visited) return stateColors.visited;
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
    if (node.isEvidence) return 8 * (vizConfig?.nodeRadiusScale ?? 1.0);
    if (node.isOnPath || node.isActivePath) return 6 * (vizConfig?.nodeRadiusScale ?? 1.0);
    const nodeType = node.pz_type || node.type;
    const settings = getNodeTypeSettings(nodeType, vizConfig);
    const baseSize = settings.size || vizConfig?.nodeBaseSize || 3;
    return Math.max(2, Math.min(baseSize * 4 * (vizConfig?.nodeRadiusScale ?? 1.0), 24));
};

// ==================== LEGEND ====================
function DynamicLegend({ data, vizConfig }) {
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
        <div className="absolute bottom-4 left-4 bg-slate-900/90 p-3 rounded border border-slate-700 z-10 max-h-48 overflow-y-auto">
            <div className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Node Types</div>
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
    );
}

// ==================== MAIN COMPONENT ====================
function GraphVisualization3D({ 
    data, 
    onNodeClick, 
    onNodeHover,
    vizConfig = {},
    graphId,
}) {
    const fgRef = useRef();
    const containerRef = useRef();
    
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [hoveredNode, setHoveredNode] = useState(null);
    const [isReady, setIsReady] = useState(false);
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
    // Keep track of node positions to prevent reset on data updates
    const nodePositionsRef = useRef(new Map());

    // Update positions ref periodically to capture simulation state
    useEffect(() => {
        const interval = setInterval(() => {
            const fg = fgRef.current;
            if (fg) {
                // Access internal graph data safely
                // Note: fg.graphData() might fail if component is not ready, so we wrap in try/catch
                try {
                    const gData = fg.graphData();
                    if (gData && gData.nodes) {
                        gData.nodes.forEach(n => {
                            if (n.id && (n.x !== undefined || n.vx !== undefined)) {
                                nodePositionsRef.current.set(n.id, {
                                    x: n.x, y: n.y, z: n.z,
                                    vx: n.vx, vy: n.vy, vz: n.vz,
                                    fx: n.fx, fy: n.fy, fz: n.fz
                                });
                            }
                        });
                    }
                } catch (e) {
                    // Ignore errors during initialization
                }
            }
        }, 1000); // Save every second
        return () => clearInterval(interval);
    }, []);

    const graphDataMemo = useMemo(() => {
        if (!data.nodes?.length) return { nodes: [], links: [] };

        // Filter visible nodes
        const visibleNodeIds = new Set();
        const nodes = data.nodes
            .filter(n => getNodeTypeSettings(n.pz_type || n.type, vizConfig).visible !== false)
            .map(n => {
                visibleNodeIds.add(n.id);
                // Restore position if available
                const pos = nodePositionsRef.current.get(n.id);
                return {
                    id: n.id,
                    label: n.label,
                    pz_type: n.pz_type,
                    type: n.type,
                    isEvidence: n.isEvidence,
                    isOnPath: n.isOnPath,
                    isActivePath: n.isActivePath,
                    visited: n.visited,
                    // Convert linear radius to cubic volume for ForceGraph (radius = cbrt(val) * nodeRelSize)
                    // nodeRelSize defaults to 4
                    val: Math.pow(getNodeRadius(n, vizConfig) / 4, 3),
                    // Spread restored positions
                    ...(pos || {})
                };
            });

        // Filter visible links
        const allLinks = data.allLinks || data.links || [];
        const links = allLinks
            .filter(l => {
                // Don't filter by visibility here - keep them for physics!
                // Just check if endpoints exist
                const sid = l.source?.id ?? l.source;
                const tid = l.target?.id ?? l.target;
                return visibleNodeIds.has(sid) && visibleNodeIds.has(tid);
            })
            .map(l => ({
                source: l.source?.id ?? l.source,
                target: l.target?.id ?? l.target,
                type: l.pz_type || l.type,
                visible: getEdgeTypeSettings(l.pz_type || l.type, vizConfig).visible !== false
            }));

        return { nodes, links };
    }, [data, vizConfig]);

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
            
            setIsReady(true);
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

        // Link force
        const link = fg.d3Force('link');
        if (link) {
            link.distance(vizConfig?.linkDistance ?? 60);
            if (typeof link.strength === 'function') {
                link.strength(vizConfig?.linkStrength ?? 0.5);
            }
        }

        // Center forces (replace default center with x/y forces)
        fg.d3Force('center', null);
        fg.d3Force('x', d3.forceX(0).strength(vizConfig?.centerStrength ?? 0.03));
        fg.d3Force('y', d3.forceY(0).strength(vizConfig?.centerStrength ?? 0.03));
        
        // No collision for performance on large graphs
        fg.d3Force('collide', null);

    }, [vizConfig?.chargeStrength, vizConfig?.chargeDistanceMax, vizConfig?.linkDistance, vizConfig?.linkStrength, vizConfig?.centerStrength]);

    // ==================== PAUSE/RESUME ====================
    useEffect(() => {
        const fg = fgRef.current;
        if (!fg) return;
        
        if (vizConfig?.runLayout) {
            fg.d3ReheatSimulation();
        }
    }, [vizConfig?.runLayout]);

    // ==================== NODE RENDERING ====================
    const nodeColor = useCallback((node) => {
        return getNodeColor(node, vizConfig);
    }, [vizConfig]);

    // For large graphs, use simpler rendering
    const isLargeGraph = graphDataMemo.nodes.length > 1000;
    const useBatchedLinks = graphDataMemo.links.length > 5000;  // Use batched rendering for many links
    
    // Shared geometries for node shapes
    const geometries = useMemo(() => ({
        circle: new THREE.SphereGeometry(1, 8, 8), // Low poly sphere
        square: new THREE.BoxGeometry(1.5, 1.5, 1.5),
        triangle: new THREE.TetrahedronGeometry(1.5),
        diamond: new THREE.OctahedronGeometry(1.5),
        hexagon: new THREE.CylinderGeometry(1, 1, 1, 6),
    }), []);

    // Custom node renderer to support shapes
    const nodeThreeObject = useCallback((node) => {
        const settings = getNodeTypeSettings(node.pz_type || node.type, vizConfig);
        const shape = settings.shape || 'circle';
        
        // If circle, return undefined to use default optimized sphere (unless we really want custom material)
        // Actually, default sphere is better optimized by the library (instancing support maybe?)
        // But if we want mixed shapes, we might need to provide objects for all.
        // However, for 9k nodes, creating 9k meshes is heavy.
        // Let's try to use default for circles and custom for others.
        if (shape === 'circle') return undefined;
        
        const geometry = geometries[shape] || geometries.circle;
        const color = getNodeColor(node, vizConfig);
        
        // Use MeshLambertMaterial to match default look
        const material = new THREE.MeshLambertMaterial({ 
            color, 
            transparent: true, 
            opacity: settings.opacity ?? 0.9 
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        // Scale is handled by the library if we don't set it? No, we return the object.
        // The library scales the object by `val` if we don't set scale?
        // Actually, the library applies `val` to the default sphere radius.
        // If we provide an object, we should probably scale it ourselves or let the library scale it?
        // The library does: `obj.scale.multiplyScalar(val)` if `nodeThreeObjectExtend` is true?
        // We set `nodeThreeObjectExtend={false}`.
        // So we must scale it.
        
        const size = getNodeRadius(node, vizConfig);
        mesh.scale.set(size, size, size);
        
        return mesh;
    }, [vizConfig, geometries]);

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
                    
                    // Link appearance - DISABLED when using batched rendering
                    linkVisibility={!useBatchedLinks}
                    linkColor={useBatchedLinks ? () => 'transparent' : linkColor}
                    linkWidth={useBatchedLinks ? 0 : (vizConfig?.edgeWidth ?? 0.3)}
                    linkOpacity={useBatchedLinks ? 0 : (vizConfig?.edgeOpacity ?? 0.4)}
                    linkResolution={2}
                    
                    // Physics
                    d3VelocityDecay={vizConfig?.d3VelocityDecay ?? 0.3}
                    warmupTicks={50}
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
            <DynamicLegend data={data} vizConfig={vizConfig} />

            {/* Zoom Controls */}
            <div className="absolute bottom-20 right-4 flex flex-col gap-2 z-10">
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
                <div className="absolute bottom-4 right-4 bg-slate-900/95 p-3 rounded border border-slate-700 z-10 max-w-sm">
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
