import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react';
import GraphSearch from './GraphSearch';

const getNodeColor = (d) => {
    if (d.isEvidence) return "#ffd700";
    if (d.type === 'entry') return "#9c27b0";
    if (d.isOnPath) return "#00ff88";
    if (d.isActivePath) return "#00bcd4"; // Cyan for active path
    if (d.visited) return "#2196f3";
    if (d.type === 'candidate') return "#9e9e9e";
    if (d.type === 'static') return "#333";
    return "#555";
};

const getLinkColor = (d) => {
    if (d.isOnPath) return "#00ff88";
    if (d.isActivePath) return "#00bcd4"; // Cyan for active path
    if (d.type === 'reference') return "#4caf50";
    return "#555";
};

const GraphVisualization = ({ graphData, onNodeClick, onNodeHover }) => {
    const svgRef = useRef(null);
    const gRef = useRef(null);
    const containerRef = useRef(null);
    const simulationRef = useRef(null);
    const zoomRef = useRef(null);
    const initializedRef = useRef(false);

    // Initialize D3
    useEffect(() => {
        if (!svgRef.current || !containerRef.current || initializedRef.current) return;
        initializedRef.current = true;

        const svg = d3.select(svgRef.current);
        const g = d3.select(gRef.current);
        
        // Initial dimensions
        const { width, height } = containerRef.current.getBoundingClientRect();

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
            
        svg.call(zoom);
        zoomRef.current = zoom;

        // Arrow marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 15)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#555");

        // Simulation
        simulationRef.current = d3.forceSimulation([])
            .force("link", d3.forceLink([]).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide(20))
            .on("tick", () => {
                g.selectAll(".link")
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                g.selectAll(".node")
                    .attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
            });

        // Handle Resize
        const resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                if (simulationRef.current) {
                    simulationRef.current.force("center", d3.forceCenter(width / 2, height / 2));
                    simulationRef.current.alpha(0.3).restart();
                }
            }
        });
        
        resizeObserver.observe(containerRef.current);

        return () => {
            if (simulationRef.current) simulationRef.current.stop();
            resizeObserver.disconnect();
        };
    }, []);

    const [adjacency, setAdjacency] = useState(new Map());

    // Update graph data
    useEffect(() => {
        if (!simulationRef.current || !gRef.current) return;
        if (!graphData || !graphData.nodes) return;

        // Pre-calculate adjacency
        const newAdjacency = new Map();
        graphData.links.forEach(l => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;
            
            if (!newAdjacency.has(sourceId)) newAdjacency.set(sourceId, new Set());
            if (!newAdjacency.has(targetId)) newAdjacency.set(targetId, new Set());
            
            newAdjacency.get(sourceId).add(targetId);
            newAdjacency.get(targetId).add(sourceId);
        });
        setAdjacency(newAdjacency);

        const g = d3.select(gRef.current);
        const simulation = simulationRef.current;

        // Merge new data with existing positions
        const oldNodesMap = new Map(simulation.nodes().map(d => [d.id, d]));
        const nodes = graphData.nodes.map(d => {
            const old = oldNodesMap.get(d.id);
            return old ? Object.assign(old, d) : { ...d };
        });

        const links = graphData.links.map(d => ({ ...d }));

        // Update simulation
        simulation.nodes(nodes);
        simulation.force("link").links(links);
        
        // Only reheat if structure changed significantly or it's the first load
        if (nodes.length !== oldNodesMap.size || (nodes.length > 0 && simulation.alpha() < 0.05)) {
            simulation.alpha(0.3).restart();
        }

        // === LINKS ===
        const link = g.selectAll(".link").data(links, d => 
            `${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`
        );
        
        link.exit().remove();
        
        link.enter()
            .append("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)")
          .merge(link)
            .attr("stroke", d => getLinkColor(d))
            .attr("stroke-opacity", d => d.isOnPath ? 1 : 0.3)
            .attr("stroke-width", d => d.isOnPath ? 3 : 1);

        // === NODES ===
        const node = g.selectAll(".node").data(nodes, d => d.id);
        
        node.exit().remove();
        
        const nodeEnter = node.enter()
            .append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
                .on("end", (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }))
            .on("click", (event, d) => { event.stopPropagation(); onNodeClick?.(d); })
            .on("mouseover", (event, d) => handleNodeMouseOver(event, d))
            .on("mouseout", handleNodeMouseOut);

        nodeEnter.append("circle").attr("stroke", "#fff").attr("stroke-width", 1.5);
        nodeEnter.append("text").attr("x", 8).attr("y", 3).attr("font-size", "10px").attr("fill", "#ccc");

        const nodeMerge = nodeEnter.merge(node);
        
        nodeMerge.select("circle")
            .attr("r", d => d.isEvidence ? 10 : 6)
            .attr("fill", d => getNodeColor(d));
        
        nodeMerge.select("text")
            .text(d => d.summary ? d.summary.substring(0, 15) + "..." : d.id.substring(0, 8));

    }, [graphData, onNodeClick, onNodeHover]);

    const handleZoom = (factor) => {
        if (!svgRef.current || !zoomRef.current) return;
        const svg = d3.select(svgRef.current);
        svg.transition().duration(300).call(zoomRef.current.scaleBy, factor);
    };

    const handleReset = () => {
        if (!svgRef.current || !zoomRef.current) return;
        const svg = d3.select(svgRef.current);
        svg.transition().duration(750).call(zoomRef.current.transform, d3.zoomIdentity);
    };

    const handleSearchSelect = (node) => {
        onNodeClick?.(node);
        // Center view on node
        if (!svgRef.current || !zoomRef.current || !node.x) return;
        
        const svg = d3.select(svgRef.current);
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        const scale = 1.5;
        const x = -node.x * scale + width / 2;
        const y = -node.y * scale + height / 2;
        
        svg.transition().duration(750).call(
            zoomRef.current.transform, 
            d3.zoomIdentity.translate(x, y).scale(scale)
        );
    };

    // Hover Effects
    const handleNodeMouseOver = (event, d) => {
        onNodeHover?.(event, d);
        
        const g = d3.select(gRef.current);
        const neighbors = adjacency.get(d.id) || new Set();
        
        // Fast update using data binding or direct selection if possible
        // But for now, let's optimize the selection filter
        
        g.selectAll(".link").attr("stroke-opacity", l => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;
            return (sourceId === d.id || targetId === d.id) ? 1 : 0.05;
        }).attr("stroke", l => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;
            return (sourceId === d.id || targetId === d.id) ? "#fff" : getLinkColor(l);
        });

        g.selectAll(".node").attr("opacity", n => {
            return (n.id === d.id || neighbors.has(n.id)) ? 1 : 0.1;
        });
    };

    const handleNodeMouseOut = () => {
        onNodeHover?.(null, null);
        
        const g = d3.select(gRef.current);
        
        // Reset styles
        g.selectAll(".link")
            .attr("stroke", d => getLinkColor(d))
            .attr("stroke-opacity", d => d.isOnPath ? 1 : 0.3);
            
        g.selectAll(".node").attr("opacity", 1);
    };

    return (
        <div ref={containerRef} className="w-full h-full bg-[#111] relative">
            <svg ref={svgRef} width="100%" height="100%">
                <g ref={gRef}></g>
            </svg>
            
            <div className="absolute top-4 right-4 z-10">
                <GraphSearch nodes={graphData?.nodes} onSelectNode={handleSearchSelect} />
            </div>

            <div className="absolute bottom-20 right-4 flex flex-col gap-2">
                <button onClick={() => handleZoom(1.2)} className="p-2 bg-gray-800 text-gray-300 rounded hover:bg-gray-700 hover:text-white shadow-lg border border-gray-700">
                    <ZoomIn size={16} />
                </button>
                <button onClick={() => handleZoom(0.8)} className="p-2 bg-gray-800 text-gray-300 rounded hover:bg-gray-700 hover:text-white shadow-lg border border-gray-700">
                    <ZoomOut size={16} />
                </button>
                <button onClick={handleReset} className="p-2 bg-gray-800 text-gray-300 rounded hover:bg-gray-700 hover:text-white shadow-lg border border-gray-700">
                    <Maximize size={16} />
                </button>
            </div>
        </div>
    );
};

export default GraphVisualization;
