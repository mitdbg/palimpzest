import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

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
    const simulationRef = useRef(null);
    const initializedRef = useRef(false);

    // Initialize D3 once
    useEffect(() => {
        if (!svgRef.current || initializedRef.current) return;
        initializedRef.current = true;

        const svg = d3.select(svgRef.current);
        const g = d3.select(gRef.current);
        const width = 800;
        const height = 600;

        // Zoom
        svg.call(d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform)));

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

        return () => {
            if (simulationRef.current) {
                simulationRef.current.stop();
            }
        };
    }, []);

    // Update graph data
    useEffect(() => {
        if (!simulationRef.current || !gRef.current) return;
        if (!graphData || !graphData.nodes) return;

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
        
        // Only reheat if structure changed
        if (nodes.length !== oldNodesMap.size) {
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
            .on("mouseover", (event, d) => onNodeHover?.(event, d))
            .on("mouseout", () => onNodeHover?.(null, null));

        nodeEnter.append("circle").attr("stroke", "#fff").attr("stroke-width", 1.5);
        nodeEnter.append("text").attr("x", 8).attr("y", 3).attr("font-size", "10px").attr("fill", "#ccc");

        const nodeMerge = nodeEnter.merge(node);
        
        nodeMerge.select("circle")
            .attr("r", d => d.isEvidence ? 10 : 6)
            .attr("fill", d => getNodeColor(d));
        
        nodeMerge.select("text")
            .text(d => d.summary ? d.summary.substring(0, 15) + "..." : d.id.substring(0, 8));

    }, [graphData, onNodeClick, onNodeHover]);

    return (
        <svg ref={svgRef} width="100%" height="100%" style={{ background: '#111' }}>
            <g ref={gRef}></g>
        </svg>
    );
};

export default GraphVisualization;
