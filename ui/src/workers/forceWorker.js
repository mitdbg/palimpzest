/**
 * Web Worker for d3-force simulation.
 * Runs physics calculations off the main thread.
 * 
 * Uses typed arrays for efficient data transfer.
 */
import * as d3 from 'd3';

let simulation = null;
let nodes = [];
let links = [];
let nodeIdToIndex = new Map();
let isRunning = false;

// Configuration
let config = {
    chargeStrength: -200,
    chargeDistanceMax: 500,
    linkDistance: 90,
    linkStrength: 0.7,
    centerStrength: 0.04,
    velocityDecay: 0.3,
    alphaDecay: 0.01,
};

function initSimulation() {
    if (simulation) {
        simulation.stop();
    }

    simulation = d3.forceSimulation(nodes)
        .velocityDecay(config.velocityDecay)
        .alphaDecay(config.alphaDecay)
        .force('charge', d3.forceManyBody()
            .strength(config.chargeStrength)
            .distanceMax(config.chargeDistanceMax)
        )
        .force('link', d3.forceLink(links)
            .id(d => d.id)
            .distance(config.linkDistance)
            .strength(config.linkStrength)
        )
        .force('x', d3.forceX(0).strength(config.centerStrength))
        .force('y', d3.forceY(0).strength(config.centerStrength))
        .on('tick', onTick);

    // Keep simulation warm
    simulation.alphaTarget(0.01);
    isRunning = true;
}

// Throttle position updates to ~15 fps (66ms)
let lastSendTime = 0;
const SEND_INTERVAL = 66;

function onTick() {
    if (!isRunning) return;
    
    const now = performance.now();
    if (now - lastSendTime < SEND_INTERVAL) return;
    lastSendTime = now;
    
    // Send positions as typed array (zero-copy transfer)
    const positions = new Float32Array(nodes.length * 2);
    for (let i = 0; i < nodes.length; i++) {
        positions[i * 2] = nodes[i].x || 0;
        positions[i * 2 + 1] = nodes[i].y || 0;
    }
    
    self.postMessage({ type: 'positions', positions }, [positions.buffer]);
}

function handleMessage(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'init': {
            // Receive node IDs and initial positions
            const { nodeIds, nodePositions, linkData, cfg } = data;
            
            // Update config
            if (cfg) Object.assign(config, cfg);
            
            // Create nodes from typed arrays
            nodes = [];
            nodeIdToIndex.clear();
            for (let i = 0; i < nodeIds.length; i++) {
                const id = nodeIds[i];
                nodeIdToIndex.set(id, i);
                nodes.push({
                    id,
                    x: nodePositions ? nodePositions[i * 2] : Math.random() * 1000 - 500,
                    y: nodePositions ? nodePositions[i * 2 + 1] : Math.random() * 1000 - 500,
                });
            }
            
            // Create links from typed arrays
            // linkData: [sourceIdx, targetIdx, distance, strength, ...]
            links = [];
            if (linkData && linkData.length > 0) {
                const numLinks = linkData.length / 4;
                for (let i = 0; i < numLinks; i++) {
                    const sourceIdx = linkData[i * 4];
                    const targetIdx = linkData[i * 4 + 1];
                    // distance and strength are packed but we use global config for now
                    if (sourceIdx < nodes.length && targetIdx < nodes.length) {
                        links.push({
                            source: nodes[sourceIdx].id,
                            target: nodes[targetIdx].id,
                        });
                    }
                }
            }
            
            initSimulation();
            
            self.postMessage({ 
                type: 'initialized', 
                nodeCount: nodes.length,
                linkCount: links.length,
            });
            break;
        }

        case 'updateConfig': {
            Object.assign(config, data);
            if (simulation) {
                const charge = simulation.force('charge');
                if (charge) {
                    charge.strength(config.chargeStrength);
                    charge.distanceMax(config.chargeDistanceMax);
                }
                const link = simulation.force('link');
                if (link) {
                    link.distance(config.linkDistance);
                    link.strength(config.linkStrength);
                }
                const fx = simulation.force('x');
                const fy = simulation.force('y');
                if (fx) fx.strength(config.centerStrength);
                if (fy) fy.strength(config.centerStrength);
                
                simulation.velocityDecay(config.velocityDecay);
                simulation.alpha(0.3).restart();
            }
            break;
        }

        case 'pause': {
            isRunning = false;
            if (simulation) simulation.stop();
            break;
        }

        case 'resume': {
            isRunning = true;
            if (simulation) simulation.alpha(0.1).restart();
            break;
        }

        case 'dragNode': {
            const idx = nodeIdToIndex.get(data.id);
            if (idx !== undefined && nodes[idx]) {
                nodes[idx].x = data.x;
                nodes[idx].y = data.y;
                nodes[idx].fx = data.x;
                nodes[idx].fy = data.y;
                if (simulation && isRunning) {
                    simulation.alpha(0.3).restart();
                }
            }
            break;
        }

        case 'pinNode': {
            const idx = nodeIdToIndex.get(data.id);
            if (idx !== undefined && nodes[idx]) {
                nodes[idx].fx = data.x;
                nodes[idx].fy = data.y;
            }
            break;
        }

        case 'unpinNode': {
            const idx = nodeIdToIndex.get(data.id);
            if (idx !== undefined && nodes[idx]) {
                nodes[idx].fx = null;
                nodes[idx].fy = null;
            }
            break;
        }

        case 'stop': {
            isRunning = false;
            if (simulation) {
                simulation.stop();
                simulation = null;
            }
            break;
        }
    }
}

self.onmessage = handleMessage;
