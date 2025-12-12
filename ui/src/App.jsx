import React, { useState, useEffect, useRef, useMemo } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import GraphVisualization from './components/GraphVisualization';
import ControlPanel from './components/ControlPanel';
import LogViewer from './components/LogViewer';
import NodeDetails from './components/NodeDetails';
import HistoryModal from './components/HistoryModal';
import ErrorBoundary from './components/ErrorBoundary';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8002";
const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8002";

// Stable empty array to avoid re-renders
const EMPTY_ARRAY = [];

function App() {
  const [status, setStatus] = useState({ running: false, last_query: "" });
  const [allEvents, setAllEvents] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(100); // ms per step
  const [mode, setMode] = useState('live'); // 'live' or 'file'
  const [currentRunId, setCurrentRunId] = useState(null);
  
  // Multi-query support
  const [queries, setQueries] = useState([]);
  const [selectedQueryId, setSelectedQueryId] = useState(null);

  // Derived state for the current point in time
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [metrics, setMetrics] = useState({ cost: 0, calls: 0, tokens: 0, shortcuts: 0 });
  const [evidence, setEvidence] = useState([]);
  const [queue, setQueue] = useState([]);
  const [currentAction, setCurrentAction] = useState("Ready.");

  const [tooltip, setTooltip] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  
  const [ws, setWs] = useState(null);
  const fileInputRef = useRef(null);
  const [showLogs, setShowLogs] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [runHistory, setRunHistory] = useState([]);

  const [resources, setResources] = useState({ indices: [], workloads: [] });
  
  const [runConfig, setRunConfig] = useState({
      index: "data/hcg_medical.json",
      inputType: "query", // 'query' or 'workload'
      query: "What is the capital of France?",
      workload: "",
      model: "openrouter/x-ai/grok-4.1-fast",
      entry_points: 5
  });

  // Full Graph Visualization
  const [showFullGraph, setShowFullGraph] = useState(false);
  const [fullGraphData, setFullGraphData] = useState(null);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);

  const fetchFullGraph = async (index) => {
      if (!index) return;
      setIsLoadingGraph(true);
      try {
          const res = await fetch(`${API_BASE}/api/graph?index=${encodeURIComponent(index)}`);
          if (!res.ok) throw new Error("Failed to load graph");
          const data = await res.json();
          setFullGraphData(data);
      } catch (e) {
          console.error("Error loading full graph:", e);
          setFullGraphData(null);
      } finally {
          setIsLoadingGraph(false);
      }
  };

  useEffect(() => {
      if (runConfig.index) {
          // Clear previous data when index changes
          setFullGraphData(null);
          fetchFullGraph(runConfig.index);
      }
  }, [runConfig.index]);

  const availableModels = [
      "openrouter/x-ai/grok-4.1-fast",
      "openrouter/google/gemini-2.0-flash-001",
      "gpt-4o",
      "gpt-4o-mini",
      "claude-3-5-sonnet-20240620"
  ];

  const fetchHistory = async () => {
      try {
          const res = await fetch(`${API_BASE}/api/runs`);
          const data = await res.json();
          setRunHistory(data.history || []);
      } catch (e) {
          console.error("Failed to fetch history", e);
      }
  };

  useEffect(() => {
      if (showHistory) {
          fetchHistory();
      }
  }, [showHistory]);

  const [finalAnswer, setFinalAnswer] = useState(null);

  const loadRun = (runId) => {
      // Validate Index
      const runMeta = runHistory.find(r => r.run_id === runId);
      if (runMeta && runMeta.index && runMeta.index !== runConfig.index) {
          alert(`Cannot load trace. \n\nThis run used index: ${runMeta.index}\nCurrent selection: ${runConfig.index}\n\nPlease select the correct index in the sidebar.`);
          return;
      }

      if (ws) ws.close();
      setAllEvents([]);
      setCurrentStep(0);
      setMode('history');
      setShowHistory(false);
      setSelectedQueryId(null);
      
      const socket = new WebSocket(`${WS_BASE}/ws/${runId}`);
      
      let messageBuffer = [];
      let batchTimeout = null;

      const processBatch = () => {
          if (messageBuffer.length > 0) {
              const batch = [...messageBuffer];
              messageBuffer = [];
              
              setAllEvents(prev => {
                  const next = [...prev, ...batch];
                  // Auto-scroll if we were at the end or it's the start
                  // We use a timeout to allow state to settle, or just rely on the useEffect auto-scroll logic if we enable it for history
                  // For now, let's force scroll to end during loading
                  setCurrentStep(next.length - 1);
                  return next;
              });
          }
          batchTimeout = null;
      };

      socket.onmessage = (event) => {
          try {
              const newEvent = JSON.parse(event.data);
              messageBuffer.push(newEvent);
              
              if (!batchTimeout) {
                  batchTimeout = setTimeout(processBatch, 100); // Batch updates every 100ms
              }
          } catch (e) {
              console.error("WS Parse Error", e);
          }
      };
      
      socket.onclose = () => {
          if (batchTimeout) {
              clearTimeout(batchTimeout);
              processBatch(); // Flush remaining
          }
      };

      setWs(socket);
      setCurrentRunId(runId);
  };

  // Fetch resources
  useEffect(() => {
      const loadResources = () => {
        fetch(`${API_BASE}/api/resources`)
            .then(res => res.json())
            .then(data => {
                console.log("Loaded resources:", data);
                if (data.indices && data.indices.length > 0) {
                    console.log("First index:", data.indices[0]);
                }
                setResources(data);
                if (data.indices && data.indices.length > 0) {
                    // Prefer hcg_medical.json if available
                    const medicalIndex = data.indices.find(i => i.includes('hcg_medical.json'));
                    console.log("Selected medical index:", medicalIndex);
                    setRunConfig(prev => ({ ...prev, index: medicalIndex || data.indices[0] }));
                }
                if (data.workloads && data.workloads.length > 0) {
                    setRunConfig(prev => ({ ...prev, workload: data.workloads[0] }));
                }
            })
            .catch(err => console.error("Failed to fetch resources", err));
      };
      
      loadResources();
      // Retry once after 1s in case server wasn't ready
      setTimeout(loadResources, 1000);
  }, []);

  // Poll status
  useEffect(() => {
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket cleanup
  useEffect(() => {
      return () => {
          if (ws) ws.close();
      }
  }, [ws]);

  // Parse Queries from All Events
  useEffect(() => {
      const queryMap = new Map();
      // First pass: find start events
      allEvents.forEach(e => {
          if (e.event_type === 'query_start') {
              queryMap.set(e.query_id, {
                  id: e.query_id,
                  text: e.query_text || e.data?.query || "Unknown Query",
                  events: []
              });
          }
      });
      
      // If no explicit query_start, maybe create a default one?
      if (queryMap.size === 0 && allEvents.length > 0) {
          queryMap.set("default", { id: "default", text: "Single Query", events: [] });
      }

      // Second pass: assign events
      allEvents.forEach(e => {
          if (e.query_id && queryMap.has(e.query_id)) {
              queryMap.get(e.query_id).events.push(e);
          } else if (queryMap.has("default")) {
              queryMap.get("default").events.push(e);
          }
      });

      const queryList = Array.from(queryMap.values());
      setQueries(queryList);
      
      // Auto-select latest query if none selected or if we are in live mode
      if (queryList.length > 0) {
          const currentExists = queryList.find(q => q.id === selectedQueryId);
          if (!selectedQueryId || mode === 'live' || !currentExists) {
             // If live, we usually want to follow the latest query
             // OR if the current selection is no longer valid (e.g. switched from default to real ID)
             setSelectedQueryId(queryList[queryList.length - 1].id);
          }
      }
  }, [allEvents, mode, selectedQueryId]);

  // Get current active events based on selection
  const activeEvents = useMemo(() => {
      if (!selectedQueryId) return [];
      const q = queries.find(q => q.id === selectedQueryId);
      return q ? q.events : [];
  }, [queries, selectedQueryId]);

  const logs = useMemo(() => {
      return allEvents.filter(e => e.event_type === 'stdout' || e.event_type === 'stderr');
  }, [allEvents]);

  // Playback Loop
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= activeEvents.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, activeEvents.length, playbackSpeed]);

  // Re-process graph when step changes
  const processingRef = useRef(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
        if (processingRef.current) return;
        processingRef.current = true;
        
        try {
            if (activeEvents.length > 0) {
                // Ensure step is valid for new query
                const safeStep = Math.min(currentStep, activeEvents.length - 1);
                processGraph(activeEvents, safeStep);
            } else if (showFullGraph && fullGraphData) {
                // Show full graph even if no events
                processGraph([], -1);
            } else {
                // Clear graph if no events
                setGraphData({ nodes: [], links: [] });
                setMetrics({ cost: 0, calls: 0, tokens: 0, shortcuts: 0 });
                setEvidence([]);
            }
        } catch (e) {
            console.error("Error processing graph:", e);
        } finally {
            processingRef.current = false;
        }
    }, 10); // Low debounce (10ms) because batching handles the flood
    return () => clearTimeout(timer);
  }, [currentStep, activeEvents, showFullGraph, fullGraphData]);

  // Reset step when query changes (unless live)
  useEffect(() => {
      if (mode !== 'live') {
          setCurrentStep(0);
      }
  }, [selectedQueryId, mode]);

  // Auto-scroll in live mode
  useEffect(() => {
      if (mode === 'live' && activeEvents.length > 0) {
          setCurrentStep(activeEvents.length - 1);
      }
  }, [activeEvents.length, mode, selectedQueryId]);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      console.error("Status fetch failed", e);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const text = e.target.result;
        // Hack to fix Python NaN in JSON
        const fixedText = text.replace(/: NaN/g, ': null');
        const lines = fixedText.split('\n');
        const events = [];
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                events.push(JSON.parse(line));
            } catch (err) {
                console.error("Error parsing line", err);
            }
        }
        setAllEvents(events);
        setCurrentStep(0);
        setMode('file');
        setIsPlaying(false);
        if (ws) {
            ws.close();
            setWs(null);
        }
    };
    reader.readAsText(file);
  };

  const runQuery = async () => {
    try {
      setMode('live');
      if (ws) ws.close();
      
      const payload = {
          index: runConfig.index,
          model: runConfig.model,
          ranking_model: runConfig.ranking_model || null,
          admittance_model: runConfig.admittance_model || null,
          termination_model: runConfig.termination_model || null,
          entry_points: runConfig.entry_points
      };
      
      if (runConfig.inputType === 'query') {
          payload.query = runConfig.query;
      } else {
          payload.workload_file = runConfig.workload;
      }

      const res = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      
      // Reset UI
      setAllEvents([]);
      setCurrentStep(0);
      setIsPlaying(true);
      
      // Connect WS
      const socket = new WebSocket(`${WS_BASE}/ws/${data.run_id}`);
      
      let messageBuffer = [];
      let batchTimeout = null;

      const processBatch = () => {
          if (messageBuffer.length > 0) {
              const batch = [...messageBuffer];
              messageBuffer = [];
              
              setAllEvents(prev => {
                  const next = [...prev, ...batch];
                  // Always auto-scroll to latest in live run
                  setCurrentStep(next.length - 1);
                  return next;
              });
          }
          batchTimeout = null;
      };

      socket.onmessage = (event) => {
          // console.log("WS Message:", event.data); // Reduce log spam
          try {
              const newEvent = JSON.parse(event.data);
              messageBuffer.push(newEvent);
              
              if (!batchTimeout) {
                  batchTimeout = setTimeout(processBatch, 100); // Batch updates every 100ms
              }
          } catch (e) {
              console.error("WS Parse Error", e);
          }
      };

      socket.onclose = () => {
          if (batchTimeout) {
              clearTimeout(batchTimeout);
              processBatch(); // Flush remaining
          }
      };

      setWs(socket);
      setCurrentRunId(data.run_id);
      
      fetchStatus();
    } catch (e) {
      console.error("Run failed", e);
    }
  };

  const stopQuery = async () => {
    try {
      if (currentRunId) {
          await fetch(`${API_BASE}/api/stop/${currentRunId}`, { method: 'POST' });
      } else {
          // Fallback or legacy stop
          await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
      }
      fetchStatus();
    } catch (e) {
      console.error("Stop failed", e);
    }
  };

  const processGraph = (events, limitIndex) => {
    const nodes = new Map();
    const links = [];
    const linkSet = new Set(); // Track existing links to avoid duplicates
    const parentMap = new Map();
    let currentMetrics = { cost: 0, calls: 0, tokens: 0, shortcuts: 0 };
    let currentEvidence = [];
    const evidenceSet = new Set(); // Track evidence to avoid duplicates
    let currentQueue = []; // We need to track queue state
    let currentAnswer = null;
    let lastAction = "Ready.";
    let lastExploredNodeId = null;
    
    // Track active nodes to filter graph edges
    const activeNodeIds = new Set();
    const nodeStates = new Map(); // id -> { score, visited, isEvidence, ... }
    const rootEdges = []; // List of target IDs connected to ROOT

    // Initialize with full graph if available
    if (showFullGraph && fullGraphData) {
        fullGraphData.nodes.forEach(n => {
            nodes.set(n.id, { ...n, visited: false, type: 'static', score: 0 });
        });
        fullGraphData.links.forEach(l => {
            // Ensure source and target exist to prevent D3 crashes
            if (nodes.has(l.source) && nodes.has(l.target)) {
                links.push({ ...l, isOnPath: false });
                linkSet.add(`${l.source}-${l.target}`);
            }
        });
    }

    // 1. Scan events to determine active nodes and their state
    for (let i = 0; i <= limitIndex && i < events.length; i++) {
        const e = events[i];
        
        // Update Action
        if (e.event_type) lastAction = `${e.event_type}`;

        // Metrics
        if (e.event_type === 'llm_score_request' || e.event_type === 'llm_interaction') {
            currentMetrics.calls++;
        }

        if (e.event_type === 'trace_init') {
            lastAction = "Trace Initialized";
        }

        if (e.event_type === 'search_step') {
            const { node_id } = e.data;
            lastExploredNodeId = node_id;
            activeNodeIds.add(node_id);
            
            const state = nodeStates.get(node_id) || {};
            state.visited = true;
            state.type = 'visited';
            nodeStates.set(node_id, state);
            
            // Remove from queue
            currentQueue = currentQueue.filter(q => q.id !== node_id);
        }

        if (e.event_type === 'node_evaluation') {
            const { node_id, score, is_relevant, reasoning, metadata } = e.data;
            activeNodeIds.add(node_id);
            
            const state = nodeStates.get(node_id) || {};
            state.score = score;
            if (is_relevant) {
                state.isEvidence = true;
                state.type = 'evidence';
                
                if (!evidenceSet.has(node_id)) {
                    currentEvidence.push({
                        node_id,
                        score,
                        reasoning,
                        summary: metadata?.summary || ""
                    });
                    evidenceSet.add(node_id);
                }
            }
            nodeStates.set(node_id, state);
        }

        if (e.event_type === 'evidence_collected') {
            const { node_id, content, score } = e.data;
            activeNodeIds.add(node_id);
            
            const state = nodeStates.get(node_id) || {};
            state.isEvidence = true;
            state.type = 'evidence';
            state.score = score;
            nodeStates.set(node_id, state);
            
            if (!evidenceSet.has(node_id)) {
                currentEvidence.push({
                    node_id,
                    score,
                    summary: content,
                    reasoning: "Collected via Admittance"
                });
                evidenceSet.add(node_id);
            }
        }

        if (e.event_type === 'frontier_update') {
            let { parent_id, candidates } = e.data;
            
            // Handle empty parent_id (Entry Points)
            if (!parent_id) {
                parent_id = "ROOT";
                // Capture entry points for Root Edges
                if (candidates) candidates.forEach(c => rootEdges.push(c.id));
            }
            
            activeNodeIds.add(parent_id);

            if (candidates && Array.isArray(candidates)) {
                candidates.forEach(cand => {
                    const { id, score, summary } = cand;
                    if (!id) return;
                    
                    activeNodeIds.add(id);
                    const state = nodeStates.get(id) || {};
                    if (score !== undefined && score !== null) state.score = score;
                    // Only set summary if we don't have graph data later
                    if (summary) state._traceSummary = summary; 
                    if (!state.visited && !state.isEvidence) state.type = 'candidate';
                    nodeStates.set(id, state);

                    // Track parent for path reconstruction
                    if (parent_id) {
                        parentMap.set(id, parent_id);
                    }

                    // Add to queue if not visited
                    // Note: We use trace summary for queue for now, but could upgrade
                    if (!state.visited && !currentQueue.find(q => q.id === id)) {
                        currentQueue.push({ id, score: state.score, summary: summary || "" });
                    } else if (!state.visited) {
                        // Update score in queue
                        const qItem = currentQueue.find(q => q.id === id);
                        if (qItem) {
                            qItem.score = state.score;
                            qItem.summary = summary || qItem.summary;
                        }
                    }
                });
                
                // Sort queue
                currentQueue.sort((a, b) => (b.score || 0) - (a.score || 0));
            }
        }

        if (e.event_type === 'result') {
            const { answer, path } = e.data;
            currentAnswer = answer;
            
            // Highlight path
            if (path && Array.isArray(path)) {
                path.forEach((nodeId) => {
                    activeNodeIds.add(nodeId);
                    const state = nodeStates.get(nodeId) || {};
                    state.isOnPath = true;
                    nodeStates.set(nodeId, state);
                });
            }
        }
    }

    // 2. Build Nodes (Merging Graph Data + Trace State)
    // Always add ROOT if we have root edges
    if (rootEdges.length > 0) {
        nodes.set("ROOT", { id: "ROOT", type: 'entry', summary: "Query", visited: true });
    }

    activeNodeIds.forEach(id => {
        if (id === "ROOT") return;

        let nodeData = { id, type: 'candidate', summary: "", visited: false, score: 0 };
        
        // Prefer Full Graph Data
        if (fullGraphData && fullGraphData.nodes) {
            const graphNode = fullGraphData.nodes.find(n => n.id === id);
            if (graphNode) {
                // Use graph data as base
                nodeData = { ...graphNode, ...nodeData, summary: graphNode.summary, type: 'candidate' }; 
            }
        }

        // Merge Dynamic State
        const state = nodeStates.get(id);
        if (state) {
            // If we have a trace summary and NO graph summary, use trace summary
            if (state._traceSummary && !nodeData.summary) {
                nodeData.summary = state._traceSummary;
            }
            delete state._traceSummary;
            Object.assign(nodeData, state);
        }
        
        nodes.set(id, nodeData);
    });

    // 3. Build Links
    // A. Graph Edges (The "All Edges" requirement)
    if (fullGraphData && fullGraphData.links) {
        fullGraphData.links.forEach(l => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;

            if (nodes.has(sourceId) && nodes.has(targetId)) {
                const key = `${sourceId}-${targetId}`;
                if (!linkSet.has(key)) {
                    links.push({ ...l, source: sourceId, target: targetId });
                    linkSet.add(key);
                }
            }
        });
    }

    // B. Root Edges (The "Exception" requirement)
    rootEdges.forEach(targetId => {
        if (nodes.has(targetId)) {
             const key = `ROOT-${targetId}`;
             if (!linkSet.has(key)) {
                 links.push({ source: "ROOT", target: targetId, type: 'entry' });
                 linkSet.add(key);
             }
        }
    });

    // C. Trace Edges (Fallback for missing graph data)
    parentMap.forEach((parentId, childId) => {
        if (parentId === "ROOT") return; // Handled above
        
        if (nodes.has(parentId) && nodes.has(childId)) {
             const key = `${parentId}-${childId}`;
             const revKey = `${childId}-${parentId}`;
             
             if (!linkSet.has(key) && !linkSet.has(revKey)) {
                 links.push({ source: parentId, target: childId, type: 'trace' });
                 linkSet.add(key);
             }
        }
    });

    // Highlight path edges
    if (currentAnswer && events.length > 0) {
        // Find result event
        const resEvent = events.find(e => e.event_type === 'result');
        if (resEvent && resEvent.data.path) {
             const path = resEvent.data.path;
             path.forEach((nodeId, idx) => {
                if (idx > 0) {
                    const prevId = path[idx-1];
                    const link = links.find(l => 
                        (l.source === prevId && l.target === nodeId) || 
                        (l.source.id === prevId && l.target.id === nodeId)
                    );
                    if (link) link.isOnPath = true;
                }
            });
        }
    }
    
    setQueue(currentQueue);

    // Highlight path to the last explored node if no answer yet
    if (lastExploredNodeId && !currentAnswer) {
        let curr = lastExploredNodeId;
        const visitedPath = new Set(); // Prevent infinite loops
        
        while (curr && parentMap.has(curr)) {
            if (visitedPath.has(curr)) {
                break;
            }
            visitedPath.add(curr);
            
            const parent = parentMap.get(curr);
            
            // Mark node
            if (nodes.has(curr)) nodes.get(curr).isActivePath = true;
            if (nodes.has(parent)) nodes.get(parent).isActivePath = true;

            // Mark edge
            const link = links.find(l => 
                (l.source === parent && l.target === curr) ||
                (l.source === curr && l.target === parent) ||
                (l.source.id === parent && l.target.id === curr) ||
                (l.source.id === curr && l.target.id === parent)
            );
            if (link) link.isActivePath = true;

            curr = parent;
        }
    }

    setMetrics(currentMetrics);
    setEvidence(currentEvidence);
    setCurrentAction(lastAction);
    setFinalAnswer(currentAnswer);
    
    setGraphData({
      nodes: Array.from(nodes.values()),
      links: links
    });
  };

  const handleNodeHover = (event, node) => {
    if (!event && !node) {
        setTooltip(null);
        return;
    }
    if (node) {
        setTooltip({
            x: event.pageX,
            y: event.pageY,
            content: { id: node.id, score: node.score, type: node.type, summary: node.summary }
        });
    } else if (event) {
        // Just update position if tooltip exists
        setTooltip(prev => prev ? ({ ...prev, x: event.pageX, y: event.pageY }) : null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans flex flex-col">
      <Header 
        status={status}
        mode={mode}
        setMode={setMode}
        showHistory={showHistory}
        setShowHistory={setShowHistory}
        fileInputRef={fileInputRef}
        handleFileUpload={handleFileUpload}
        showLogs={showLogs}
        setShowLogs={setShowLogs}
        runHistory={runHistory}
        loadRun={loadRun}
        runConfig={runConfig}
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar 
            runConfig={runConfig}
            setRunConfig={setRunConfig}
            resources={resources}
            availableModels={availableModels}
            runQuery={runQuery}
            stopQuery={stopQuery}
            status={status}
            queries={queries}
            selectedQueryId={selectedQueryId}
            setSelectedQueryId={setSelectedQueryId}
            metrics={metrics}
            currentAction={currentAction}
            finalAnswer={finalAnswer}
            evidence={evidence}
            queue={queue}
            showFullGraph={showFullGraph}
            setShowFullGraph={setShowFullGraph}
            isLoadingGraph={isLoadingGraph}
            fullGraphData={fullGraphData}
        />

        <div className="flex-1 bg-black relative flex flex-col">
          <ControlPanel 
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            currentStep={currentStep}
            setCurrentStep={setCurrentStep}
            maxStep={Math.max(0, activeEvents.length - 1)}
            activeEventsCount={activeEvents.length}
          />

          <div className="flex-1 relative overflow-hidden">
            <ErrorBoundary>
                <GraphVisualization 
                    graphData={graphData}
                    onNodeClick={setSelectedNode}
                    onNodeHover={handleNodeHover}
                />
            </ErrorBoundary>
            
            {/* Overlay Stats */}
            <div className="absolute top-4 right-4 bg-gray-900/80 p-4 rounded border border-gray-700 backdrop-blur pointer-events-none">
                <div className="text-xs text-gray-400">Nodes</div>
                <div className="text-xl font-bold">{graphData.nodes.length}</div>
            </div>
            
            {/* Legend */}
            <div className="absolute top-4 left-4 bg-gray-900/90 p-3 rounded border border-gray-700 backdrop-blur pointer-events-none text-xs">
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-purple-500"></div> Entry Point
                </div>
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div> Visited
                </div>
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-yellow-400"></div> Evidence node
                </div>
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-gray-400"></div> Candidate / Queue
                </div>
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-5 h-1 bg-green-400"></div> Answer Path
                </div>
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-5 h-1 bg-gray-600"></div> Traversal Edge
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-5 h-1 border-t border-green-600 opacity-50"></div> Reference Edge
                </div>
            </div>
          </div>

          {showLogs && <LogViewer logs={logs} onClose={() => setShowLogs(false)} />}
          
          {showHistory && (
            <HistoryModal 
                history={runHistory} 
                onClose={() => setShowHistory(false)} 
                onLoadRun={loadRun} 
            />
          )}
        </div>
      </div>
      
      {/* Tooltip */}
      {tooltip && (
        <div 
            className="fixed z-50 bg-gray-900 border border-gray-600 p-2 rounded shadow-lg text-xs pointer-events-none"
            style={{ left: tooltip.x + 10, top: tooltip.y + 10 }}
        >
            <div className="font-bold text-blue-400 mb-1">{tooltip.content.id.substring(0, 12)}...</div>
            {tooltip.content.score !== undefined && <div>Score: <span className="font-mono">{typeof tooltip.content.score === 'number' ? tooltip.content.score.toFixed(3) : 'N/A'}</span></div>}
            {tooltip.content.type && <div>Type: {tooltip.content.type}</div>}
            {tooltip.content.summary && <div className="mt-1 text-gray-400 max-w-xs">{tooltip.content.summary}</div>}
        </div>
      )}

      {selectedNode && (
          <NodeDetails node={selectedNode} onClose={() => setSelectedNode(null)} />
      )}
    </div>
  );
}

export default App;
