import React, { useState, useEffect, useRef, useMemo } from 'react';
import GraphVisualization from './components/GraphVisualization';
import ControlPanel from './components/ControlPanel';
import LogViewer from './components/LogViewer';
import NodeDetails from './components/NodeDetails';
import ErrorBoundary from './components/ErrorBoundary';
import HistorySidebar from './components/HistorySidebar';
import ChatInput from './components/ChatInput';
import RunDetails from './components/RunDetails';
import SettingsModal from './components/SettingsModal';
import ReasoningFeed from './components/ReasoningFeed';
import Toast from './components/Toast';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8002";
const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8002";

// Simple Logger
const logger = {
    info: (msg, data) => console.log(`[INFO] ${new Date().toISOString()} - ${msg}`, data || ''),
    error: (msg, err) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`, err || ''),
    warn: (msg, data) => console.warn(`[WARN] ${new Date().toISOString()} - ${msg}`, data || '')
};

function App() {
  const [status, setStatus] = useState({ running: false, last_query: "" });
  const [allEvents, setAllEvents] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(100); // ms per step
  const [mode, setMode] = useState('live'); // 'live' or 'file' or 'history'
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
  const [showLogs, setShowLogs] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [runHistory, setRunHistory] = useState([]);

  const [resources, setResources] = useState({ indices: [], workloads: [] });
  
  const [runConfig, setRunConfig] = useState(() => {
      const saved = localStorage.getItem('graphrag_runConfig');
      return saved ? JSON.parse(saved) : {
          index: "data/hcg_medical.json",
          inputType: "query",
          query: "",
          workload: "",
          model: "openrouter/x-ai/grok-4.1-fast",
          entry_points: 5,
          ranking_model: "",
          admittance_model: "",
          termination_model: "",
          max_steps: 200,
          edge_type: ""
      };
  });

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [devMode, setDevMode] = useState(() => {
      return localStorage.getItem('graphrag_devMode') === 'true';
  });

  // Persist settings
  useEffect(() => {
      localStorage.setItem('graphrag_runConfig', JSON.stringify(runConfig));
  }, [runConfig]);

  useEffect(() => {
      localStorage.setItem('graphrag_devMode', devMode);
  }, [devMode]);

  const [toast, setToast] = useState(null); // { message, type }

  const showToast = (message, type = 'success') => {
      setToast({ message, type });
  };

  // Full Graph Visualization
  const [showFullGraph, setShowFullGraph] = useState(false);
  const [fullGraphData, setFullGraphData] = useState(null);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);

  const fetchFullGraph = async (index) => {
      if (!index) return;
      setIsLoadingGraph(true);
      logger.info(`Fetching full graph for index: ${index}`);
      try {
          // First, tell backend to load this graph
          const loadRes = await fetch(`${API_BASE}/api/load_graph`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ index })
          });
          if (!loadRes.ok) {
              const err = await loadRes.json();
              throw new Error(err.detail || "Failed to switch graph");
          }

          // Then fetch the payload
          const res = await fetch(`${API_BASE}/api/graph?index=${encodeURIComponent(index)}`);
          if (!res.ok) throw new Error("Failed to load graph payload");
          const data = await res.json();
          logger.info(`Graph loaded. Nodes: ${data.nodes?.length}, Links: ${data.links?.length}`);
          setFullGraphData(data);
      } catch (e) {
          logger.error("Error loading full graph", e);
          showToast("Failed to load graph: " + e.message, "error");
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
          showToast("Failed to load history", "error");
      }
  };

  useEffect(() => {
      fetchHistory();
  }, []);

  const [finalAnswer, setFinalAnswer] = useState(null);

  const loadRun = (runId) => {
      // Validate Index
      const runMeta = runHistory.find(r => r.run_id === runId);
      if (runMeta && runMeta.index && runMeta.index !== runConfig.index) {
          if (confirm(`This run used index: ${runMeta.index}\nCurrent selection: ${runConfig.index}\n\nSwitch index to load trace?`)) {
              setRunConfig(prev => ({ ...prev, index: runMeta.index }));
              // The useEffect for runConfig.index will trigger graph load
          } else {
              return;
          }
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
              showToast("Error parsing history update", "error");
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
      
      // Sync query input
      const run = runHistory.find(r => r.run_id === runId);
      if (run && run.query) {
          setRunConfig(prev => ({ ...prev, query: run.query }));
      }
  };

  // Fetch resources
  useEffect(() => {
      const loadResources = () => {
        fetch(`${API_BASE}/api/resources`)
            .then(res => res.json())
            .then(data => {
                console.log("Loaded resources:", data);
                setResources(data);
                if (data.indices && data.indices.length > 0) {
                    // If current config index is not in list, pick first available
                    if (!runConfig.index || !data.indices.includes(runConfig.index)) {
                         // Prefer cms_standard if available
                         const defaultIdx = data.indices.find(i => i.includes('cms_standard')) || data.indices[0];
                         setRunConfig(prev => ({ ...prev, index: defaultIdx }));
                    }
                }
                if (data.workloads && data.workloads.length > 0) {
                    setRunConfig(prev => ({ ...prev, workload: data.workloads[0] }));
                }
            })
            .catch(err => console.error("Failed to fetch resources", err));
      };
      
      loadResources();
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
      return allEvents.filter(e => 
          e.event_type === 'stdout' || 
          e.event_type === 'stderr' ||
          e.event_type === 'search_step' ||
          e.event_type === 'node_evaluation' ||
          e.event_type === 'evidence_collected' ||
          e.event_type === 'result' ||
          e.event_type === 'query_start' ||
          e.event_type === 'query_end'
      );
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
                const safeStep = Math.min(currentStep, activeEvents.length - 1);
                processGraph(activeEvents, safeStep);
            } else if (showFullGraph && fullGraphData) {
                processGraph([], -1);
            } else {
                setGraphData({ nodes: [], links: [] });
                setMetrics({ cost: 0, calls: 0, tokens: 0, shortcuts: 0 });
                setEvidence([]);
            }
        } catch (e) {
            console.error("Error processing graph:", e);
        } finally {
            processingRef.current = false;
        }
    }, 10);
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

  // Keyboard Shortcuts
  useEffect(() => {
      const handleKeyDown = (e) => {
          // Cmd+Enter to Run
          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
              if (!status.running && runConfig.query.trim()) {
                  runQuery();
              }
          }
          // Esc to close panels
          if (e.key === 'Escape') {
              if (isSettingsOpen) setIsSettingsOpen(false);
              if (selectedNode) setSelectedNode(null);
              if (showLogs) setShowLogs(false);
          }
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
  }, [status.running, runConfig.query, isSettingsOpen, selectedNode, showLogs]);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      console.error("Status fetch failed", e);
    }
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
          entry_points: runConfig.entry_points,
          max_steps: runConfig.max_steps,
          edge_type: runConfig.edge_type
      };
      
      if (runConfig.inputType === 'query') {
          payload.query = runConfig.query;
      } else {
          payload.workload_file = runConfig.workload;
      }

      // Sync dev mode with debug trace
      payload.debug_trace = devMode;

      logger.info("Starting run", payload);

      const res = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      
      setAllEvents([]);
      setCurrentStep(0);
      setIsPlaying(true);
      
      logger.info(`Run started. ID: ${data.run_id}`);
      const socket = new WebSocket(`${WS_BASE}/ws/${data.run_id}`);
      
      let messageBuffer = [];
      let batchTimeout = null;

      const processBatch = () => {
          if (messageBuffer.length > 0) {
              const batch = [...messageBuffer];
              messageBuffer = [];
              
              setAllEvents(prev => {
                  const next = [...prev, ...batch];
                  setCurrentStep(next.length - 1);
                  return next;
              });
          }
          batchTimeout = null;
      };

      socket.onopen = () => {
          logger.info("WebSocket connected");
      };

      socket.onmessage = (event) => {
          try {
              const newEvent = JSON.parse(event.data);
              messageBuffer.push(newEvent);
              
              if (!batchTimeout) {
                  batchTimeout = setTimeout(processBatch, 100);
              }
          } catch (e) {
              logger.error("WS Parse Error", e);
              showToast("Error parsing live update", "error");
          }
      };

      socket.onclose = () => {
          logger.info("WebSocket closed");
          if (batchTimeout) {
              clearTimeout(batchTimeout);
              processBatch();
          }
      };

      setWs(socket);
      setCurrentRunId(data.run_id);
      
      fetchStatus();
      fetchHistory(); // Update history list
    } catch (e) {
      logger.error("Run failed", e);
      showToast("Run failed: " + e.message, "error");
    }
  };

  const stopQuery = async () => {
    try {
      if (currentRunId) {
          await fetch(`${API_BASE}/api/stop/${currentRunId}`, { method: 'POST' });
      } else {
          await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
      }
      fetchStatus();
      showToast("Query stopped");
    } catch (e) {
      console.error("Stop failed", e);
      showToast("Failed to stop query", "error");
    }
  };

  const processGraph = (events, limitIndex) => {
    const nodes = new Map();
    const links = [];
    const linkSet = new Set();
    const parentMap = new Map();
    let currentMetrics = { cost: 0, calls: 0, tokens: 0, shortcuts: 0 };
    let currentEvidence = [];
    const evidenceSet = new Set();
    let currentQueue = [];
    let currentAnswer = null;
    let lastAction = "Ready.";
    let lastExploredNodeId = null;
    
    const activeNodeIds = new Set();
    const nodeStates = new Map();
    const rootEdges = [];

    if (showFullGraph && fullGraphData) {
        fullGraphData.nodes.forEach(n => {
            nodes.set(n.id, { ...n, visited: false, type: 'static', score: 0 });
        });
        fullGraphData.links.forEach(l => {
            if (nodes.has(l.source) && nodes.has(l.target)) {
                links.push({ ...l, isOnPath: false });
                linkSet.add(`${l.source}-${l.target}`);
            }
        });
    }

    for (let i = 0; i <= limitIndex && i < events.length; i++) {
        const e = events[i];
        
        if (e.event_type) lastAction = `${e.event_type}`;

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
            
            if (!parent_id) {
                parent_id = "ROOT";
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
                    if (summary) state._traceSummary = summary; 
                    if (!state.visited && !state.isEvidence) state.type = 'candidate';
                    nodeStates.set(id, state);

                    if (parent_id) {
                        parentMap.set(id, parent_id);
                    }

                    if (!state.visited && !currentQueue.find(q => q.id === id)) {
                        currentQueue.push({ id, score: state.score, summary: summary || "" });
                    } else if (!state.visited) {
                        const qItem = currentQueue.find(q => q.id === id);
                        if (qItem) {
                            qItem.score = state.score;
                            qItem.summary = summary || qItem.summary;
                        }
                    }
                });
                
                currentQueue.sort((a, b) => (b.score || 0) - (a.score || 0));
            }
        }

        if (e.event_type === 'result') {
            const { answer, path } = e.data;
            currentAnswer = answer;
            
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

    if (rootEdges.length > 0) {
        nodes.set("ROOT", { id: "ROOT", type: 'entry', summary: "Query", visited: true });
    }

    activeNodeIds.forEach(id => {
        if (id === "ROOT") return;

        let nodeData = { id, type: 'candidate', summary: "", visited: false, score: 0 };
        
        if (fullGraphData && fullGraphData.nodes) {
            const graphNode = fullGraphData.nodes.find(n => n.id === id);
            if (graphNode) {
                nodeData = { ...graphNode, ...nodeData, summary: graphNode.summary, type: 'candidate' }; 
            }
        }

        const state = nodeStates.get(id);
        if (state) {
            if (state._traceSummary && !nodeData.summary) {
                nodeData.summary = state._traceSummary;
            }
            delete state._traceSummary;
            Object.assign(nodeData, state);
        }
        
        nodes.set(id, nodeData);
    });

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

    rootEdges.forEach(targetId => {
        if (nodes.has(targetId)) {
             const key = `ROOT-${targetId}`;
             if (!linkSet.has(key)) {
                 links.push({ source: "ROOT", target: targetId, type: 'entry' });
                 linkSet.add(key);
             }
        }
    });

    parentMap.forEach((parentId, childId) => {
        if (parentId === "ROOT") return;
        
        if (nodes.has(parentId) && nodes.has(childId)) {
             const key = `${parentId}-${childId}`;
             const revKey = `${childId}-${parentId}`;
             
             if (!linkSet.has(key) && !linkSet.has(revKey)) {
                 links.push({ source: parentId, target: childId, type: 'trace' });
                 linkSet.add(key);
             }
        }
    });

    if (currentAnswer && events.length > 0) {
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

    if (lastExploredNodeId && !currentAnswer) {
        let curr = lastExploredNodeId;
        const visitedPath = new Set();
        
        while (curr && parentMap.has(curr)) {
            if (visitedPath.has(curr)) {
                break;
            }
            visitedPath.add(curr);
            
            const parent = parentMap.get(curr);
            
            if (nodes.has(curr)) nodes.get(curr).isActivePath = true;
            if (nodes.has(parent)) nodes.get(parent).isActivePath = true;

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
        setTooltip(prev => prev ? ({ ...prev, x: event.pageX, y: event.pageY }) : null);
    }
  };

  const handleNewChat = () => {
      setAllEvents([]);
      setCurrentStep(0);
      setGraphData({ nodes: [], links: [] });
      setMetrics({ cost: 0, calls: 0, tokens: 0, shortcuts: 0 });
      setEvidence([]);
      setQueue([]);
      setFinalAnswer(null);
      setSelectedQueryId(null);
      setRunConfig(prev => ({ ...prev, query: "" }));
      setMode('live');
  };

  const handleExport = () => {
      if (allEvents.length === 0) return;
      
      const blob = new Blob([allEvents.map(e => JSON.stringify(e)).join('\n')], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trace-${currentRunId || 'export'}.jsonl`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast("Trace exported successfully");
  };

  const suggestedQueries = [
      "What are the main side effects of Aspirin?",
      "Find papers about machine learning in healthcare.",
      "Who authored the key papers on Transformers?",
      "Explain the relationship between diabetes and heart disease."
  ];

  const hasRun = allEvents.length > 0 || showFullGraph;

  return (
    <div className="h-screen bg-black text-white font-sans flex overflow-hidden">
      <HistorySidebar 
        queries={queries}
        selectedQueryId={selectedQueryId}
        setSelectedQueryId={setSelectedQueryId}
        onNewChat={handleNewChat}
        onOpenSettings={() => setIsSettingsOpen(true)}
        runHistory={runHistory}
        loadRun={loadRun}
      />

      <div className="flex-1 flex flex-col relative">
        {/* Main Content Area */}
        {!hasRun ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8">
                <div className="mb-8 text-center">
                    <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">GraphRAG Explorer</h1>
                    <p className="text-gray-400">Explore your knowledge graph with natural language</p>
                </div>
                <ChatInput 
                    runConfig={runConfig}
                    setRunConfig={setRunConfig}
                    runQuery={runQuery}
                    stopQuery={stopQuery}
                    isRunning={status.running}
                    onOpenSettings={() => setIsSettingsOpen(true)}
                    mode="centered"
                />
                
                <div className="mt-8 flex flex-wrap justify-center gap-2 max-w-2xl">
                    {suggestedQueries.map((q, i) => (
                        <button
                            key={i}
                            onClick={() => {
                                setRunConfig(prev => ({ ...prev, query: q }));
                                // Optional: Auto-run
                                // setTimeout(runQuery, 100); 
                            }}
                            className="px-3 py-1.5 bg-gray-900 border border-gray-800 rounded-full text-xs text-gray-400 hover:text-white hover:border-gray-600 transition-colors"
                        >
                            {q}
                        </button>
                    ))}
                </div>
            </div>
        ) : (
            <div className="flex-1 flex flex-col h-full">
                {/* Top Bar with Input */}
                <div className="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur z-10">
                    <ChatInput 
                        runConfig={runConfig}
                        setRunConfig={setRunConfig}
                        runQuery={runQuery}
                        stopQuery={stopQuery}
                        isRunning={status.running}
                        onOpenSettings={() => setIsSettingsOpen(true)}
                        mode="bottom"
                    />
                </div>

                {/* Graph Area */}
                <div className="flex-1 relative overflow-hidden bg-gray-950">
                    <ReasoningFeed events={activeEvents} currentStep={currentStep} />
                    <ErrorBoundary>
                        <GraphVisualization 
                            graphData={graphData}
                            onNodeClick={setSelectedNode}
                            onNodeHover={handleNodeHover}
                        />
                    </ErrorBoundary>

                    {/* Overlay Stats */}
                    <div className="absolute bottom-20 right-16 bg-gray-900/80 p-3 rounded border border-gray-700 backdrop-blur pointer-events-none">
                        <div className="text-[10px] text-gray-400 uppercase tracking-wider">Graph Size</div>
                        <div className="text-lg font-bold">{graphData.nodes.length} <span className="text-sm font-normal text-gray-500">nodes</span></div>
                    </div>

                    {/* Legend */}
                    <div className="absolute bottom-24 left-4 bg-gray-900/90 p-3 rounded border border-gray-700 backdrop-blur pointer-events-none text-xs">
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

                    {/* Final Answer Overlay */}
                    {finalAnswer && (
                        <div className="absolute bottom-20 left-4 right-4 md:left-10 md:right-10 bg-gray-900/95 border border-blue-500/30 p-6 rounded-xl backdrop-blur shadow-2xl max-h-[40vh] overflow-y-auto">
                            <div className="text-xs font-bold text-blue-400 mb-2 uppercase tracking-wider">Final Answer</div>
                            <div className="text-sm text-gray-200 leading-relaxed prose prose-invert prose-sm max-w-none">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {finalAnswer}
                                </ReactMarkdown>
                            </div>
                        </div>
                    )}

                    {/* Playback Controls */}
                    <div className="absolute bottom-0 left-0 right-0">
                        <ControlPanel 
                            isPlaying={isPlaying}
                            setIsPlaying={setIsPlaying}
                            currentStep={currentStep}
                            setCurrentStep={setCurrentStep}
                            maxStep={Math.max(0, activeEvents.length - 1)}
                            activeEventsCount={activeEvents.length}
                        />
                    </div>
                </div>
            </div>
        )}
      </div>

      {/* Right Sidebar (Details) */}
      {hasRun && (
          <RunDetails 
            metrics={metrics}
            evidence={evidence}
            queue={queue}
            currentAction={currentAction}
            finalAnswer={finalAnswer}
            devMode={devMode}
            onShowLogs={() => setShowLogs(true)}
            onExport={handleExport}
            onSelectNode={(node) => {
                // Find full node data if possible
                const fullNode = graphData.nodes.find(n => n.id === node.id) || node;
                setSelectedNode(fullNode);
                // Also trigger graph zoom if we can access the ref (complex, so maybe just select for now)
            }}
          />
      )}

      {/* Modals & Overlays */}
      <SettingsModal 
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        runConfig={runConfig}
        setRunConfig={setRunConfig}
        resources={resources}
        availableModels={availableModels}
        devMode={devMode}
        setDevMode={setDevMode}
      />

      {tooltip && (
        <div 
            className="fixed z-50 bg-gray-900 border border-gray-600 p-2 rounded shadow-lg text-xs pointer-events-none"
            style={{ 
                left: Math.min(tooltip.x + 10, window.innerWidth - 220), 
                top: Math.min(tooltip.y + 10, window.innerHeight - 100) 
            }}
        >
            <div className="font-bold text-blue-400 mb-1">{tooltip.content.id.substring(0, 12)}...</div>
            {tooltip.content.score !== undefined && <div>Score: <span className="font-mono">{typeof tooltip.content.score === 'number' ? tooltip.content.score.toFixed(3) : 'N/A'}</span></div>}
            {tooltip.content.type && <div>Type: {tooltip.content.type}</div>}
            {tooltip.content.summary && <div className="mt-1 text-gray-400 max-w-xs">{tooltip.content.summary}</div>}
        </div>
      )}

      {selectedNode && (
          <NodeDetails node={selectedNode} onClose={() => setSelectedNode(null)} devMode={devMode} />
      )}
      
      {showLogs && <LogViewer logs={logs} onClose={() => setShowLogs(false)} />}
      
      {toast && (
          <Toast 
              message={toast.message} 
              type={toast.type} 
              onClose={() => setToast(null)} 
          />
      )}
    </div>
  );
}

export default App;
