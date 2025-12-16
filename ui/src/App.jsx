import React, { useState, useEffect, useRef, useMemo } from 'react';
import GraphVisualization from './components/GraphVisualization';
import GraphVisualization3D from './components/GraphVisualization3D';
import ControlPanel from './components/ControlPanel';
import LogViewer from './components/LogViewer';
import NodeDetails from './components/NodeDetails';
import ErrorBoundary from './components/ErrorBoundary';
import HistorySidebar from './components/HistorySidebar';
import ChatInput from './components/ChatInput';
import RunDetails from './components/RunDetails';
import SettingsModal from './components/SettingsModal';
// ReasoningFeed overlay removed to avoid UI overlap; reasoning is shown in RunDetails.
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
  const [metrics, setMetrics] = useState({ cost: 0, calls: 0, tokens: 0, shortcuts: 0, filtered: 0 });
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
          edge_type: "",
          filters: []
      };
  });

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [devMode, setDevMode] = useState(() => {
      return localStorage.getItem('graphrag_devMode') === 'true';
  });

  const defaultVizConfig = {
      use3D: false,           // Toggle 3D mode (GPU + worker physics)
      runLayout: true,
      maxEdges: 10000,
      showLabels: true,
      hoverHighlight: true,
      labelPolicy: 'important',

      // Force Layout Config (tuned for dense graphs)
      chargeStrength: -800,      // Stronger repulsion for better spread
    // Use null to mean "infinite" (d3 default). JSON can't represent Infinity.
    chargeDistanceMax: null,
      linkDistance: 60,          // Shorter links to keep structure
      linkStrength: 0.3,         // Weaker links to allow more movement
      centerStrength: 0.02,      // Gentle centering
      collisionRadius: 2,        // Prevent overlap
      d3VelocityDecay: 0.3,      // Slightly faster settling

      // Node styling (global defaults)
      nodeRadiusScale: 1.0,
      nodeBaseSize: 3,
      nodeOpacity: 1.0,
      nodeBorder: 'none',

      // Edge styling (global defaults)
      edgeWidth: 0.7,
      edgeOpacity: 0.08,
      dimOpacity: 0.02,
      pathOpacity: 0.9,
      arrowLength: 0,
      edgeCurvature: 0,
      edgeStyle: 'solid',

      // Labels
      labelFontSize: 10,
      labelMaxLength: 20,

      // Per-type node settings: { [type]: { color, size, opacity, visible } }
      nodeTypeSettings: {},
      
      // Per-type edge settings: { [type]: { color, width, opacity, style, visible } }
      edgeTypeSettings: {},

      // State colors (for path highlighting etc.)
      stateColors: {
          evidence: '#f59e0b',
          onPath: '#10b981',
          activePath: '#06b6d4',
          visited: '#3b82f6',
          entry: '#ec4899',
      },
  };

  const [vizConfig, setVizConfig] = useState(() => {
      const saved = localStorage.getItem('graphrag_vizConfig');
      if (saved) {
          const parsed = JSON.parse(saved);
          // Always start with Canvas mode - GPU mode is opt-in each session
          return { ...parsed, use3D: false };
      }
      return defaultVizConfig;
  });

  const resetVizConfig = () => {
      setVizConfig(defaultVizConfig);
      showToast("Visualization settings reset", "success");
  };

  const [vizRefreshNonce, setVizRefreshNonce] = useState(0);

  const handleRefreshViz = () => {
      setVizRefreshNonce((n) => n + 1);
      showToast("Visualization refreshed", "success");
  };

  const handleToggleSim = () => {
      setVizConfig(prev => {
          const nextRunLayout = !prev.runLayout;
          showToast(nextRunLayout ? "Simulation running" : "Simulation paused", "success");
          return { ...prev, runLayout: nextRunLayout };
      });
  };

  // Persist settings
  useEffect(() => {
      localStorage.setItem('graphrag_runConfig', JSON.stringify(runConfig));
  }, [runConfig]);

  useEffect(() => {
      localStorage.setItem('graphrag_devMode', devMode);
  }, [devMode]);

  useEffect(() => {
      localStorage.setItem('graphrag_vizConfig', JSON.stringify(vizConfig));
  }, [vizConfig]);

  const [toast, setToast] = useState(null); // { message, type }

  const showToast = (message, type = 'success') => {
      setToast({ message, type });
  };

  // Full Graph Visualization
  const [showFullGraph, setShowFullGraph] = useState(false);
  const [fullGraphData, setFullGraphData] = useState(null);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);

  const isExploreOnly = showFullGraph && allEvents.length === 0;

  const fullGraphNodeById = useMemo(() => {
      const map = new Map();
      if (fullGraphData && Array.isArray(fullGraphData.nodes)) {
          fullGraphData.nodes.forEach(n => {
              if (n && n.id) map.set(n.id, n);
          });
      }
      return map;
  }, [fullGraphData]);

  const fullGraphAdjacency = useMemo(() => {
      const adj = new Map();
      const links = fullGraphData?.links;
      if (!Array.isArray(links)) return adj;
      links.forEach(l => {
          const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
          const targetId = typeof l.target === 'object' ? l.target.id : l.target;
          if (!sourceId || !targetId) return;
          if (!adj.has(sourceId)) adj.set(sourceId, new Set());
          if (!adj.has(targetId)) adj.set(targetId, new Set());
          adj.get(sourceId).add(targetId);
          adj.get(targetId).add(sourceId);
      });
      return adj;
  }, [fullGraphData]);

  const selectNodeById = (id) => {
      if (!id) return;
      const full = fullGraphNodeById.get(id);
      const dynamic = graphData?.nodes?.find(n => n.id === id);
      // Merge dynamic state on top of static state so we get prompt/raw_output/isThinking
      const merged = {
          ...(full || {}),
          ...(dynamic || {}),
          id 
      };
      setSelectedNode(merged);
  };

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

  const handleExploreGraph = () => {
      // Enter exploration mode without running a query.
      if (ws) ws.close();
      setAllEvents([]);
      setCurrentStep(0);
      setIsPlaying(false);
      setFinalAnswer(null);
      setSelectedNode(null);
      setMode('live');
      setShowFullGraph(true);
  };

  const handleExitExplore = () => {
      setShowFullGraph(false);
      setSelectedNode(null);
      // Leave run state untouched otherwise; exiting explore returns to landing if there is no trace.
  };

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

  const deleteRun = async (runId) => {
      try {
          const res = await fetch(`${API_BASE}/api/runs/${runId}`, { method: 'DELETE' });
          const data = await res.json();
          if (data.ok) {
              setRunHistory(prev => prev.filter(r => r.run_id !== runId));
              showToast("Run deleted");
          } else {
              showToast("Failed to delete run", "error");
          }
      } catch (e) {
          console.error("Delete run failed", e);
          showToast("Failed to delete run", "error");
      }
  };

  const clearHistory = async () => {
      try {
          const res = await fetch(`${API_BASE}/api/runs`, { method: 'DELETE' });
          const data = await res.json();
          if (data.ok) {
              setRunHistory([]);
              showToast(`Cleared ${data.deleted} runs`);
          } else {
              showToast("Failed to clear history", "error");
          }
      } catch (e) {
          console.error("Clear history failed", e);
          showToast("Failed to clear history", "error");
      }
  };

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

  // Playback should reflect meaningful graph updates, not every low-level trace event.
  const playbackEvents = useMemo(() => {
      return activeEvents.filter(e =>
          e.event_type === 'frontier_update' ||
          e.event_type === 'search_step' ||
          e.event_type === 'node_evaluation' ||
          e.event_type === 'evidence_collected' ||
          e.event_type === 'result' ||
          e.event_type === 'answer_update' ||
          e.event_type === 'query_start' ||
          e.event_type === 'query_end' ||
          e.event_type === 'error' ||
          e.event_type === 'traverse_trace'
      );
  }, [activeEvents]);

  const reasoningItems = useMemo(() => {
      const upto = Math.min(currentStep + 1, playbackEvents.length);
      return playbackEvents
          .slice(0, upto)
          .filter(e => e.event_type === 'search_step' || e.event_type === 'node_evaluation' || e.event_type === 'evidence_collected')
          .slice(-5);
  }, [playbackEvents, currentStep]);

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
                    if (prev >= playbackEvents.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
    }, [isPlaying, playbackEvents.length, playbackSpeed]);

  // Re-process graph when step changes
  const processingRef = useRef(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
        if (processingRef.current) return;
        processingRef.current = true;
        
        try {
            if (playbackEvents.length > 0) {
                const safeStep = Math.min(currentStep, playbackEvents.length - 1);
                processGraph(playbackEvents, safeStep);
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
  }, [currentStep, playbackEvents, showFullGraph, fullGraphData]);

  // Keep currentStep in range for the current playback event list.
  useEffect(() => {
      if (playbackEvents.length === 0) return;
      const maxStep = playbackEvents.length - 1;
      if (currentStep > maxStep) setCurrentStep(maxStep);
  }, [playbackEvents.length, currentStep]);

  // Reset step when query changes (unless live)
  useEffect(() => {
      if (mode !== 'live') {
          setCurrentStep(0);
      }
  }, [selectedQueryId, mode]);

  // Auto-scroll in live mode
  useEffect(() => {
      if (mode === 'live' && playbackEvents.length > 0) {
          setCurrentStep(playbackEvents.length - 1);
      }
  }, [playbackEvents.length, mode, selectedQueryId]);

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
          edge_type: runConfig.edge_type,
          filters: runConfig.filters,
          expand_filtered_nodes: runConfig.expand_filtered_nodes,
          admittance_instructions: runConfig.admittance_instructions
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
          // Refresh history after run completes (WS closes when run is done)
          fetchHistory();
          fetchStatus();
          
          // Show completion status
          setIsPlaying(false);
      };

      setWs(socket);
      setCurrentRunId(data.run_id);
      
      fetchStatus();
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
    let currentMetrics = { cost: 0, calls: 0, tokens: 0, shortcuts: 0, filtered: 0 };
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
            // Preserve the dataset's node type in `pz_type` so the viz/type editor
            // can classify nodes correctly even when we use `type` for traversal state.
            const pzType = n.pz_type || n.type;
            nodes.set(n.id, { ...n, pz_type: pzType, visited: false, type: 'static', score: 0 });
        });
        fullGraphData.links.forEach(l => {
            // Normalize src/dst to source/target (data may use either convention)
            const sourceId = l.source || l.src;
            const targetId = l.target || l.dst;
            if (nodes.has(sourceId) && nodes.has(targetId)) {
                links.push({ ...l, source: sourceId, target: targetId, isOnPath: false });
                linkSet.add(`${sourceId}-${targetId}`);
            }
        });
    }

    for (let i = 0; i <= limitIndex && i < events.length; i++) {
        let e = events[i];

        // Unwrap traverse_trace events
        if (e.event_type === 'traverse_trace' && e.data?.event_type) {
            e = { event_type: e.data.event_type, data: e.data };
        }
        
        if (e.event_type) lastAction = `${e.event_type}`;

        if (e.event_type === 'step_summary') {
            if (e.data?.skipped && e.data?.skip_reason === 'visit_filter_rejected') {
                currentMetrics.filtered += 1;
                const { node_id } = e.data;
                activeNodeIds.add(node_id);
                const state = nodeStates.get(node_id) || {};
                state.isFiltered = true;
                state.skipReason = "Skipped by Filter";
                nodeStates.set(node_id, state);
            }
        }

        if (e.event_type === 'step_gate_admittance') {
            if (e.data?.cost_usd) currentMetrics.cost += e.data.cost_usd;
            if (e.data?.tokens?.input) currentMetrics.tokens += e.data.tokens.input;
            if (e.data?.tokens?.output) currentMetrics.tokens += e.data.tokens.output;
            if (!e.data?.cache_hit) currentMetrics.calls += 1;
            else currentMetrics.shortcuts += 1;

            const { node_id, decision } = e.data;
            if (node_id) {
                activeNodeIds.add(node_id);
                const state = nodeStates.get(node_id) || {};
                state.isThinking = false; // Clear thinking state
                if (e.data.prompt) state.prompt = e.data.prompt;
                if (e.data.raw_output) state.raw_output = e.data.raw_output;
                
                // Capture admittance decision
                if (decision) {
                    state.admit = decision.passed;
                    state.reason = decision.reason;
                    state.model = decision.model;
                    
                    if (decision.passed) {
                        state.isEvidence = true;
                        state.type = 'evidence';
                        if (!evidenceSet.has(node_id)) {
                            currentEvidence.push({
                                node_id,
                                score: 1.0, // Admitted nodes are implicitly relevant
                                reasoning: decision.reason,
                                summary: state.summary || ""
                            });
                            evidenceSet.add(node_id);
                        }
                    }
                }
                
                nodeStates.set(node_id, state);
            }
        }

        if (e.event_type === 'step_node_loaded') {
            const { node } = e.data;
            if (node && node.id) {
                activeNodeIds.add(node.id);
                const state = nodeStates.get(node.id) || {};
                state.label = node.label;
                state.pz_type = node.type;
                state.summary = node.summary || node.text_preview;
                // Don't overwrite 'type' (viz state) if it's already set to 'visited'/'evidence'
                if (!state.type) state.type = 'candidate';
                nodeStates.set(node.id, state);
            }
        }

        if (e.event_type === 'step_expand') {
            const { node_id, neighbors } = e.data;
            if (node_id && neighbors) {
                activeNodeIds.add(node_id);
                neighbors.forEach(nb => {
                    const targetId = nb.neighbor_id;
                    activeNodeIds.add(targetId);
                    
                    const source = nb.direction === 'in' ? targetId : node_id;
                    const target = nb.direction === 'in' ? node_id : targetId;
                    
                    const key = `${source}-${target}`;
                    if (!linkSet.has(key)) {
                        links.push({ 
                            source, 
                            target, 
                            type: nb.edge_type,
                            id: nb.edge_id 
                        });
                        linkSet.add(key);
                    }
                });
            }
        }

        if (e.event_type === 'step_begin') {
            const { node_id } = e.data;
            if (node_id) {
                activeNodeIds.add(node_id);
                const state = nodeStates.get(node_id) || {};
                state.isThinking = true;
                nodeStates.set(node_id, state);
            }
        }

        if (e.event_type === 'step_end' || e.event_type === 'step_summary') {
             const { node_id } = e.data;
             if (node_id) {
                 const state = nodeStates.get(node_id) || {};
                 state.isThinking = false;
                 nodeStates.set(node_id, state);
             }
        }

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
            if (e.data?.summary) state._traceSummary = e.data.summary;
            if (e.data?.decision) state.decision = e.data.decision;
            if (e.data?.admit !== undefined) state.admit = e.data.admit;
            if (e.data?.reason) state.reason = e.data.reason;
            if (e.data?.model) state.model = e.data.model;
            if (e.data?.raw_output) state.raw_output = e.data.raw_output;
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

        // Handle streaming answer updates
        if (e.event_type === 'answer_update') {
            if (e.data?.answer) {
                currentAnswer = e.data.answer;
            }
        }

        // Handle error events
        if (e.event_type === 'error') {
            lastAction = `Error: ${e.data?.error || 'Unknown error'}`;
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
                // Keep the graph's semantic type in `pz_type`.
                const pzType = graphNode.pz_type || graphNode.type;
                nodeData = { ...graphNode, ...nodeData, pz_type: pzType, summary: graphNode.summary, type: 'candidate' }; 
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
                // Only add if not already added by step_expand
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
            content: {
                id: node.id,
                score: node.score,
                type: node.type,
                summary: node.summary,
                decision: node.decision,
                admit: node.admit,
                reason: node.reason,
                model: node.model,
                isFiltered: node.isFiltered,
                skipReason: node.skipReason
            }
        });
    } else if (event) {
        setTooltip(prev => prev ? ({ ...prev, x: event.pageX, y: event.pageY }) : null);
    }
  };

  const handleNewChat = () => {
      setAllEvents([]);
      setCurrentStep(0);
      setGraphData({ nodes: [], links: [] });
      setMetrics({ cost: 0, calls: 0, tokens: 0, shortcuts: 0, filtered: 0 });
      setEvidence([]);
      setQueue([]);
      setFinalAnswer(null);
      setSelectedQueryId(null);
      setRunConfig(prev => ({ ...prev, query: "" }));
      setMode('live');
      setShowFullGraph(false);
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

  const hasRun = allEvents.length > 0 || showFullGraph || status.running;

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
        onDeleteRun={deleteRun}
        onClearHistory={clearHistory}
        onExploreGraph={handleExploreGraph}
        isExploring={isExploreOnly}
        isRunning={status.running}
        currentQuery={runConfig.query}
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

                {/* Always-visible graph selection + exploration */}
                <div className="mt-4 w-full max-w-2xl mx-auto flex flex-col gap-2">
                    <div className="flex items-center justify-between gap-3 bg-gray-900 border border-gray-800 rounded-lg p-3">
                        <div className="flex flex-col gap-1 min-w-0">
                            <div className="text-xs text-gray-500 uppercase tracking-wider">Graph</div>
                            <select
                                className="w-full bg-gray-800 border border-gray-700 rounded p-2 text-xs text-white focus:ring-2 focus:ring-blue-500 outline-none"
                                value={runConfig.index}
                                onChange={(e) => setRunConfig(prev => ({ ...prev, index: e.target.value }))}
                            >
                                {resources.indices.length === 0 && <option value="">Loading graphs...</option>}
                                {resources.indices.map(idx => (
                                    <option key={idx} value={idx}>{idx}</option>
                                ))}
                            </select>
                            <div className="text-[11px] text-gray-500 truncate">
                                {isLoadingGraph
                                    ? 'Loading graph…'
                                    : (fullGraphData
                                        ? `Loaded: ${fullGraphData.nodes?.length || 0} nodes, ${fullGraphData.links?.length || 0} edges`
                                        : 'No graph loaded')}
                            </div>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                            <button
                                onClick={handleExploreGraph}
                                disabled={!runConfig.index || isLoadingGraph}
                                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Explore graph
                            </button>
                            <button
                                onClick={() => {
                                    // If switching to GPU mode, check WebGL first
                                    if (!vizConfig?.use3D) {
                                        const canvas = document.createElement('canvas');
                                        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                                        if (!gl) {
                                            showToast('WebGL not available. Close ALL browser windows and reopen, or try a different browser.', 'error');
                                            return;
                                        }
                                        // Clean up test context
                                        const ext = gl.getExtension('WEBGL_lose_context');
                                        if (ext) ext.loseContext();
                                    }
                                    setVizConfig(prev => ({ ...prev, use3D: !prev.use3D }));
                                }}
                                className={`px-3 py-2 border rounded text-xs hover:bg-gray-700 ${
                                    vizConfig?.use3D 
                                        ? 'bg-cyan-900 border-cyan-600 text-cyan-200' 
                                        : 'bg-gray-800 border-gray-700 text-gray-200'
                                }`}
                            >
                                {vizConfig?.use3D ? 'GPU 🚀' : 'Canvas'}
                            </button>
                            <button
                                onClick={handleToggleSim}
                                disabled={!fullGraphData || isLoadingGraph}
                                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {vizConfig?.runLayout ? 'Pause sim' : 'Run sim'}
                            </button>
                            <button
                                onClick={handleRefreshViz}
                                disabled={!fullGraphData || isLoadingGraph}
                                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Refresh viz
                            </button>
                            <button
                                onClick={() => setIsSettingsOpen(true)}
                                className="px-3 py-2 bg-blue-600 rounded text-xs text-white hover:bg-blue-500"
                            >
                                Settings
                            </button>
                        </div>
                    </div>
                </div>
                
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
                    {/* Always-visible graph selection */}
                    <div className="w-full max-w-4xl mx-auto flex items-center gap-3 mb-3">
                        <div className="flex-1 min-w-0">
                            <div className="text-[11px] text-gray-500 uppercase tracking-wider">Graph</div>
                            <div className="flex items-center gap-2">
                                <select
                                    className="flex-1 min-w-0 bg-gray-800 border border-gray-700 rounded p-2 text-xs text-white focus:ring-2 focus:ring-blue-500 outline-none"
                                    value={runConfig.index}
                                    onChange={(e) => setRunConfig(prev => ({ ...prev, index: e.target.value }))}
                                >
                                    {resources.indices.length === 0 && <option value="">Loading graphs...</option>}
                                    {resources.indices.map(idx => (
                                        <option key={idx} value={idx}>{idx}</option>
                                    ))}
                                </select>
                                {isExploreOnly ? (
                                    <button
                                        onClick={handleExitExplore}
                                        className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700"
                                    >
                                        Exit explore
                                    </button>
                                ) : (
                                    <button
                                        onClick={handleExploreGraph}
                                        disabled={isLoadingGraph}
                                        className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Explore
                                    </button>
                                )}
                                <button
                                    onClick={handleToggleSim}
                                    disabled={!fullGraphData || isLoadingGraph}
                                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {vizConfig?.runLayout ? 'Pause sim' : 'Run sim'}
                                </button>
                                <button
                                    onClick={handleRefreshViz}
                                    disabled={!fullGraphData || isLoadingGraph}
                                    className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Refresh viz
                                </button>
                                <button
                                    onClick={() => setIsSettingsOpen(true)}
                                    className="px-3 py-2 bg-blue-600 rounded text-xs text-white hover:bg-blue-500"
                                >
                                    Settings
                                </button>
                            </div>
                            <div className="text-[11px] text-gray-500 truncate mt-1">
                                {isLoadingGraph
                                    ? 'Loading graph…'
                                    : (fullGraphData
                                        ? `Loaded: ${fullGraphData.nodes?.length || 0} nodes, ${fullGraphData.links?.length || 0} edges`
                                        : 'No graph loaded')}
                            </div>
                        </div>
                    </div>
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
                    <ErrorBoundary>
                        {vizConfig?.use3D ? (
                            <GraphVisualization3D 
                                data={graphData}
                                onNodeClick={setSelectedNode}
                                onNodeHover={handleNodeHover}
                                vizConfig={vizConfig}
                                graphId={currentRunId}
                            />
                        ) : (
                            <GraphVisualization 
                                graphData={graphData}
                                onNodeClick={setSelectedNode}
                                onNodeHover={handleNodeHover}
                                vizConfig={vizConfig}
                                refreshNonce={vizRefreshNonce}
                            />
                        )}
                    </ErrorBoundary>

                    {/* Overlay Stats */}
                    <div className="absolute bottom-20 right-16 bg-gray-900/80 p-3 rounded border border-gray-700 backdrop-blur pointer-events-none">
                        <div className="text-[10px] text-gray-400 uppercase tracking-wider">Graph Size</div>
                        <div className="text-lg font-bold">{graphData.nodes.length} <span className="text-sm font-normal text-gray-500">nodes</span></div>
                    </div>



                    {/* Final Answer Overlay */}
                    {!isExploreOnly && finalAnswer && (
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
                    {!isExploreOnly && (
                        <div className="absolute bottom-0 left-0 right-0">
                            <ControlPanel 
                                isPlaying={isPlaying}
                                setIsPlaying={setIsPlaying}
                                currentStep={currentStep}
                                setCurrentStep={setCurrentStep}
                                maxStep={Math.max(0, playbackEvents.length - 1)}
                                activeEventsCount={playbackEvents.length}
                                metrics={metrics}
                            />
                        </div>
                    )}
                </div>
            </div>
        )}
      </div>

      {/* Right Sidebar (Details) */}
            {hasRun && !isExploreOnly && (
          <RunDetails 
            metrics={metrics}
            evidence={evidence}
            queue={queue}
                        reasoning={reasoningItems}
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
        vizConfig={vizConfig}
        setVizConfig={setVizConfig}
        onResetVizConfig={resetVizConfig}
        graphData={fullGraphData || graphData}
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
            {(tooltip.content.decision || tooltip.content.admit !== undefined) && (
                <div>
                    Decision: <span className="font-mono">{String(tooltip.content.decision || (tooltip.content.admit ? 'admit' : 'reject'))}</span>
                </div>
            )}
            {tooltip.content.model && <div>Model: <span className="font-mono">{tooltip.content.model}</span></div>}
            {tooltip.content.summary && <div className="mt-1 text-gray-400 max-w-xs">{tooltip.content.summary}</div>}
            {tooltip.content.reason && <div className="mt-1 text-gray-300 max-w-xs">{tooltip.content.reason}</div>}
        </div>
      )}

      {selectedNode && (
                    <NodeDetails
                        node={graphData.nodes.find(n => n.id === selectedNode.id) || selectedNode}
                        onClose={() => setSelectedNode(null)}
                        devMode={devMode}
                        neighbors={Array.from(fullGraphAdjacency.get(selectedNode.id) || []).sort()}
                        onSelectNeighbor={selectNodeById}
                    />
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
