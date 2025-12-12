# SCT UI Test Plan

## Objective
To verify the functionality, reliability, and user experience of the SCT Professional Dashboard frontend in isolation. This plan focuses on end-to-end (E2E) testing using Playwright with fully mocked backend dependencies to ensure tests are deterministic and fast.

## Testing Strategy
- **Framework**: Playwright
- **Environment**: Local development (`npm run dev`) or built assets.
- **Data Strategy**: 
  - **API Mocking**: All HTTP requests to `http://localhost:8001` will be intercepted and fulfilled with static JSON fixtures.
  - **WebSocket Mocking**: The WebSocket connection used for live event streaming will be mocked at the window object level to simulate server-sent events without a running backend.

## Test Scope

### 1. Initial Load & Layout
**Goal**: Ensure the application renders the core shell and default state correctly.
- **Verify**:
  - Header is visible with "SCT Professional Dashboard" title.
  - Status indicator shows "IDLE".
  - Sidebar is present with default configuration loaded (e.g., "hcg_medical.json" index).
  - Main graph area is visible.
  - Control panel (timeline) is visible.

### 2. Configuration & Inputs
**Goal**: Ensure user inputs in the sidebar update the application state.
- **Verify**:
  - Switching between "Single Query" and "Workload File" modes toggles the appropriate input fields (Textarea vs Select).
  - Dropdowns for Index and Models are populated from the mocked `/api/resources` endpoint.

### 3. Live Run Simulation (Core Flow)
**Goal**: Verify the "Run" button triggers the correct sequence of actions and updates the UI based on streaming data.
- **Steps**:
  1. Click "Run".
  2. Intercept `/api/run` POST request.
  3. Establish (Mock) WebSocket connection.
  4. Simulate incoming events: `query_start`, `entry_points`, `node_visit`, `llm_score_request`.
- **Verify**:
  - Graph updates: New nodes appear in the SVG area.
  - Metrics update: "Calls", "Cost", etc., increment based on events.
  - Queue update: Items appear in the Priority Queue list.
  - Status changes (if applicable).

### 4. Graph Interaction
**Goal**: Ensure the visualization is interactive.
- **Verify**:
  - Nodes are clickable.
  - Clicking a node opens the **Node Details** side panel.
  - Node Details panel displays correct metadata (ID, Score, Summary).
  - Closing the panel works.

### 5. Auxiliary Panels
**Goal**: Verify secondary UI features.
- **History**:
  - Clicking "History" opens the modal.
  - Mocked history items are displayed.
- **Logs**:
  - Clicking "Logs" opens the bottom panel.
  - `stdout`/`stderr` events from the WebSocket appear in the log viewer.

## Mock Data Requirements

### API Endpoints
- `GET /api/status`: Returns `{ running: false }`.
- `GET /api/resources`: Returns lists of indices and workloads.
- `GET /api/runs`: Returns a list of past run summaries.
- `POST /api/run`: Returns a `run_id`.
- `GET /api/graph`: Returns static graph nodes/links (optional).

### WebSocket Events
- `query_start`: Metadata about the query.
- `entry_points`: Initial nodes added to the graph.
- `node_visit`: Updates node status to "visited".
- `llm_score_request`: Updates metrics.
- `stdout`: System logs.

## Execution
Run tests using the standard Playwright runner:
```bash
cd ui
npx playwright test
```
