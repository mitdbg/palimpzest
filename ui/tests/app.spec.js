import { test, expect } from '@playwright/test';

test.describe('SCT Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Mock the API routes
    await page.route('**/api/status', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ running: false, last_query: "" })
      });
    });

    await page.route('**/api/resources', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          indices: ['data/hcg_medical.json', 'data/wiki.json'],
          workloads: ['data/workload.jsonl']
        })
      });
    });

    await page.route('**/api/run', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ run_id: 'test_run_123' })
      });
    });

    await page.route('**/api/runs', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          history: [
            { run_id: 'run1', timestamp: '2023-01-01T12:00:00', query: 'Test Query 1' },
            { run_id: 'run2', timestamp: '2023-01-01T13:00:00', query: 'Test Query 2' }
          ]
        })
      });
    });

    // Inject Mock WebSocket
    await page.addInitScript(() => {
      class MockWebSocket {
        constructor(url) {
          this.url = url;
          this.readyState = 0; // CONNECTING
          setTimeout(() => {
            this.readyState = 1; // OPEN
            if (this.onopen) this.onopen();
          }, 100);
          window.mockWebSocketInstance = this;
        }

        send(data) {
          console.log('WS Send:', data);
        }

        close() {
          this.readyState = 3; // CLOSED
          if (this.onclose) this.onclose();
        }
      }
      window.WebSocket = MockWebSocket;
    });

    await page.goto('http://localhost:5173');
  });

  test('loads initial layout correctly', async ({ page }) => {
    await expect(page).toHaveTitle(/SCT Professional Dashboard/);
    await expect(page.getByText('SCT Professional Dashboard')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Run' })).toBeVisible();
    await expect(page.getByRole('combobox').first()).toBeVisible(); // Index selector
  });

  test('can switch input modes', async ({ page }) => {
    await page.getByText('Workload File').click();
    await expect(page.getByPlaceholder('Enter your query...')).not.toBeVisible();
    
    await page.getByText('Single Query').click();
    await expect(page.getByPlaceholder('Enter your query...')).toBeVisible();
  });

  test('runs a query and updates graph', async ({ page }) => {
    // 1. Start Query
    await page.getByPlaceholder('Enter your query...').fill('Test Query');
    await page.getByRole('button', { name: 'Run' }).click();

    // Wait for WS connection
    await page.waitForFunction(() => window.mockWebSocketInstance && window.mockWebSocketInstance.readyState === 1);

    // Simulate WS events
    await page.evaluate(() => {
        // 1. Send query_start
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'query_start',
                query_id: 'q1',
                query_text: 'Test Query'
            })
        });
        
        // 2. Send entry_points
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'entry_points',
                query_id: 'q1',
                data: {
                    entries: [
                        { node_id: 'node1', score: 0.9, summary: 'Detailed Summary Info', type: 'entry' }
                    ]
                }
            })
        });

        // 3. Send node_visit
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'node_visit',
                query_id: 'q1',
                data: {
                    node_id: 'node1',
                    score: 0.9
                }
            })
        });
    });

    // Check if graph updated
    // Note: D3 nodes are rendered as <g class="node">
    await expect(page.locator('g.node text').filter({ hasText: 'Detailed' })).toBeVisible({ timeout: 5000 });
  });

  test('shows node details on click', async ({ page }) => {
     // Setup graph with one node
     await page.getByPlaceholder('Enter your query...').fill('Test Query');
     await page.getByRole('button', { name: 'Run' }).click();
     await page.waitForFunction(() => window.mockWebSocketInstance && window.mockWebSocketInstance.readyState === 1);
 
     await page.evaluate(() => {
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'query_start',
                query_id: 'q1',
                query_text: 'Test Query'
            })
        });
         window.mockWebSocketInstance.onmessage({
             data: JSON.stringify({
                 event_type: 'entry_points',
                 query_id: 'q1',
                 data: {
                     entries: [
                         { node_id: 'node1', score: 0.9, summary: 'Detailed Summary Info', type: 'entry' }
                     ]
                 }
             })
         });
     });
     
     // Click the node
     const node = page.locator('g.node text').filter({ hasText: 'Detailed' });
     await node.waitFor();
     await node.click({ force: true });
     
     // Check details panel
     await expect(page.getByText('Detailed Summary Info').first()).toBeVisible();
     await expect(page.getByText('Score: 0.9').first()).toBeVisible();
  });

  test('history modal works', async ({ page }) => {
      await page.getByRole('button', { name: 'History' }).click();
      
      await expect(page.getByText('Run History')).toBeVisible({ timeout: 5000 });
      await expect(page.getByText('Test Query 1')).toBeVisible();
      await expect(page.getByText('Test Query 2')).toBeVisible();
      
      // Close modal using the close button (X icon)
      // The X icon is usually a button inside the modal
      await page.locator('button').filter({ has: page.locator('svg.lucide-x') }).click();
      await expect(page.getByText('Run History')).not.toBeVisible();
  });

  test('logs panel toggles', async ({ page }) => {
      // Start a run to initialize WS
      await page.getByPlaceholder('Enter your query...').fill('Test Query');
      await page.getByRole('button', { name: 'Run' }).click();
      await page.waitForFunction(() => window.mockWebSocketInstance && window.mockWebSocketInstance.readyState === 1);

      await page.getByRole('button', { name: 'Logs' }).click();
      await expect(page.getByText('System Logs')).toBeVisible();
      
      // Simulate log event
      await page.evaluate(() => {
          window.mockWebSocketInstance.onmessage({
              data: JSON.stringify({
                  event_type: 'stdout',
                  timestamp: 1700000000,
                  data: { message: 'Test Log Message' }
              })
          });
      });

      await expect(page.getByText('Test Log Message')).toBeVisible();
      
      await page.getByRole('button', { name: 'Logs' }).click(); // Toggle off
      await expect(page.getByText('System Logs')).not.toBeVisible();
  });

  test('sidebar configuration options work', async ({ page }) => {
    // Index Selection (1st select)
    const indexSelect = page.locator('select').nth(0);
    await expect(indexSelect).toBeVisible();
    await expect(indexSelect).toHaveValue('data/hcg_medical.json');
    await indexSelect.selectOption('data/wiki.json');
    await expect(indexSelect).toHaveValue('data/wiki.json');

    // Orchestrator Model (2nd select)
    const modelSelect = page.locator('select').nth(1);
    await expect(modelSelect).toBeVisible();
    // Check if 'gpt-4o' is an option
    await modelSelect.selectOption('gpt-4o');
    await expect(modelSelect).toHaveValue('gpt-4o');

    // Advanced Config
    await expect(page.getByText('Advanced Model Config')).toBeVisible();
    
    // Ranking Model (3rd select)
    const rankingSelect = page.locator('select').nth(2);
    await rankingSelect.selectOption('cross-encoder/ms-marco-MiniLM-L-6-v2');
    await expect(rankingSelect).toHaveValue('cross-encoder/ms-marco-MiniLM-L-6-v2');

    // Entry Points (Input type number)
    const entryPointsInput = page.locator('input[type="number"]');
    await expect(entryPointsInput).toHaveValue('5');
    await entryPointsInput.fill('10');
    await expect(entryPointsInput).toHaveValue('10');
  });

  test('graph toggles work', async ({ page }) => {
    const fullGraphCheckbox = page.getByLabel('Show Full Graph');
    await expect(fullGraphCheckbox).not.toBeChecked();
    await fullGraphCheckbox.check();
    await expect(fullGraphCheckbox).toBeChecked();

    const crossRefCheckbox = page.getByLabel('Cross-Reference Nodes');
    await expect(crossRefCheckbox).not.toBeChecked();
    await crossRefCheckbox.check();
    await expect(crossRefCheckbox).toBeChecked();
  });

  test('metrics and evidence update correctly', async ({ page }) => {
    // Start a run
    await page.getByPlaceholder('Enter your query...').fill('Test Query');
    await page.getByRole('button', { name: 'Run' }).click();
    await page.waitForFunction(() => window.mockWebSocketInstance && window.mockWebSocketInstance.readyState === 1);

    // Initial metrics
    await expect(page.getByText('$0.0000')).toBeVisible();
    await expect(page.getByText('0.0k')).toBeVisible(); // Tokens

    // Simulate events
    await page.evaluate(() => {
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'llm_score_request',
                data: {}
            })
        });
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'evidence_found',
                data: {
                    node_id: 'ev1',
                    score: 0.95,
                    summary: 'Evidence Summary'
                }
            })
        });
    });

    // Check metrics update (Calls should be 1)
    // Find the "Calls" label and check the value below it
    const callsContainer = page.locator('div.bg-gray-800', { has: page.getByText('Calls') });
    await expect(callsContainer.getByText('1')).toBeVisible();

    // Check Evidence List
    await expect(page.getByText('Evidence (1)')).toBeVisible();
    await expect(page.getByText('ev1')).toBeVisible();
    await expect(page.getByText('Evidence Summary')).toBeVisible();
  });

  test('priority queue updates', async ({ page }) => {
    // Start a run
    await page.getByPlaceholder('Enter your query...').fill('Test Query');
    await page.getByRole('button', { name: 'Run' }).click();
    await page.waitForFunction(() => window.mockWebSocketInstance && window.mockWebSocketInstance.readyState === 1);

    await expect(page.getByText('Empty')).toBeVisible();

    // Simulate queue update
    await page.evaluate(() => {
        window.mockWebSocketInstance.onmessage({
            data: JSON.stringify({
                event_type: 'queue_update',
                data: {
                    queue: [
                        { id: 'nodeA', score: 0.8, summary: 'Node A Summary' },
                        { id: 'nodeB', score: 0.6, summary: 'Node B Summary' }
                    ]
                }
            })
        });
    });

    await expect(page.getByText('Empty')).not.toBeVisible();
    await expect(page.getByText('Node A Summary', { exact: true })).toBeVisible();
    await expect(page.getByText('0.80')).toBeVisible();
    await expect(page.getByText('Node B Summary', { exact: true })).toBeVisible();
    await expect(page.getByText('0.60')).toBeVisible();
  });
});
