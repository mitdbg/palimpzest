export const mockEventsFull = [
    {
        "event_type": "query_start",
        "query_id": "q1",
        "query_text": "What is the capital of France?",
        "depth": 0,
        "data": { "query": "What is the capital of France?" }
    },
    {
        "event_type": "entry_points",
        "query_id": "q1",
        "depth": 0,
        "data": {
            "entries": [
                { "node_id": "node_paris", "score": 0.9, "summary": "Paris is the capital of France." },
                { "node_id": "node_lyon", "score": 0.5, "summary": "Lyon is a city in France." }
            ]
        }
    },
    {
        "event_type": "queue_update",
        "query_id": "q1",
        "depth": 0,
        "data": {
            "queue": [
                { "id": "node_paris", "score": 0.9, "summary": "Paris is the capital of France." },
                { "id": "node_lyon", "score": 0.5, "summary": "Lyon is a city in France." }
            ]
        }
    },
    {
        "event_type": "node_visit",
        "query_id": "q1",
        "depth": 0,
        "data": { "node_id": "node_paris", "score": 0.9 }
    },
    {
        "event_type": "expansion",
        "query_id": "q1",
        "depth": 1,
        "data": {
            "from_node": { "id": "node_paris", "summary": "Paris..." },
            "children": [
                { "node_id": "node_eiffel", "summary": "The Eiffel Tower is in Paris." }
            ]
        }
    },
    {
        "event_type": "evidence_found",
        "query_id": "q1",
        "depth": 1,
        "data": {
            "node_id": "node_paris",
            "score": 1.0,
            "summary": "Paris is the capital of France."
        }
    },
    {
        "event_type": "answer_found",
        "query_id": "q1",
        "depth": 1,
        "data": {
            "answer": "The capital of France is Paris.",
            "path": ["node_paris"]
        }
    },
    {
        "event_type": "query_end",
        "query_id": "q1",
        "depth": 0,
        "data": {
            "answer": "The capital of France is Paris.",
            "path": ["node_paris"]
        }
    }
];
