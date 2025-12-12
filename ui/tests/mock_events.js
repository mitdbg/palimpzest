export const mockEvents = [
    {
        "event_type": "query_start",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": { "query": "What is basal cell skin cancer?", "metadata": {} }
    },
    {
        "event_type": "stdout",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": { "message": "Starting search..." }
    },
    {
        "event_type": "entry_points",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": {
            "count": 3,
            "entries": [
                { "node_id": "node1", "score": 0.9, "summary": "Basal cell carcinoma is a type of skin cancer." },
                { "node_id": "node2", "score": 0.8, "summary": "It begins in the basal cells." },
                { "node_id": "node3", "score": 0.7, "summary": "Sun exposure is a risk factor." }
            ]
        }
    },
    {
        "event_type": "queue_update",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": {
            "queue": [
                { "id": "node1", "score": 0.9, "summary": "Basal cell carcinoma is a type of skin cancer." },
                { "id": "node2", "score": 0.8, "summary": "It begins in the basal cells." },
                { "id": "node3", "score": 0.7, "summary": "Sun exposure is a risk factor." }
            ]
        }
    },
    {
        "event_type": "llm_score_request",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": { "llm_call_id": 1, "candidate_count": 1 }
    },
    {
        "event_type": "llm_score_response",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": {
            "llm_call_id": 1,
            "scored_candidates": [
                { "node_id": "node1", "score": 0.95, "summary": "Basal cell carcinoma is the most common form of skin cancer." }
            ]
        }
    },
    {
        "event_type": "evidence_added",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": {
            "node_id": "node1",
            "evidence": "Basal cell carcinoma is the most common form of skin cancer.",
            "score": 0.95
        }
    },
    {
        "event_type": "answer_found",
        "query_id": 1,
        "query_text": "What is basal cell skin cancer?",
        "depth": 0,
        "data": {
            "answer": "Basal cell skin cancer is a common form of skin cancer that begins in the basal cells.",
            "confidence": 0.9
        }
    }
];
