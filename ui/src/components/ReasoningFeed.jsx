import React, { useEffect, useRef } from 'react';
import { Brain, CheckCircle, ArrowRight } from 'lucide-react';

const ReasoningFeed = ({ events, currentStep }) => {
    const scrollRef = useRef(null);

    // Filter for relevant events to show in the feed
    const feedItems = events.slice(0, currentStep + 1).filter(e => 
        e.event_type === 'search_step' || 
        e.event_type === 'node_evaluation' ||
        e.event_type === 'evidence_collected'
    ).slice(-5); // Show last 5

    return (
        <div className="absolute top-4 left-4 z-10 w-80 pointer-events-none">
            <div className="space-y-2">
                {feedItems.map((item, idx) => {
                    const isLast = idx === feedItems.length - 1;
                    const opacity = isLast ? 1 : 0.6;
                    
                    let content = null;
                    if (item.event_type === 'search_step') {
                        content = (
                            <div className="flex items-start gap-2 bg-gray-900/90 backdrop-blur border border-blue-500/30 p-3 rounded-lg shadow-lg text-xs animate-in fade-in slide-in-from-left-4 duration-300">
                                <ArrowRight size={14} className="text-blue-400 mt-0.5 shrink-0" />
                                <div>
                                    <div className="font-bold text-blue-200">Exploring</div>
                                    <div className="text-gray-300 font-mono">{item.data.node_id.substring(0, 12)}...</div>
                                </div>
                            </div>
                        );
                    } else if (item.event_type === 'node_evaluation') {
                        content = (
                            <div className="flex items-start gap-2 bg-gray-900/90 backdrop-blur border border-purple-500/30 p-3 rounded-lg shadow-lg text-xs animate-in fade-in slide-in-from-left-4 duration-300">
                                <Brain size={14} className="text-purple-400 mt-0.5 shrink-0" />
                                <div>
                                    <div className="font-bold text-purple-200">Evaluating</div>
                                    <div className="text-gray-300 line-clamp-2">{item.data.reasoning || "Analyzing relevance..."}</div>
                                </div>
                            </div>
                        );
                    } else if (item.event_type === 'evidence_collected') {
                        content = (
                            <div className="flex items-start gap-2 bg-gray-900/90 backdrop-blur border border-yellow-500/30 p-3 rounded-lg shadow-lg text-xs animate-in fade-in slide-in-from-left-4 duration-300">
                                <CheckCircle size={14} className="text-yellow-400 mt-0.5 shrink-0" />
                                <div>
                                    <div className="font-bold text-yellow-200">Evidence Found</div>
                                    <div className="text-gray-300 line-clamp-2">{item.data.content}</div>
                                </div>
                            </div>
                        );
                    }

                    return (
                        <div key={item.seq || idx} style={{ opacity }}>
                            {content}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default ReasoningFeed;
