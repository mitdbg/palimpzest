import React, { useState, useMemo, useEffect } from 'react';
import { Search, X } from 'lucide-react';

const GraphSearch = ({ nodes, onSelectNode }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState("");

    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                setIsOpen(true);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const filteredNodes = useMemo(() => {
        if (!query || !nodes) return [];
        const lowerQ = query.toLowerCase();
        return nodes.filter(n => 
            (n.id && n.id.toLowerCase().includes(lowerQ)) || 
            (n.summary && n.summary.toLowerCase().includes(lowerQ))
        ).slice(0, 10); // Limit results
    }, [query, nodes]);

    if (!isOpen) {
        return (
            <button 
                onClick={() => setIsOpen(true)}
                className="p-2 bg-gray-800 text-gray-300 rounded hover:bg-gray-700 hover:text-white shadow-lg border border-gray-700"
                title="Search Nodes (Cmd+K)"
            >
                <Search size={16} />
            </button>
        );
    }

    return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl w-64 flex flex-col animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center p-2 border-b border-gray-700 gap-2">
                <Search size={14} className="text-gray-500" />
                <input 
                    autoFocus
                    type="text"
                    placeholder="Find node..."
                    className="bg-transparent border-none outline-none text-xs text-white flex-1"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Escape') setIsOpen(false);
                    }}
                />
                <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-white">
                    <X size={14} />
                </button>
            </div>
            
            {query && (
                <div className="max-h-48 overflow-y-auto py-1">
                    {filteredNodes.length === 0 ? (
                        <div className="px-3 py-2 text-xs text-gray-500 italic">No matches found</div>
                    ) : (
                        filteredNodes.map(node => (
                            <button
                                key={node.id}
                                onClick={() => {
                                    onSelectNode(node);
                                    setIsOpen(false);
                                    setQuery("");
                                }}
                                className="w-full text-left px-3 py-2 text-xs text-gray-300 hover:bg-gray-800 hover:text-white truncate"
                            >
                                <div className="font-bold">{node.id}</div>
                                <div className="text-gray-500 truncate">{node.summary}</div>
                            </button>
                        ))
                    )}
                </div>
            )}
        </div>
    );
};

export default GraphSearch;
