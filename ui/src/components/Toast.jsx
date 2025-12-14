import React, { useEffect } from 'react';
import { CheckCircle, AlertCircle, X } from 'lucide-react';

const Toast = ({ message, type = 'success', onClose, duration = 3000 }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, duration);
        return () => clearTimeout(timer);
    }, [duration, onClose]);

    const bgColors = {
        success: 'bg-green-900/90 border-green-500/50 text-green-100',
        error: 'bg-red-900/90 border-red-500/50 text-red-100',
        info: 'bg-blue-900/90 border-blue-500/50 text-blue-100'
    };

    const icons = {
        success: <CheckCircle size={16} className="text-green-400" />,
        error: <AlertCircle size={16} className="text-red-400" />,
        info: <AlertCircle size={16} className="text-blue-400" />
    };

    return (
        <div className={`fixed bottom-4 right-4 z-[60] flex items-center gap-3 px-4 py-3 rounded-lg border shadow-xl backdrop-blur animate-in slide-in-from-right duration-300 ${bgColors[type]}`}>
            {icons[type]}
            <span className="text-sm font-medium">{message}</span>
            <button onClick={onClose} className="ml-2 hover:opacity-70">
                <X size={14} />
            </button>
        </div>
    );
};

export default Toast;
