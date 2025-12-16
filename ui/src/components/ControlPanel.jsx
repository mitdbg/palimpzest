import React from 'react';
import { Play, Pause, ChevronLeft, ChevronRight } from 'lucide-react';

const ControlPanel = ({ isPlaying, setIsPlaying, currentStep, setCurrentStep, maxStep, activeEventsCount, metrics }) => {
    return (
        <div className="h-16 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-4 z-10">
            <button 
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 flex items-center justify-center bg-blue-600 rounded hover:bg-blue-500 text-white"
            >
                {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            
            <div className="flex-1 flex flex-col justify-center">
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Step {currentStep}</span>
                    <div className="flex gap-3">
                        <span>{activeEventsCount} Events</span>
                        {metrics?.filtered > 0 && (
                            <span className="text-red-400 font-medium">Filtered: {metrics.filtered}</span>
                        )}
                    </div>
                </div>
                <input 
                    type="range" 
                    min="0" 
                    max={Math.max(0, maxStep)} 
                    value={currentStep} 
                    onChange={(e) => {
                        setIsPlaying(false);
                        setCurrentStep(parseInt(e.target.value));
                    }}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            <div className="flex gap-1">
                <button onClick={() => setCurrentStep(Math.max(0, currentStep - 1))} className="p-1 hover:bg-gray-700 rounded"><ChevronLeft size={16}/></button>
                <button onClick={() => setCurrentStep(Math.min(maxStep, currentStep + 1))} className="p-1 hover:bg-gray-700 rounded"><ChevronRight size={16}/></button>
            </div>
        </div>
    );
};

export default ControlPanel;
