export declare class AwaitableQueue<T> {
    _buffer: T[];
    _promise: Promise<void>;
    _resolve: () => void;
    constructor();
    _wait(): Promise<void>;
    _notifyAll(): void;
    dequeue(): Promise<T>;
    enqueue(x: T): void;
}
