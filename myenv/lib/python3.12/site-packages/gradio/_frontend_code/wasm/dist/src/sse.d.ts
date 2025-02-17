import type { SendEvent } from "./asgi-types";
import type { WorkerProxy } from "./worker-proxy";
export declare class WasmWorkerEventSource extends EventTarget {
    /**
     * 0 — connecting
     * 1 — open
     * 2 — closed
     * https://developer.mozilla.org/en-US/docs/Web/API/EventSource/readyState
     */
    readyState: number;
    port: MessagePort;
    url: URL;
    onopen: ((this: WasmWorkerEventSource, ev: Event) => any) | undefined;
    onmessage: ((this: WasmWorkerEventSource, ev: MessageEvent) => any) | undefined;
    onerror: ((this: WasmWorkerEventSource, ev: Event) => any) | undefined;
    constructor(workerProxy: WorkerProxy, url: URL);
    close(): void;
    _handleAsgiSendEvent(e: MessageEvent<SendEvent>): void;
    interpretEventStream(streamContent: string): void;
}
