import type { EmscriptenFile, EmscriptenFileUrl, InMessage, OutMessage } from "./message-types";
import { PromiseDelegate } from "./promise-delegate";
import { type HttpRequest, type HttpResponse } from "./http";
export interface WorkerProxyOptions {
    gradioWheelUrl: string;
    gradioClientWheelUrl: string;
    files: Record<string, EmscriptenFile | EmscriptenFileUrl>;
    requirements: string[];
    sharedWorkerMode: boolean;
}
export declare class WorkerProxy extends EventTarget {
    worker: globalThis.Worker | globalThis.SharedWorker;
    postMessageTarget: globalThis.Worker | MessagePort;
    firstRunPromiseDelegate: PromiseDelegate<void>;
    constructor(options: WorkerProxyOptions);
    runPythonCode(code: string): Promise<void>;
    runPythonFile(path: string): Promise<void>;
    postMessageAsync(msg: InMessage): Promise<unknown>;
    _processWorkerMessage(msg: OutMessage): void;
    requestAsgi(scope: Record<string, unknown>): MessagePort;
    httpRequest(request: HttpRequest): Promise<HttpResponse>;
    writeFile(path: string, data: string | ArrayBufferView, opts?: Record<string, unknown>): Promise<void>;
    renameFile(oldPath: string, newPath: string): Promise<void>;
    unlink(path: string): Promise<void>;
    install(requirements: string[]): Promise<void>;
    terminate(): void;
}
