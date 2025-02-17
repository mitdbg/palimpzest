export declare class CrossOriginWorkerMaker {
    readonly worker: Worker | SharedWorker;
    constructor(url: URL, options?: WorkerOptions & {
        shared?: boolean;
    });
}
