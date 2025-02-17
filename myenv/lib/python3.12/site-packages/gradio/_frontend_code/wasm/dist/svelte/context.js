import { setContext, getContext } from "svelte";
const WORKER_PROXY_CONTEXT_KEY = "WORKER_PROXY_CONTEXT_KEY";
export function setWorkerProxyContext(workerProxy) {
    setContext(WORKER_PROXY_CONTEXT_KEY, workerProxy);
}
export function getWorkerProxyContext() {
    return getContext(WORKER_PROXY_CONTEXT_KEY);
}
