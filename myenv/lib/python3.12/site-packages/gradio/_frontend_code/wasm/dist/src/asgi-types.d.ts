import type { PyProxy } from "pyodide/ffi";
export type ASGIScope = Record<string, unknown>;
export type ASGIApplication = (scope: ASGIScope, receive: () => Promise<ReceiveEvent>, send: (event: PyProxy) => Promise<void>) => Promise<void>;
export type ReceiveEvent = RequestReceiveEvent | DisconnectReceiveEvent;
export interface RequestReceiveEvent {
    type: "http.request";
    body?: Uint8Array;
    more_body?: boolean;
}
export interface DisconnectReceiveEvent {
    type: "http.disconnect";
}
export type SendEvent = ResponseStartSendEvent | ResponseBodySendEvent;
export interface ResponseStartSendEvent {
    type: "http.response.start";
    status: number;
    headers: Iterable<[Uint8Array, Uint8Array]>;
    trailers: boolean;
}
export interface ResponseBodySendEvent {
    type: "http.response.body";
    body: Uint8Array;
    more_body: boolean;
}
