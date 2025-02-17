export interface HttpRequest {
    method: "GET" | "POST" | "PUT" | "DELETE";
    path: string;
    query_string: string;
    headers: Record<string, string>;
    body?: Uint8Array | ReadableStream<Uint8Array> | null;
}
export interface HttpResponse {
    status: number;
    headers: Record<string, string>;
    body: Uint8Array;
}
export declare function headersToASGI(headers: HttpRequest["headers"]): [string, string][];
export declare function uint8ArrayToString(buf: Uint8Array): string;
export declare function asgiHeadersToRecord(headers: any): Record<string, string>;
export declare function getHeaderValue(headers: HttpRequest["headers"], key: string): string | undefined;
export declare function logHttpReqRes(request: HttpRequest, response: HttpResponse): void;
