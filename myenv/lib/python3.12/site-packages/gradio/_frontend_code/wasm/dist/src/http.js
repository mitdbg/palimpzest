// Inspired by https://github.com/rstudio/shinylive/blob/v0.1.2/src/messageporthttp.ts
export function headersToASGI(headers) {
    const result = [];
    for (const [key, value] of Object.entries(headers)) {
        result.push([key, value]);
    }
    return result;
}
export function uint8ArrayToString(buf) {
    let result = "";
    for (let i = 0; i < buf.length; i++) {
        result += String.fromCharCode(buf[i]);
    }
    return result;
}
export function asgiHeadersToRecord(headers) {
    headers = headers.map(([key, val]) => {
        return [uint8ArrayToString(key), uint8ArrayToString(val)];
    });
    return Object.fromEntries(headers);
}
export function getHeaderValue(headers, key) {
    // The keys in `headers` are case-insensitive.
    const unifiedKey = key.toLowerCase();
    for (const [k, v] of Object.entries(headers)) {
        if (k.toLowerCase() === unifiedKey) {
            return v;
        }
    }
}
export function logHttpReqRes(request, response) {
    if (Math.floor(response.status / 100) !== 2) {
        let bodyText;
        let bodyJson;
        try {
            bodyText = new TextDecoder().decode(response.body);
        }
        catch (e) {
            bodyText = "(failed to decode body)";
        }
        try {
            bodyJson = JSON.parse(bodyText);
        }
        catch (e) {
            bodyJson = "(failed to parse body as JSON)";
        }
        console.error("Wasm HTTP error", {
            request,
            response,
            bodyText,
            bodyJson
        });
    }
}
