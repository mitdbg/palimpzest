import type { PyodideInterface } from "pyodide";
export declare const globalHomeDir = "/home/pyodide";
export declare const getAppHomeDir: (appId: string) => string;
export declare const resolveAppHomeBasedPath: (appId: string, filePath: string) => string;
export declare function writeFileWithParents(pyodide: PyodideInterface, filePath: string, data: string | ArrayBufferView, opts?: Parameters<PyodideInterface["FS"]["writeFile"]>[2]): void;
export declare function renameWithParents(pyodide: PyodideInterface, oldPath: string, newPath: string): void;
