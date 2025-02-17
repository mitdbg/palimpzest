import type { PyodideInterface } from "pyodide";
export declare function verifyRequirements(requirements: string[]): void;
export declare function patchRequirements(pyodide: PyodideInterface, requirements: string[]): string[];
