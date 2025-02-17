import { SvelteComponent } from "svelte";
import type { FileNode } from "./types";
declare const __propDef: {
    props: {
        interactive: boolean;
        file_count?: ("single" | "multiple") | undefined;
        value?: string[][] | undefined;
        ls_fn: (path: string[]) => Promise<FileNode[]>;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type DirectoryExplorerProps = typeof __propDef.props;
export type DirectoryExplorerEvents = typeof __propDef.events;
export type DirectoryExplorerSlots = typeof __propDef.slots;
export default class DirectoryExplorer extends SvelteComponent<DirectoryExplorerProps, DirectoryExplorerEvents, DirectoryExplorerSlots> {
}
export {};
