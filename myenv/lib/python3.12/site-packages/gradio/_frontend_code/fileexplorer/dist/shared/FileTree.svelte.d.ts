import { SvelteComponent } from "svelte";
import type { FileNode } from "./types";
declare const __propDef: {
    props: {
        path?: string[] | undefined;
        selected_files?: string[][] | undefined;
        selected_folders?: string[][] | undefined;
        is_selected_entirely?: boolean | undefined;
        interactive: boolean;
        ls_fn: (path: string[]) => Promise<FileNode[]>;
        file_count?: ("single" | "multiple") | undefined;
        valid_for_selection: boolean;
    };
    events: {
        check: CustomEvent<{
            path: string[];
            checked: boolean;
            type: "file" | "folder";
        }>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type FileTreeProps = typeof __propDef.props;
export type FileTreeEvents = typeof __propDef.events;
export type FileTreeSlots = typeof __propDef.slots;
export default class FileTree extends SvelteComponent<FileTreeProps, FileTreeEvents, FileTreeSlots> {
}
export {};
