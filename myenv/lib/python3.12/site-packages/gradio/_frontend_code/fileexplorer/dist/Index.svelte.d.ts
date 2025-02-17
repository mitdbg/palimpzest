import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
import type { FileNode } from "./shared/types";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: string[][];
        label: string;
        show_label: boolean;
        height: number | string | undefined;
        min_height: number | string | undefined;
        max_height: number | string | undefined;
        file_count?: ("single" | "multiple") | undefined;
        root_dir: string;
        glob: string;
        ignore_glob: string;
        loading_status: LoadingStatus;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        gradio: Gradio<{
            change: never;
            clear_status: LoadingStatus;
        }>;
        server: {
            ls: (path: string[]) => Promise<FileNode[]>;
        };
        interactive: boolean;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get value(): string[][];
    /**accessor*/
    set value(_: string[][]);
    get label(): string;
    /**accessor*/
    set label(_: string);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get height(): string | number | undefined;
    /**accessor*/
    set height(_: string | number | undefined);
    get min_height(): string | number | undefined;
    /**accessor*/
    set min_height(_: string | number | undefined);
    get max_height(): string | number | undefined;
    /**accessor*/
    set max_height(_: string | number | undefined);
    get file_count(): "single" | "multiple" | undefined;
    /**accessor*/
    set file_count(_: "single" | "multiple" | undefined);
    get root_dir(): string;
    /**accessor*/
    set root_dir(_: string);
    get glob(): string;
    /**accessor*/
    set glob(_: string);
    get ignore_glob(): string;
    /**accessor*/
    set ignore_glob(_: string);
    get loading_status(): LoadingStatus;
    /**accessor*/
    set loading_status(_: LoadingStatus);
    get container(): boolean | undefined;
    /**accessor*/
    set container(_: boolean | undefined);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get gradio(): Gradio<{
        change: never;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        clear_status: LoadingStatus;
    }>);
    get server(): {
        ls: (path: string[]) => Promise<FileNode[]>;
    };
    /**accessor*/
    set server(_: {
        ls: (path: string[]) => Promise<FileNode[]>;
    });
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
}
export {};
