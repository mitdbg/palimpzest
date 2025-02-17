import { SvelteComponent } from "svelte";
export { default as FilePreview } from "./shared/FilePreview.svelte";
export { default as BaseFileUpload } from "./shared/FileUpload.svelte";
export { default as BaseFile } from "./shared/File.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { FileData } from "@gradio/client";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: null | FileData | FileData[];
        interactive: boolean;
        root: string;
        label: string;
        show_label: boolean;
        height?: number | undefined;
        _selectable?: boolean | undefined;
        loading_status: LoadingStatus;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        gradio: Gradio<{
            change: never;
            error: string;
            upload: never;
            clear: never;
            select: SelectData;
            clear_status: LoadingStatus;
            delete: FileData;
            download: FileData;
        }>;
        file_count: "single" | "multiple" | "directory";
        file_types?: string[] | undefined;
        input_ready: boolean;
        allow_reordering?: boolean | undefined;
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
    get value(): any;
    /**accessor*/
    set value(_: any);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get label(): string;
    /**accessor*/
    set label(_: string);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get height(): number | undefined;
    /**accessor*/
    set height(_: number | undefined);
    get _selectable(): boolean | undefined;
    /**accessor*/
    set _selectable(_: boolean | undefined);
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
        error: string;
        upload: never;
        clear: never;
        select: SelectData;
        clear_status: LoadingStatus;
        delete: FileData;
        download: FileData;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        error: string;
        upload: never;
        clear: never;
        select: SelectData;
        clear_status: LoadingStatus;
        delete: FileData;
        download: FileData;
    }>);
    get file_count(): "single" | "multiple" | "directory";
    /**accessor*/
    set file_count(_: "single" | "multiple" | "directory");
    get file_types(): string[] | undefined;
    /**accessor*/
    set file_types(_: string[] | undefined);
    get input_ready(): boolean;
    /**accessor*/
    set input_ready(_: boolean);
    get allow_reordering(): boolean | undefined;
    /**accessor*/
    set allow_reordering(_: boolean | undefined);
}
