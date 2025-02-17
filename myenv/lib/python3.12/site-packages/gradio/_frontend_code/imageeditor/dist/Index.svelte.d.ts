import { SvelteComponent } from "svelte";
import type { Brush, Eraser } from "./shared/tools/Brush.svelte";
import type { EditorData, ImageBlobs } from "./shared/InteractiveImageEditor.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: (EditorData | null) | undefined;
        label: string;
        show_label: boolean;
        show_download_button: boolean;
        root: string;
        value_is_output?: boolean | undefined;
        height: number | undefined;
        width: number | undefined;
        _selectable?: boolean | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        show_share_button?: boolean | undefined;
        sources?: ("clipboard" | "webcam" | "upload")[] | undefined;
        interactive: boolean;
        placeholder: string | undefined;
        brush: Brush;
        eraser: Eraser;
        crop_size?: ([number, number] | `${string}:${string}` | null) | undefined;
        transforms?: "crop"[] | undefined;
        layers?: boolean | undefined;
        attached_events?: string[] | undefined;
        server: {
            accept_blobs: (a: any) => void;
        };
        canvas_size: [number, number];
        fixed_canvas?: boolean | undefined;
        show_fullscreen_button?: boolean | undefined;
        full_history?: any;
        gradio: Gradio<{
            change: never;
            error: string;
            input: never;
            edit: never;
            drag: never;
            apply: never;
            upload: never;
            clear: never;
            select: SelectData;
            share: ShareData;
            clear_status: LoadingStatus;
        }>;
        get_value?: (() => Promise<ImageBlobs | {
            id: string;
        }>) | undefined;
    };
    events: {
        error: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
    get get_value(): () => Promise<ImageBlobs | {
        id: string;
    }>;
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get value(): EditorData | null | undefined;
    /**accessor*/
    set value(_: EditorData | null | undefined);
    get label(): string;
    /**accessor*/
    set label(_: string);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get show_download_button(): boolean;
    /**accessor*/
    set show_download_button(_: boolean);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
    get height(): number | undefined;
    /**accessor*/
    set height(_: number | undefined);
    get width(): number | undefined;
    /**accessor*/
    set width(_: number | undefined);
    get _selectable(): boolean | undefined;
    /**accessor*/
    set _selectable(_: boolean | undefined);
    get container(): boolean | undefined;
    /**accessor*/
    set container(_: boolean | undefined);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get loading_status(): LoadingStatus;
    /**accessor*/
    set loading_status(_: LoadingStatus);
    get show_share_button(): boolean | undefined;
    /**accessor*/
    set show_share_button(_: boolean | undefined);
    get sources(): ("clipboard" | "upload" | "webcam")[] | undefined;
    /**accessor*/
    set sources(_: ("clipboard" | "upload" | "webcam")[] | undefined);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get placeholder(): string | undefined;
    /**accessor*/
    set placeholder(_: string | undefined);
    get brush(): Brush;
    /**accessor*/
    set brush(_: Brush);
    get eraser(): Eraser;
    /**accessor*/
    set eraser(_: Eraser);
    get crop_size(): [number, number] | `${string}:${string}` | null | undefined;
    /**accessor*/
    set crop_size(_: [number, number] | `${string}:${string}` | null | undefined);
    get transforms(): "crop"[] | undefined;
    /**accessor*/
    set transforms(_: "crop"[] | undefined);
    get layers(): boolean | undefined;
    /**accessor*/
    set layers(_: boolean | undefined);
    get attached_events(): string[] | undefined;
    /**accessor*/
    set attached_events(_: string[] | undefined);
    get server(): {
        accept_blobs: (a: any) => void;
    };
    /**accessor*/
    set server(_: {
        accept_blobs: (a: any) => void;
    });
    get canvas_size(): [number, number];
    /**accessor*/
    set canvas_size(_: [number, number]);
    get fixed_canvas(): boolean | undefined;
    /**accessor*/
    set fixed_canvas(_: boolean | undefined);
    get show_fullscreen_button(): boolean | undefined;
    /**accessor*/
    set show_fullscreen_button(_: boolean | undefined);
    get full_history(): any;
    /**accessor*/
    set full_history(_: any);
    get gradio(): Gradio<{
        change: never;
        error: string;
        input: never;
        edit: never;
        drag: never;
        apply: never;
        upload: never;
        clear: never;
        select: SelectData;
        share: ShareData;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        error: string;
        input: never;
        edit: never;
        drag: never;
        apply: never;
        upload: never;
        clear: never;
        select: SelectData;
        share: ShareData;
        clear_status: LoadingStatus;
    }>);
    get undefined(): any;
    /**accessor*/
    set undefined(_: any);
}
export {};
