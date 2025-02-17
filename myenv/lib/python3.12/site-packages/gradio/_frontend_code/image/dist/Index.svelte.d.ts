import { SvelteComponent } from "svelte";
export { default as Webcam } from "./shared/Webcam.svelte";
export { default as BaseImageUploader } from "./shared/ImageUploader.svelte";
export { default as BaseStaticImage } from "./shared/ImagePreview.svelte";
export { default as BaseExample } from "./Example.svelte";
export { default as BaseImage } from "./shared/Image.svelte";
import type { Gradio, SelectData, ValueData } from "@gradio/utils";
import { type FileData } from "@gradio/client";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        modify_stream_state?: ((state: "open" | "closed" | "waiting") => void) | undefined;
        get_stream_state?: (() => void) | undefined;
        set_time_limit: (arg0: number) => void;
        value_is_output?: boolean | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: null | FileData;
        label: string;
        show_label: boolean;
        show_download_button: boolean;
        root: string;
        height: number | undefined;
        width: number | undefined;
        stream_every: number;
        _selectable?: boolean | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        show_share_button?: boolean | undefined;
        sources?: ("clipboard" | "webcam" | "upload")[] | undefined;
        interactive: boolean;
        streaming: boolean;
        pending: boolean;
        mirror_webcam: boolean;
        placeholder?: string | undefined;
        show_fullscreen_button: boolean;
        input_ready: boolean;
        webcam_constraints?: {
            [key: string]: any;
        } | undefined;
        gradio: Gradio<{
            input: never;
            change: never;
            error: string;
            edit: never;
            stream: ValueData;
            drag: never;
            upload: never;
            clear: never;
            select: SelectData;
            share: ShareData;
            clear_status: LoadingStatus;
            close_stream: string;
        }>;
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
    get modify_stream_state(): (state: "open" | "closed" | "waiting") => void;
    get get_stream_state(): () => void;
    get undefined(): any;
    /**accessor*/
    set undefined(_: any);
    get set_time_limit(): (arg0: number) => void;
    /**accessor*/
    set set_time_limit(_: (arg0: number) => void);
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
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
    get height(): number | undefined;
    /**accessor*/
    set height(_: number | undefined);
    get width(): number | undefined;
    /**accessor*/
    set width(_: number | undefined);
    get stream_every(): number;
    /**accessor*/
    set stream_every(_: number);
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
    get streaming(): boolean;
    /**accessor*/
    set streaming(_: boolean);
    get pending(): boolean;
    /**accessor*/
    set pending(_: boolean);
    get mirror_webcam(): boolean;
    /**accessor*/
    set mirror_webcam(_: boolean);
    get placeholder(): string | undefined;
    /**accessor*/
    set placeholder(_: string | undefined);
    get show_fullscreen_button(): boolean;
    /**accessor*/
    set show_fullscreen_button(_: boolean);
    get input_ready(): boolean;
    /**accessor*/
    set input_ready(_: boolean);
    get webcam_constraints(): {
        [key: string]: any;
    } | undefined;
    /**accessor*/
    set webcam_constraints(_: {
        [key: string]: any;
    } | undefined);
    get gradio(): Gradio<{
        input: never;
        change: never;
        error: string;
        edit: never;
        stream: ValueData;
        drag: never;
        upload: never;
        clear: never;
        select: SelectData;
        share: ShareData;
        clear_status: LoadingStatus;
        close_stream: string;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        input: never;
        change: never;
        error: string;
        edit: never;
        stream: ValueData;
        drag: never;
        upload: never;
        clear: never;
        select: SelectData;
        share: ShareData;
        clear_status: LoadingStatus;
        close_stream: string;
    }>);
}
