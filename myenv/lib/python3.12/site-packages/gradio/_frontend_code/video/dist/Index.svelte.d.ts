import { SvelteComponent } from "svelte";
import type { Gradio, ShareData } from "@gradio/utils";
import type { FileData } from "@gradio/client";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: ({
            video: FileData;
            subtitles: FileData | null;
        } | null) | undefined;
        label: string;
        sources: ["webcam"] | ["upload"] | ["webcam", "upload"] | ["upload", "webcam"];
        root: string;
        show_label: boolean;
        loading_status: LoadingStatus;
        height: number | undefined;
        width: number | undefined;
        webcam_constraints?: ({
            [key: string]: any;
        } | null) | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        autoplay?: boolean | undefined;
        show_share_button?: boolean | undefined;
        show_download_button: boolean;
        gradio: Gradio<{
            change: never;
            clear: never;
            play: never;
            pause: never;
            upload: never;
            stop: never;
            end: never;
            start_recording: never;
            stop_recording: never;
            share: ShareData;
            error: string;
            warning: string;
            clear_status: LoadingStatus;
        }>;
        interactive: boolean;
        mirror_webcam: boolean;
        include_audio: boolean;
        loop?: boolean | undefined;
        input_ready: boolean;
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
    get value(): {
        video: FileData;
        subtitles: FileData | null;
    } | null | undefined;
    /**accessor*/
    set value(_: {
        video: FileData;
        subtitles: FileData | null;
    } | null | undefined);
    get label(): string;
    /**accessor*/
    set label(_: string);
    get sources(): ["upload"] | ["webcam"] | ["webcam", "upload"] | ["upload", "webcam"];
    /**accessor*/
    set sources(_: ["upload"] | ["webcam"] | ["webcam", "upload"] | ["upload", "webcam"]);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get loading_status(): LoadingStatus;
    /**accessor*/
    set loading_status(_: LoadingStatus);
    get height(): number | undefined;
    /**accessor*/
    set height(_: number | undefined);
    get width(): number | undefined;
    /**accessor*/
    set width(_: number | undefined);
    get webcam_constraints(): {
        [key: string]: any;
    } | null | undefined;
    /**accessor*/
    set webcam_constraints(_: {
        [key: string]: any;
    } | null | undefined);
    get container(): boolean | undefined;
    /**accessor*/
    set container(_: boolean | undefined);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get autoplay(): boolean | undefined;
    /**accessor*/
    set autoplay(_: boolean | undefined);
    get show_share_button(): boolean | undefined;
    /**accessor*/
    set show_share_button(_: boolean | undefined);
    get show_download_button(): boolean;
    /**accessor*/
    set show_download_button(_: boolean);
    get gradio(): Gradio<{
        change: never;
        clear: never;
        play: never;
        pause: never;
        upload: never;
        stop: never;
        end: never;
        start_recording: never;
        stop_recording: never;
        share: ShareData;
        error: string;
        warning: string;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        clear: never;
        play: never;
        pause: never;
        upload: never;
        stop: never;
        end: never;
        start_recording: never;
        stop_recording: never;
        share: ShareData;
        error: string;
        warning: string;
        clear_status: LoadingStatus;
    }>);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get mirror_webcam(): boolean;
    /**accessor*/
    set mirror_webcam(_: boolean);
    get include_audio(): boolean;
    /**accessor*/
    set include_audio(_: boolean);
    get loop(): boolean | undefined;
    /**accessor*/
    set loop(_: boolean | undefined);
    get input_ready(): boolean;
    /**accessor*/
    set input_ready(_: boolean);
}
export {};
