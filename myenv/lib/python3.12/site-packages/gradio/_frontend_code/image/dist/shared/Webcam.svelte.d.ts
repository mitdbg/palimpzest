import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
import { type FileData, type Client } from "@gradio/client";
import type { Base64File } from "./types";
declare const __propDef: {
    props: {
        modify_stream?: ((state: "open" | "closed" | "waiting") => void) | undefined;
        set_time_limit?: ((time: number) => void) | undefined;
        streaming?: boolean | undefined;
        pending?: boolean | undefined;
        root?: string | undefined;
        stream_every?: number | undefined;
        mode?: ("image" | "video") | undefined;
        mirror_webcam: boolean;
        include_audio: boolean;
        webcam_constraints?: ({
            [key: string]: any;
        } | null) | undefined;
        i18n: I18nFormatter;
        upload: Client["upload"];
        value?: FileData | null | Base64File;
        click_outside?: ((node: Node, cb: any) => any) | undefined;
    };
    events: {
        stream: CustomEvent<string | Blob>;
        capture: CustomEvent<any>;
        error: CustomEvent<string>;
        start_recording: CustomEvent<undefined>;
        stop_recording: CustomEvent<undefined>;
        close_stream: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type WebcamProps = typeof __propDef.props;
export type WebcamEvents = typeof __propDef.events;
export type WebcamSlots = typeof __propDef.slots;
export default class Webcam extends SvelteComponent<WebcamProps, WebcamEvents, WebcamSlots> {
    get modify_stream(): (state: "open" | "closed" | "waiting") => void;
    get set_time_limit(): (time: number) => void;
    get click_outside(): (node: Node, cb: any) => any;
}
export {};
