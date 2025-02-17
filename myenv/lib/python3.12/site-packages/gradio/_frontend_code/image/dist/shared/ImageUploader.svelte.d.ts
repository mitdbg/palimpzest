import { SvelteComponent } from "svelte";
import { type SelectData, type I18nFormatter, type ValueData } from "@gradio/utils";
import { FileData, type Client } from "@gradio/client";
import type { Base64File } from "./types";
declare const __propDef: {
    props: {
        value?: null | FileData | Base64File;
        label?: string | undefined;
        show_label: boolean;
        sources?: ("clipboard" | "upload" | "microphone" | "webcam" | null)[] | undefined;
        streaming?: boolean | undefined;
        pending?: boolean | undefined;
        mirror_webcam: boolean;
        selectable?: boolean | undefined;
        root: string;
        i18n: I18nFormatter;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        stream_every: number;
        modify_stream: (state: "open" | "closed" | "waiting") => void;
        set_time_limit: (arg0: number) => void;
        show_fullscreen_button?: boolean | undefined;
        uploading?: boolean | undefined;
        active_source?: ("clipboard" | "upload" | "microphone" | "webcam" | null) | undefined;
        webcam_constraints?: {
            [key: string]: any;
        } | undefined;
        dragging?: boolean | undefined;
    };
    events: {
        error: CustomEvent<string> | CustomEvent<any>;
        drag: CustomEvent<any>;
        close_stream: CustomEvent<undefined>;
        change?: CustomEvent<undefined> | undefined;
        stream: CustomEvent<ValueData>;
        clear?: CustomEvent<undefined> | undefined;
        upload?: CustomEvent<undefined> | undefined;
        select: CustomEvent<SelectData>;
        end_stream: CustomEvent<never>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type ImageUploaderProps = typeof __propDef.props;
export type ImageUploaderEvents = typeof __propDef.events;
export type ImageUploaderSlots = typeof __propDef.slots;
export default class ImageUploader extends SvelteComponent<ImageUploaderProps, ImageUploaderEvents, ImageUploaderSlots> {
}
export {};
