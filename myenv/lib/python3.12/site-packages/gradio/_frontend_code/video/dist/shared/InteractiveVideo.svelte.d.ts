import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: FileData | null;
        subtitle?: FileData | null;
        sources?: (["webcam"] | ["upload"] | ["webcam", "upload"] | ["upload", "webcam"]) | undefined;
        label?: string | undefined;
        show_download_button?: boolean | undefined;
        show_label?: boolean | undefined;
        mirror_webcam?: boolean | undefined;
        include_audio: boolean;
        autoplay: boolean;
        root: string;
        i18n: I18nFormatter;
        active_source?: ("webcam" | "upload") | undefined;
        handle_reset_value?: (() => void) | undefined;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        loop: boolean;
        uploading?: boolean | undefined;
        webcam_constraints?: ({
            [key: string]: any;
        } | null) | undefined;
    };
    events: {
        error: CustomEvent<any>;
        start_recording: CustomEvent<any>;
        stop_recording: CustomEvent<any>;
        play: CustomEvent<any>;
        pause: CustomEvent<any>;
        stop: CustomEvent<undefined>;
        end: CustomEvent<any>;
        change: CustomEvent<any>;
        clear?: CustomEvent<undefined> | undefined;
        drag: CustomEvent<boolean>;
        upload: CustomEvent<FileData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type InteractiveVideoProps = typeof __propDef.props;
export type InteractiveVideoEvents = typeof __propDef.events;
export type InteractiveVideoSlots = typeof __propDef.slots;
export default class InteractiveVideo extends SvelteComponent<InteractiveVideoProps, InteractiveVideoEvents, InteractiveVideoSlots> {
}
export {};
