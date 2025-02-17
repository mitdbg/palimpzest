import { SvelteComponent } from "svelte";
import { type I18nFormatter } from "@gradio/utils";
import { type Client } from "@gradio/client";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        background_file: FileData | null;
        root: string;
        sources?: ("upload" | "webcam" | "clipboard")[] | undefined;
        mirror_webcam?: boolean | undefined;
        i18n: I18nFormatter;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        dragging: boolean;
        canvas_size: [number, number];
        fixed_canvas?: boolean | undefined;
        active_mode?: ("webcam" | "color" | null) | undefined;
        bg?: boolean | undefined;
    };
    events: {
        error: CustomEvent<any> | CustomEvent<string>;
        drag: CustomEvent<any>;
        upload: CustomEvent<never>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type SourcesProps = typeof __propDef.props;
export type SourcesEvents = typeof __propDef.events;
export type SourcesSlots = typeof __propDef.slots;
export default class Sources extends SvelteComponent<SourcesProps, SourcesEvents, SourcesSlots> {
}
export {};
