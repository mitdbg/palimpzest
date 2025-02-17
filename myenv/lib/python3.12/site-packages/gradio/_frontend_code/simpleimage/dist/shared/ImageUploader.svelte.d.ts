import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
declare const __propDef: {
    props: {
        value: null | FileData;
        label?: string | undefined;
        show_label: boolean;
        root: string;
        upload: Client["upload"];
        stream_handler: Client["stream"];
    };
    events: {
        error: CustomEvent<any>;
        change?: CustomEvent<undefined> | undefined;
        clear?: CustomEvent<undefined> | undefined;
        drag: CustomEvent<boolean>;
        upload?: CustomEvent<undefined> | undefined;
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
