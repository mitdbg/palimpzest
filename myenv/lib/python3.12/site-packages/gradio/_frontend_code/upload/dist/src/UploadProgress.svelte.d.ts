import { SvelteComponent } from "svelte";
import { FileData, type Client } from "@gradio/client";
declare const __propDef: {
    props: {
        upload_id: string;
        root: string;
        files: FileData[];
        stream_handler: Client["stream"];
    };
    events: {
        done: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type UploadProgressProps = typeof __propDef.props;
export type UploadProgressEvents = typeof __propDef.events;
export type UploadProgressSlots = typeof __propDef.slots;
export default class UploadProgress extends SvelteComponent<UploadProgressProps, UploadProgressEvents, UploadProgressSlots> {
}
export {};
