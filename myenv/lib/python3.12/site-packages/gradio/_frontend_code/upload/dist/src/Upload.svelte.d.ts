import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import { type Client } from "@gradio/client";
declare const __propDef: {
    props: {
        filetype?: (string | string[] | null) | undefined;
        dragging?: boolean | undefined;
        boundedheight?: boolean | undefined;
        center?: boolean | undefined;
        flex?: boolean | undefined;
        file_count?: ("single" | "multiple" | "directory") | undefined;
        disable_click?: boolean | undefined;
        root: string;
        hidden?: boolean | undefined;
        format?: ("blob" | "file") | undefined;
        uploading?: boolean | undefined;
        hidden_upload?: (HTMLInputElement | null) | undefined;
        show_progress?: boolean | undefined;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        icon_upload?: boolean | undefined;
        height?: number | string | undefined;
        paste_clipboard?: (() => void) | undefined;
        open_file_upload?: (() => void) | undefined;
        load_files?: ((files: File[] | Blob[]) => Promise<(FileData | null)[] | void>) | undefined;
    };
    events: {
        drag: DragEvent;
        dragstart: DragEvent;
        dragend: DragEvent;
        dragover: DragEvent;
        dragenter: DragEvent;
        dragleave: DragEvent;
        drop: DragEvent;
        load: CustomEvent<any>;
        error: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type UploadProps = typeof __propDef.props;
export type UploadEvents = typeof __propDef.events;
export type UploadSlots = typeof __propDef.slots;
export default class Upload extends SvelteComponent<UploadProps, UploadEvents, UploadSlots> {
    get paste_clipboard(): () => void;
    get open_file_upload(): () => void;
    get load_files(): (files: File[] | Blob[]) => Promise<(FileData | null)[] | void>;
}
export {};
