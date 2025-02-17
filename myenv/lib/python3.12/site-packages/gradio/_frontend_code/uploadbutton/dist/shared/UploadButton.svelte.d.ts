import { SvelteComponent } from "svelte";
import { type FileData, type Client } from "@gradio/client";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        label: string | null;
        value: null | FileData | FileData[];
        file_count: string;
        file_types?: string[] | undefined;
        root: string;
        size?: ("sm" | "md" | "lg") | undefined;
        icon?: FileData | null;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        variant?: ("primary" | "secondary" | "stop") | undefined;
        disabled?: boolean | undefined;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
    };
    events: {
        click: CustomEvent<any>;
        error: CustomEvent<any>;
        change: CustomEvent<any>;
        upload: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type UploadButtonProps = typeof __propDef.props;
export type UploadButtonEvents = typeof __propDef.events;
export type UploadButtonSlots = typeof __propDef.slots;
export default class UploadButton extends SvelteComponent<UploadButtonProps, UploadButtonEvents, UploadButtonSlots> {
}
export {};
