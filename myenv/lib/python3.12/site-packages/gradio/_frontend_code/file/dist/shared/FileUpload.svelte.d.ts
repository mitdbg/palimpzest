import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: null | FileData | FileData[];
        label: string;
        show_label?: boolean | undefined;
        file_count?: ("single" | "multiple" | "directory") | undefined;
        file_types?: (string[] | null) | undefined;
        selectable?: boolean | undefined;
        root: string;
        height?: number | undefined;
        i18n: I18nFormatter;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        uploading?: boolean | undefined;
        allow_reordering?: boolean | undefined;
    };
    events: {
        error: CustomEvent<any>;
        select: CustomEvent<import("@gradio/utils").SelectData>;
        change: CustomEvent<any>;
        delete: CustomEvent<FileData>;
        clear: CustomEvent<undefined>;
        drag: CustomEvent<boolean>;
        upload: CustomEvent<any>;
        load: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type FileUploadProps = typeof __propDef.props;
export type FileUploadEvents = typeof __propDef.events;
export type FileUploadSlots = typeof __propDef.slots;
export default class FileUpload extends SvelteComponent<FileUploadProps, FileUploadEvents, FileUploadSlots> {
}
export {};
