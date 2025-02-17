import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import type { I18nFormatter, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: FileData | FileData[];
        selectable?: boolean | undefined;
        height?: number | string | undefined;
        i18n: I18nFormatter;
        allow_reordering?: boolean | undefined;
    };
    events: {
        dragenter: DragEvent;
        select: CustomEvent<SelectData>;
        change: CustomEvent<any>;
        delete: CustomEvent<FileData>;
        download: CustomEvent<FileData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type FilePreviewProps = typeof __propDef.props;
export type FilePreviewEvents = typeof __propDef.events;
export type FilePreviewSlots = typeof __propDef.slots;
export default class FilePreview extends SvelteComponent<FilePreviewProps, FilePreviewEvents, FilePreviewSlots> {
}
export {};
