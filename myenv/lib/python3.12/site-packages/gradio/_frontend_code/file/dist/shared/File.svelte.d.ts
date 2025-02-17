import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: FileData | FileData[] | null;
        label: string;
        show_label?: boolean | undefined;
        selectable?: boolean | undefined;
        height?: number | undefined;
        i18n: I18nFormatter;
    };
    events: {
        select: CustomEvent<import("@gradio/utils").SelectData>;
        download: CustomEvent<FileData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type FileProps = typeof __propDef.props;
export type FileEvents = typeof __propDef.events;
export type FileSlots = typeof __propDef.slots;
export default class File extends SvelteComponent<FileProps, FileEvents, FileSlots> {
}
export {};
