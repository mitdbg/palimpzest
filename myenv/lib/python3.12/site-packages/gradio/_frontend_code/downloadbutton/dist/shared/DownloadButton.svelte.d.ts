import { SvelteComponent } from "svelte";
import { type FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        variant?: ("primary" | "secondary" | "stop") | undefined;
        size?: ("sm" | "md" | "lg") | undefined;
        value: null | FileData;
        icon: null | FileData;
        disabled?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
    };
    events: {
        click: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type DownloadButtonProps = typeof __propDef.props;
export type DownloadButtonEvents = typeof __propDef.events;
export type DownloadButtonSlots = typeof __propDef.slots;
export default class DownloadButton extends SvelteComponent<DownloadButtonProps, DownloadButtonEvents, DownloadButtonSlots> {
}
export {};
