import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        type?: ("video" | "image" | "audio" | "file" | "csv" | "clipboard" | "gallery") | undefined;
        i18n: I18nFormatter;
        message?: string | undefined;
        mode?: ("full" | "short") | undefined;
        hovered?: boolean | undefined;
        placeholder?: string | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type UploadTextProps = typeof __propDef.props;
export type UploadTextEvents = typeof __propDef.events;
export type UploadTextSlots = typeof __propDef.slots;
export default class UploadText extends SvelteComponent<UploadTextProps, UploadTextEvents, UploadTextSlots> {
}
export {};
