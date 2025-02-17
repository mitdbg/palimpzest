import { SvelteComponent } from "svelte";
import { type FileData } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: null | FileData;
        label?: string | undefined;
        show_label: boolean;
        show_download_button?: boolean | undefined;
        selectable?: boolean | undefined;
        i18n: I18nFormatter;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ImagePreviewProps = typeof __propDef.props;
export type ImagePreviewEvents = typeof __propDef.events;
export type ImagePreviewSlots = typeof __propDef.slots;
export default class ImagePreview extends SvelteComponent<ImagePreviewProps, ImagePreviewEvents, ImagePreviewSlots> {
}
export {};
