import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
import type { I18nFormatter } from "@gradio/utils";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        value: null | FileData;
        label?: string | undefined;
        show_label: boolean;
        show_download_button?: boolean | undefined;
        selectable?: boolean | undefined;
        show_share_button?: boolean | undefined;
        i18n: I18nFormatter;
        show_fullscreen_button?: boolean | undefined;
        display_icon_button_wrapper_top_corner?: boolean | undefined;
    };
    events: {
        share: CustomEvent<import("@gradio/utils").ShareData>;
        error: CustomEvent<string>;
        load: Event;
        change: CustomEvent<string>;
        select: CustomEvent<SelectData>;
    } & {
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
