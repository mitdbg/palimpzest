import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
import type { GalleryImage, GalleryVideo } from "../types";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        show_label?: boolean | undefined;
        label: string;
        value?: ((GalleryImage | GalleryVideo)[] | null) | undefined;
        columns?: number | number[] | undefined;
        rows?: number | number[] | undefined;
        height?: (number | "auto") | undefined;
        preview: boolean;
        allow_preview?: boolean | undefined;
        object_fit?: ("contain" | "cover" | "fill" | "none" | "scale-down") | undefined;
        show_share_button?: boolean | undefined;
        show_download_button?: boolean | undefined;
        i18n: I18nFormatter;
        selected_index?: (number | null) | undefined;
        interactive: boolean;
        _fetch: typeof fetch;
        mode?: ("normal" | "minimal") | undefined;
        show_fullscreen_button?: boolean | undefined;
        display_icon_button_wrapper_top_corner?: boolean | undefined;
    };
    events: {
        share: CustomEvent<import("@gradio/utils").ShareData>;
        error: CustomEvent<string>;
        change: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
        preview_open: CustomEvent<undefined>;
        preview_close: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type GalleryProps = typeof __propDef.props;
export type GalleryEvents = typeof __propDef.events;
export type GallerySlots = typeof __propDef.slots;
export default class Gallery extends SvelteComponent<GalleryProps, GalleryEvents, GallerySlots> {
}
export {};
