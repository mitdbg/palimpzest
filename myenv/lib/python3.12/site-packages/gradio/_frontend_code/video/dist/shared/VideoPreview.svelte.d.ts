import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { I18nFormatter } from "js/core/src/gradio_helper";
declare const __propDef: {
    props: {
        value?: FileData | null;
        subtitle?: FileData | null;
        label?: string | undefined;
        show_label?: boolean | undefined;
        autoplay: boolean;
        show_share_button?: boolean | undefined;
        show_download_button?: boolean | undefined;
        loop: boolean;
        i18n: I18nFormatter;
        upload: Client["upload"];
        display_icon_button_wrapper_top_corner?: boolean | undefined;
    };
    events: {
        play: CustomEvent<any>;
        pause: CustomEvent<any>;
        stop: CustomEvent<any>;
        end: CustomEvent<any>;
        error: CustomEvent<string>;
        share: CustomEvent<import("@gradio/utils").ShareData>;
        change: CustomEvent<FileData>;
        load: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type VideoPreviewProps = typeof __propDef.props;
export type VideoPreviewEvents = typeof __propDef.events;
export type VideoPreviewSlots = typeof __propDef.slots;
export default class VideoPreview extends SvelteComponent<VideoPreviewProps, VideoPreviewEvents, VideoPreviewSlots> {
}
export {};
