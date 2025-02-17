import { SvelteComponent } from "svelte";
export { default as BaseGallery } from "./shared/Gallery.svelte";
import type { GalleryImage, GalleryVideo } from "./types";
import type { Gradio, ShareData, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        loading_status: LoadingStatus;
        show_label: boolean;
        label: string;
        root: string;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: ((GalleryImage | GalleryVideo)[] | null) | undefined;
        file_types?: (string[] | null) | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        columns?: number | number[] | undefined;
        rows?: number | number[] | undefined;
        height?: (number | "auto") | undefined;
        preview: boolean;
        allow_preview?: boolean | undefined;
        selected_index?: (number | null) | undefined;
        object_fit?: ("contain" | "cover" | "fill" | "none" | "scale-down") | undefined;
        show_share_button?: boolean | undefined;
        interactive: boolean;
        show_download_button?: boolean | undefined;
        gradio: Gradio<{
            change: (GalleryImage | GalleryVideo)[] | null;
            upload: (GalleryImage | GalleryVideo)[] | null;
            select: SelectData;
            share: ShareData;
            error: string;
            prop_change: Record<string, any>;
            clear_status: LoadingStatus;
            preview_open: never;
            preview_close: never;
        }>;
        show_fullscreen_button?: boolean | undefined;
    };
    events: {
        prop_change: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
