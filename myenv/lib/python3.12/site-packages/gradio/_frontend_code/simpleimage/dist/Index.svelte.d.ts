import { SvelteComponent } from "svelte";
export { default as BaseImageUploader } from "./shared/ImageUploader.svelte";
export { default as BaseStaticImage } from "./shared/ImagePreview.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio } from "@gradio/utils";
import type { FileData } from "@gradio/client";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: null | FileData;
        label: string;
        show_label: boolean;
        show_download_button: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        interactive: boolean;
        root: string;
        placeholder?: string | undefined;
        gradio: Gradio<{
            change: never;
            upload: never;
            clear: never;
            clear_status: LoadingStatus;
        }>;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get value(): any;
    /**accessor*/
    set value(_: any);
    get label(): string;
    /**accessor*/
    set label(_: string);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get show_download_button(): boolean;
    /**accessor*/
    set show_download_button(_: boolean);
    get container(): boolean | undefined;
    /**accessor*/
    set container(_: boolean | undefined);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get loading_status(): LoadingStatus;
    /**accessor*/
    set loading_status(_: LoadingStatus);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get placeholder(): string | undefined;
    /**accessor*/
    set placeholder(_: string | undefined);
    get gradio(): Gradio<{
        change: never;
        upload: never;
        clear: never;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        upload: never;
        clear: never;
        clear_status: LoadingStatus;
    }>);
}
