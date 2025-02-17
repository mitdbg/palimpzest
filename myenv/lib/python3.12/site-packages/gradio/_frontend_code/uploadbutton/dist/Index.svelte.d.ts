import { SvelteComponent } from "svelte";
export { default as BaseUploadButton } from "./shared/UploadButton.svelte";
import type { Gradio } from "@gradio/utils";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        label: string | null;
        value: null | FileData | FileData[];
        file_count: string;
        file_types?: string[] | undefined;
        root: string;
        size?: ("sm" | "lg") | undefined;
        scale?: (number | null) | undefined;
        icon?: FileData | null;
        min_width?: number | undefined;
        variant?: ("primary" | "secondary" | "stop") | undefined;
        gradio: Gradio<{
            change: never;
            upload: never;
            click: never;
            error: string;
        }>;
        interactive: boolean;
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
}
