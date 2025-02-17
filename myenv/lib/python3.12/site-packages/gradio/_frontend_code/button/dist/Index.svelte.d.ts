import { SvelteComponent } from "svelte";
export { default as BaseButton } from "./shared/Button.svelte";
import type { Gradio } from "@gradio/utils";
import { type FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: string | null;
        variant?: ("primary" | "secondary" | "stop") | undefined;
        interactive: boolean;
        size?: ("sm" | "lg") | undefined;
        scale?: (number | null) | undefined;
        icon?: FileData | null;
        link?: (string | null) | undefined;
        min_width?: number | undefined;
        gradio: Gradio<{
            click: never;
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
}
