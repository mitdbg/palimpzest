import { SvelteComponent } from "svelte";
export { default as BaseDropdown } from "./shared/Dropdown.svelte";
export { default as BaseMultiselect } from "./shared/Multiselect.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio, KeyUpData, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        label?: string | undefined;
        info?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        multiselect?: boolean | undefined;
        value?: (string | number) | (string | number)[] | undefined;
        value_is_output?: boolean | undefined;
        max_choices?: (number | null) | undefined;
        choices: [string, string | number][];
        show_label: boolean;
        filterable: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        allow_custom_value?: boolean | undefined;
        root: string;
        gradio: Gradio<{
            change: never;
            input: never;
            select: SelectData;
            blur: never;
            focus: never;
            key_up: KeyUpData;
            clear_status: LoadingStatus;
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
