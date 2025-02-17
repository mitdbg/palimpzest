import { SvelteComponent } from "svelte";
import type { SelectData, KeyUpData } from "@gradio/utils";
declare const __propDef: {
    props: {
        label: string;
        info?: string | undefined;
        value?: (string | number) | (string | number)[] | undefined;
        value_is_output?: boolean | undefined;
        choices: [string, string | number][];
        disabled?: boolean | undefined;
        show_label: boolean;
        container?: boolean | undefined;
        allow_custom_value?: boolean | undefined;
        filterable?: boolean | undefined;
        root: string;
    };
    events: {
        change: CustomEvent<string | undefined>;
        input: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
        blur: CustomEvent<undefined>;
        focus: CustomEvent<undefined>;
        key_up: CustomEvent<KeyUpData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type DropdownProps = typeof __propDef.props;
export type DropdownEvents = typeof __propDef.events;
export type DropdownSlots = typeof __propDef.slots;
export default class Dropdown extends SvelteComponent<DropdownProps, DropdownEvents, DropdownSlots> {
}
export {};
