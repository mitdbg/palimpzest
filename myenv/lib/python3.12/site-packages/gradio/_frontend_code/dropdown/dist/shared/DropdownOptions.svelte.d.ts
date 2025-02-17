import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        choices: [string, string | number][];
        filtered_indices: number[];
        show_options?: boolean | undefined;
        disabled?: boolean | undefined;
        selected_indices?: (string | number)[] | undefined;
        active_index?: (number | null) | undefined;
    };
    events: {
        change: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type DropdownOptionsProps = typeof __propDef.props;
export type DropdownOptionsEvents = typeof __propDef.events;
export type DropdownOptionsSlots = typeof __propDef.slots;
export default class DropdownOptions extends SvelteComponent<DropdownOptionsProps, DropdownOptionsEvents, DropdownOptionsSlots> {
}
export {};
