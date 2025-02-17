import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: boolean | undefined;
        label?: string | undefined;
        interactive: boolean;
    };
    events: {
        change: CustomEvent<boolean>;
        select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CheckboxProps = typeof __propDef.props;
export type CheckboxEvents = typeof __propDef.events;
export type CheckboxSlots = typeof __propDef.slots;
export default class Checkbox extends SvelteComponent<CheckboxProps, CheckboxEvents, CheckboxSlots> {
}
export {};
