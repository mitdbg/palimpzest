import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: {
            label?: string;
            confidences?: {
                label: string;
                confidence: number;
            }[];
        };
        color?: string | undefined;
        selectable?: boolean | undefined;
        show_heading?: boolean | undefined;
    };
    events: {
        select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type LabelProps = typeof __propDef.props;
export type LabelEvents = typeof __propDef.events;
export type LabelSlots = typeof __propDef.slots;
export default class Label extends SvelteComponent<LabelProps, LabelEvents, LabelSlots> {
}
export {};
