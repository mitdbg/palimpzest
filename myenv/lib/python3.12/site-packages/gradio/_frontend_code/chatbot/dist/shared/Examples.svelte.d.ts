import { SvelteComponent } from "svelte";
import type { ExampleMessage } from "../types";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        examples?: (ExampleMessage[] | null) | undefined;
        placeholder?: (string | null) | undefined;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        root: string;
    };
    events: {
        example_select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ExamplesProps = typeof __propDef.props;
export type ExamplesEvents = typeof __propDef.events;
export type ExamplesSlots = typeof __propDef.slots;
export default class Examples extends SvelteComponent<ExamplesProps, ExamplesEvents, ExamplesSlots> {
}
export {};
