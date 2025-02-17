import { SvelteComponent } from "svelte";
import { type Extension } from "@codemirror/state";
declare const __propDef: {
    props: {
        class_names?: string | undefined;
        value?: string | undefined;
        dark_mode: boolean;
        basic?: boolean | undefined;
        language: string;
        lines?: number | undefined;
        max_lines?: (number | null) | undefined;
        extensions?: Extension[] | undefined;
        use_tab?: boolean | undefined;
        readonly?: boolean | undefined;
        placeholder?: string | HTMLElement | null | undefined;
        wrap_lines?: boolean | undefined;
    };
    events: {
        change: CustomEvent<string>;
        blur: CustomEvent<undefined>;
        focus: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CodeProps = typeof __propDef.props;
export type CodeEvents = typeof __propDef.events;
export type CodeSlots = typeof __propDef.slots;
export default class Code extends SvelteComponent<CodeProps, CodeEvents, CodeSlots> {
}
export {};
