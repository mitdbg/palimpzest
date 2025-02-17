import { SvelteComponent } from "svelte";
import type { KeyUpData, SelectData, I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        label: string;
        info?: string | undefined;
        value?: (string | number) | (string | number)[] | undefined;
        value_is_output?: boolean | undefined;
        max_choices?: (number | null) | undefined;
        choices: [string, string | number][];
        disabled?: boolean | undefined;
        show_label: boolean;
        container?: boolean | undefined;
        allow_custom_value?: boolean | undefined;
        filterable?: boolean | undefined;
        i18n: I18nFormatter;
        root: string;
    };
    events: {
        change: CustomEvent<string | string[] | undefined>;
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
export type MultiselectProps = typeof __propDef.props;
export type MultiselectEvents = typeof __propDef.events;
export type MultiselectSlots = typeof __propDef.slots;
export default class Multiselect extends SvelteComponent<MultiselectProps, MultiselectEvents, MultiselectSlots> {
}
export {};
