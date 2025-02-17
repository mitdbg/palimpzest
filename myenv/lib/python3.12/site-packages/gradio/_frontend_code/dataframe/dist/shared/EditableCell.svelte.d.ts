import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        edit: boolean;
        value?: (string | number) | undefined;
        display_value?: (string | null) | undefined;
        styling?: string | undefined;
        header?: boolean | undefined;
        datatype?: ("str" | "markdown" | "html" | "number" | "bool" | "date") | undefined;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        clear_on_focus?: boolean | undefined;
        line_breaks?: boolean | undefined;
        editable?: boolean | undefined;
        root: string;
        el: HTMLInputElement | null;
    };
    events: {
        mousedown: MouseEvent;
        mouseup: MouseEvent;
        click: MouseEvent;
        dblclick: MouseEvent;
        focus: FocusEvent;
        blur: CustomEvent<any>;
        keydown: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type EditableCellProps = typeof __propDef.props;
export type EditableCellEvents = typeof __propDef.events;
export type EditableCellSlots = typeof __propDef.slots;
export default class EditableCell extends SvelteComponent<EditableCellProps, EditableCellEvents, EditableCellSlots> {
}
export {};
