import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        can_undo?: boolean | undefined;
        can_redo?: boolean | undefined;
        can_save?: boolean | undefined;
        changeable?: boolean | undefined;
    };
    events: {
        remove_image: CustomEvent<void>;
        undo: CustomEvent<void>;
        redo: CustomEvent<void>;
        save: CustomEvent<void>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ControlsProps = typeof __propDef.props;
export type ControlsEvents = typeof __propDef.events;
export type ControlsSlots = typeof __propDef.slots;
export default class Controls extends SvelteComponent<ControlsProps, ControlsEvents, ControlsSlots> {
}
export {};
