import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: {
            token: string;
            class_or_confidence: string | number | null;
        }[];
        category: string | number | null;
        active: string;
        labelToEdit: number;
        indexOfLabel: number;
        text: string;
        handleValueChange: () => void;
        isScoresMode?: boolean | undefined;
        _color_map: Record<string, {
            primary: string;
            secondary: string;
        }>;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type LabelInputProps = typeof __propDef.props;
export type LabelInputEvents = typeof __propDef.events;
export type LabelInputSlots = typeof __propDef.slots;
export default class LabelInput extends SvelteComponent<LabelInputProps, LabelInputEvents, LabelInputSlots> {
}
export {};
