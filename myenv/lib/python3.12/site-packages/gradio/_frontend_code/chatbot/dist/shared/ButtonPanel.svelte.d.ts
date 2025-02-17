import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import type { NormalisedMessage } from "../types";
declare const __propDef: {
    props: {
        likeable: boolean;
        feedback_options: string[];
        show_retry: boolean;
        show_undo: boolean;
        show_edit: boolean;
        in_edit_mode: boolean;
        show_copy_button: boolean;
        message: NormalisedMessage | NormalisedMessage[];
        position: "right" | "left";
        avatar: FileData | null;
        generating: boolean;
        current_feedback: string | null;
        handle_action: (selected: string | null) => void;
        layout: "bubble" | "panel";
        dispatch: any;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ButtonPanelProps = typeof __propDef.props;
export type ButtonPanelEvents = typeof __propDef.events;
export type ButtonPanelSlots = typeof __propDef.slots;
export default class ButtonPanel extends SvelteComponent<ButtonPanelProps, ButtonPanelEvents, ButtonPanelSlots> {
}
export {};
