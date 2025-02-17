import { SvelteComponent } from "svelte";
import type { Gradio, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: any;
        colors?: string[] | undefined;
        caption: string;
        show_actions_button: bool;
        gradio: Gradio<{
            select: SelectData;
        }>;
        _selectable: bool;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type AltairPlotProps = typeof __propDef.props;
export type AltairPlotEvents = typeof __propDef.events;
export type AltairPlotSlots = typeof __propDef.slots;
export default class AltairPlot extends SvelteComponent<AltairPlotProps, AltairPlotEvents, AltairPlotSlots> {
}
export {};
