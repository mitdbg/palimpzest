import { SvelteComponent } from "svelte";
import type { ThemeMode } from "js/core/src/components/types";
import type { Gradio, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: any;
        colors?: string[] | undefined;
        show_label: boolean;
        theme_mode: ThemeMode;
        caption: string;
        bokeh_version: string | null;
        show_actions_button: bool;
        gradio: Gradio<{
            select: SelectData;
        }>;
        x_lim?: ([number, number] | null) | undefined;
        _selectable: boolean;
    };
    events: {
        load: any;
        select: any;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type PlotProps = typeof __propDef.props;
export type PlotEvents = typeof __propDef.events;
export type PlotSlots = typeof __propDef.slots;
export default class Plot extends SvelteComponent<PlotProps, PlotEvents, PlotSlots> {
}
export {};
