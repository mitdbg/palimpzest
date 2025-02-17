import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: any;
        bokeh_version: string | null;
    };
    events: {
        load: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type BokehPlotProps = typeof __propDef.props;
export type BokehPlotEvents = typeof __propDef.events;
export type BokehPlotSlots = typeof __propDef.slots;
export default class BokehPlot extends SvelteComponent<BokehPlotProps, BokehPlotEvents, BokehPlotSlots> {
}
export {};
