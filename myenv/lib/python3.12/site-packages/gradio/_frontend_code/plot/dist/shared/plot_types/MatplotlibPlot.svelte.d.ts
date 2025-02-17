import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: any;
    };
    events: {
        load: Event;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type MatplotlibPlotProps = typeof __propDef.props;
export type MatplotlibPlotEvents = typeof __propDef.events;
export type MatplotlibPlotSlots = typeof __propDef.slots;
export default class MatplotlibPlot extends SvelteComponent<MatplotlibPlotProps, MatplotlibPlotEvents, MatplotlibPlotSlots> {
}
export {};
