import { SvelteComponent } from "svelte";
import "./prism.css";
import "prismjs/components/prism-python";
import "prismjs/components/prism-typescript";
declare const __propDef: {
    props: {
        docs: Record<string, {
            type: string | null;
            description: string;
            default: string | null;
            name?: string;
        }>;
        lang?: ("python" | "typescript") | undefined;
        linkify?: string[] | undefined;
        header: string | null;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ParamViewerProps = typeof __propDef.props;
export type ParamViewerEvents = typeof __propDef.events;
export type ParamViewerSlots = typeof __propDef.slots;
export default class ParamViewer extends SvelteComponent<ParamViewerProps, ParamViewerEvents, ParamViewerSlots> {
}
export {};
