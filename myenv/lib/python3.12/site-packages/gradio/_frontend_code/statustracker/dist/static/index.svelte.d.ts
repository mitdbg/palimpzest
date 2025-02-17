import { SvelteComponent } from "svelte";
import type { LoadingStatus } from "./types";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        i18n: I18nFormatter;
        eta?: (number | null) | undefined;
        queue_position: number | null;
        queue_size: number | null;
        status: "complete" | "pending" | "error" | "generating" | "streaming" | null;
        scroll_to_output?: boolean | undefined;
        timer?: boolean | undefined;
        show_progress?: ("full" | "minimal" | "hidden") | undefined;
        message?: (string | null) | undefined;
        progress?: LoadingStatus["progress"] | null | undefined;
        variant?: ("default" | "center") | undefined;
        loading_text?: string | undefined;
        absolute?: boolean | undefined;
        translucent?: boolean | undefined;
        border?: boolean | undefined;
        autoscroll: boolean;
    };
    events: {
        clear_status: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        'additional-loading-text': {};
        error: {};
    };
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
export {};
