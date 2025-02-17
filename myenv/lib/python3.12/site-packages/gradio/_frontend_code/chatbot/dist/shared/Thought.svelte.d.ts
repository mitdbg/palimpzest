import { SvelteComponent } from "svelte";
import type { Client } from "@gradio/client";
import type { NormalisedMessage } from "../types";
import type { I18nFormatter } from "js/core/src/gradio_helper";
import type { ComponentType } from "svelte";
declare const __propDef: {
    props: {
        thought: NormalisedMessage;
        rtl?: boolean | undefined;
        sanitize_html: boolean;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        render_markdown: boolean;
        _components: Record<string, ComponentType<SvelteComponent>>;
        upload: Client["upload"];
        thought_index: number;
        target: HTMLElement | null;
        root: string;
        theme_mode: "light" | "dark" | "system";
        _fetch: typeof fetch;
        scroll: () => void;
        allow_file_downloads: boolean;
        display_consecutive_in_same_bubble: boolean;
        i18n: I18nFormatter;
        line_breaks: boolean;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ThoughtProps = typeof __propDef.props;
export type ThoughtEvents = typeof __propDef.events;
export type ThoughtSlots = typeof __propDef.slots;
export default class Thought extends SvelteComponent<ThoughtProps, ThoughtEvents, ThoughtSlots> {
}
export {};
