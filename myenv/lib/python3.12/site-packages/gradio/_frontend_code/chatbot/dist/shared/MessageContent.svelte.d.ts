import { SvelteComponent } from "svelte";
import type { NormalisedMessage } from "../types";
import type { I18nFormatter } from "js/core/src/gradio_helper";
import type { Client } from "@gradio/client";
import type { ComponentType } from "svelte";
declare const __propDef: {
    props: {
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        sanitize_html: boolean;
        _fetch: typeof fetch;
        i18n: I18nFormatter;
        line_breaks: boolean;
        upload: Client["upload"];
        target: HTMLElement | null;
        root: string;
        theme_mode: "light" | "dark" | "system";
        _components: Record<string, ComponentType<SvelteComponent>>;
        render_markdown: boolean;
        scroll: () => void;
        allow_file_downloads: boolean;
        display_consecutive_in_same_bubble: boolean;
        thought_index: number;
        message: NormalisedMessage;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type MessageContentProps = typeof __propDef.props;
export type MessageContentEvents = typeof __propDef.events;
export type MessageContentSlots = typeof __propDef.slots;
export default class MessageContent extends SvelteComponent<MessageContentProps, MessageContentEvents, MessageContentSlots> {
}
export {};
