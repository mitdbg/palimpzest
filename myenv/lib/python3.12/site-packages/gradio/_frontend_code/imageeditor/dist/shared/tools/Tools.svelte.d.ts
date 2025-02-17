import { SvelteComponent } from "svelte";
import type { Subscriber, Invalidator, Unsubscriber, Writable } from "svelte/store";
import type { tool } from "../tools/";
export declare const TOOL_KEY: unique symbol;
type upload_tool = "bg_webcam" | "bg_upload" | "bg_clipboard" | "bg_color";
type transform_tool = "crop" | "rotate";
type brush_tool = "brush_color" | "brush_size";
type eraser_tool = "eraser_size";
export interface ToolOptions {
    order: number;
    label: string;
    icon: typeof Image;
    cb: (...args: any[]) => void;
    id: upload_tool | transform_tool | brush_tool | eraser_tool;
}
export interface ToolMeta {
    default: upload_tool | transform_tool | brush_tool | eraser_tool | null;
    options: ToolOptions[];
}
export interface ToolContext {
    register_tool: (type: tool, opts?: {
        cb: () => void;
    }) => () => void;
    active_tool: {
        set: (tool: tool) => void;
        subscribe(this: void, run: Subscriber<tool | null>, invalidate?: Invalidator<tool | null>): Unsubscriber;
    };
    current_color: Writable<string>;
}
import { Image } from "@gradio/icons";
import { type I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        i18n: I18nFormatter;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type ToolsProps = typeof __propDef.props;
export type ToolsEvents = typeof __propDef.events;
export type ToolsSlots = typeof __propDef.slots;
export default class Tools extends SvelteComponent<ToolsProps, ToolsEvents, ToolsSlots> {
}
export {};
