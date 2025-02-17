import { SvelteComponent } from "svelte";
import type { Writable } from "svelte/store";
import type { Spring } from "svelte/motion";
import { type PixiApp } from "./utils/pixi";
import { type CommandManager, type CommandNode } from "./utils/commands";
export declare const EDITOR_KEY: unique symbol;
export type context_type = "bg" | "layers" | "crop" | "draw" | "erase";
import { type tool } from "./tools";
export interface EditorContext {
    pixi: Writable<PixiApp | null>;
    current_layer: Writable<LayerScene | null>;
    dimensions: Writable<[number, number]>;
    editor_box: Writable<{
        parent_width: number;
        parent_height: number;
        parent_left: number;
        parent_top: number;
        parent_right: number;
        parent_bottom: number;
        child_width: number;
        child_height: number;
        child_left: number;
        child_top: number;
        child_right: number;
        child_bottom: number;
    }>;
    active_tool: Writable<tool>;
    toolbar_box: Writable<DOMRect | Record<string, never>>;
    crop: Writable<[number, number, number, number]>;
    position_spring: Spring<{
        x: number;
        y: number;
    }>;
    command_manager: CommandManager;
    current_history: CommandManager["current_history"];
    register_context: (type: context_type, { reset_fn, init_fn }: {
        reset_fn?: () => void;
        init_fn?: (dimensions?: [number, number]) => void;
    }) => void;
    reset: (clear_image: boolean, dimensions: [number, number]) => void;
}
import { type LayerScene } from "./layers/utils";
import { type ImageBlobs } from "./utils/pixi";
declare const __propDef: {
    props: {
        antialias?: boolean | undefined;
        crop_size: [number, number] | undefined;
        changeable?: boolean | undefined;
        history: boolean;
        bg?: boolean | undefined;
        sources: ("clipboard" | "webcam" | "upload")[];
        crop_constraint?: boolean | undefined;
        canvas_size: [number, number] | undefined;
        parent_height: number;
        full_history?: (CommandNode | null) | undefined;
        height?: number | undefined;
        canvas_height?: number | undefined;
        get_blobs?: (() => Promise<ImageBlobs>) | undefined;
        handle_remove?: (() => void) | undefined;
        set_tool?: ((tool: tool) => void) | undefined;
    };
    events: {
        clear?: CustomEvent<undefined> | undefined;
        save: CustomEvent<void>;
        change: CustomEvent<void>;
        history: CustomEvent<Writable<CommandNode>>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type ImageEditorProps = typeof __propDef.props;
export type ImageEditorEvents = typeof __propDef.events;
export type ImageEditorSlots = typeof __propDef.slots;
export default class ImageEditor extends SvelteComponent<ImageEditorProps, ImageEditorEvents, ImageEditorSlots> {
    get get_blobs(): () => Promise<ImageBlobs>;
    get handle_remove(): () => void;
    get set_tool(): (tool: tool) => void;
}
export {};
