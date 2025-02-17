import { type IRenderer, type Container } from "pixi.js";
import { type Command } from "../utils/commands";
import type { Writable } from "svelte/store";
export interface CropCommand extends Command {
    start: (width: number, height: number, previous_crop: [number, number, number, number], preview?: boolean, set_previous?: boolean) => void;
    stop: () => number;
    continue: (crop_size: [number, number, number, number], preview?: boolean) => void;
}
export declare function crop_canvas(renderer: IRenderer, background_container: Container, crop: Writable<[number, number, number, number]>, current_opacity?: number): CropCommand;
export declare function resize_and_reposition(original_width: number, original_height: number, anchor: "t" | "r" | "l" | "b" | "tl" | "tr" | "bl" | "br" | "c", aspect_ratio: number, max_width: number, max_height: number): {
    new_width: number;
    new_height: number;
    x_offset: number;
    y_offset: number;
};
