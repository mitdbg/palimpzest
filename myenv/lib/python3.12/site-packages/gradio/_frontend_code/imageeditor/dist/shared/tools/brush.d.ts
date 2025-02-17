import { Container, type ColorSource, type IRenderer } from "pixi.js";
import type { LayerScene } from "../layers/utils";
import { type Command } from "../utils/commands";
export interface DrawCommand extends Command {
    /**
     * sets the initial state to begin drawing
     * @param options the options for drawing
     * @returns
     */
    start: (options: DrawOptions) => void;
    /**
     * continues drawing, smoothly interpolating between points where necessary
     * @param options the options for drawing
     * @returns
     */
    continue: (options: Points) => void;
    /**
     * stops drawing
     * @returns
     */
    stop: () => void;
    /**
     * Whether or not the user is currently drawing.
     */
    drawing: boolean;
}
/**
 * Coordinates for a draw or erase path.
 */
interface Points {
    /**
     * The x coordinate of the path.
     */
    x: number;
    /**
     * The y coordinate of the path.
     */
    y: number;
}
/**
 * Options for drawing a path.
 */
interface DrawOptions extends Points {
    /**
     * The size of the brush.
     */
    size: number;
    /**
     * The color of the brush.
     */
    color?: ColorSource;
    /**
     * The opacity of the brush.
     */
    opacity: number;
    /**
     * Whether or not to set the initial texture.
     */
    set_initial_texture?: boolean;
}
export declare function draw_path(renderer: IRenderer, stage: Container, layer: LayerScene, mode: "draw" | "erase"): DrawCommand;
export {};
