import { type Container, type IRenderer, type ColorSource } from "pixi.js";
import { type Command } from "../utils/commands";
interface BgImageCommand extends Command {
    /**
     * Initial setup for the bg command
     * @returns
     */
    start: () => Promise<[number, number]>;
}
/**
 * Calculates new dimensions and position for an image to fit within a canvas while maintaining aspect ratio
 * @param image_width Original width of the image
 * @param image_height Original height of the image
 * @param canvas_width Width of the canvas
 * @param canvas_height Height of the canvas
 * @returns Object containing new dimensions and position
 */
export declare function fit_image_to_canvas(image_width: number, image_height: number, canvas_width: number, canvas_height: number): {
    width: number;
    height: number;
    x: number;
    y: number;
};
export declare function add_bg_image(container: Container, renderer: IRenderer, background: Blob | File, resize: (width: number, height: number) => void, canvas_size: [number, number], fixed_canvas: boolean): BgImageCommand;
/**
 * Command that sets a background
 */
interface BgColorCommand extends Command {
    /**
     * Initial setup for the bg command
     * @returns
     */
    start: () => [number, number];
}
/**
 * Adds a background color to the canvas.
 * @param container The container to add the image to.
 * @param renderer The renderer to use for the image.
 * @param color The background color to add.
 * @param width The width of the background.
 * @param height The height of the background.
 * @param resize The function to resize the canvas.
 * @returns A command that can be used to undo the action.
 */
export declare function add_bg_color(container: Container, renderer: IRenderer, color: ColorSource, width: number, height: number, resize: (width: number, height: number) => void): BgColorCommand;
export {};
