import { Container, Graphics, Rectangle, type IRenderer, type ICanvas } from "pixi.js";
import { type LayerScene } from "../layers/utils";
/**
 * interface holding references to pixi app components
 */
export interface PixiApp {
    /**
     * The pixi container for layers
     */
    layer_container: Container;
    /**
     * The pixi container for background images and colors
     */
    background_container: Container;
    /**
     * The pixi renderer
     */
    renderer: IRenderer;
    /**
     * The pixi canvas
     */
    view: HTMLCanvasElement & ICanvas;
    /**
     * The pixi container for masking
     */
    mask_container: Container;
    destroy(): void;
    /**
     * Resizes the pixi app
     * @param width the new width
     * @param height the new height
     */
    resize(width: number, height: number): void;
    /**
     * Gets the blobs for the background, layers, and composite
     * @param bounds the bounds of the canvas
     * @returns a promise with the blobs
     */
    get_blobs(layers: LayerScene[], bounds: Rectangle, dimensions: [number, number]): Promise<ImageBlobs>;
    /**
     * Gets the layers
     */
    get_layers?: () => LayerScene[];
}
/**
 * Creates a PIXI app and attaches it to a DOM element
 * @param target DOM element to attach PIXI app to
 * @param width Width of the PIXI app
 * @param height Height of the PIXI app
 * @param antialias Whether to use antialiasing
 * @returns object with pixi container and renderer
 */
export declare function create_pixi_app({ target, dimensions: [width, height], antialias }: {
    target: HTMLElement;
    dimensions: [number, number];
    antialias: boolean;
}): PixiApp;
/**
 * Creates a pixi graphics object.
 * @param z_index the z index of the graphics object
 * @returns a graphics object
 */
export declare function make_graphics(z_index: number): Graphics;
/**
 * Clamps a number between a min and max value.
 * @param n The number to clamp.
 * @param min The minimum value.
 * @param max The maximum value.
 * @returns The clamped number.
 */
export declare function clamp(n: number, min: number, max: number): number;
export interface ImageBlobs {
    background: Blob | null;
    layers: (Blob | null)[];
    composite: Blob | null;
}
