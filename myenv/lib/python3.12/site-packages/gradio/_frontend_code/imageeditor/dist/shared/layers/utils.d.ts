import { Container, type IRenderer, RenderTexture, Sprite, Filter } from "pixi.js";
import { type Command } from "../utils/commands";
import { type Writable } from "svelte/store";
/**
 * GLSL Shader that takes two textures and erases the second texture from the first.
 */
export declare const erase_shader = "\nprecision highp float;\n\nuniform sampler2D uDrawingTexture;\nuniform sampler2D uEraserTexture;\n\nvarying vec2 vTextureCoord;\n\nvoid main(void) {\n\tvec4 drawingColor = texture2D(uDrawingTexture,vTextureCoord);\n\tvec4 eraserColor = texture2D(uEraserTexture, vTextureCoord);\n\n\t// Use the alpha of the eraser to determine how much to \"erase\" from the drawing\n\tfloat alpha = 1.0 - eraserColor.a;\n\tgl_FragColor = vec4(drawingColor.rgb * alpha, drawingColor.a * alpha);\n}";
/**
 * Interface holding data for a layer
 */
export interface LayerScene {
    /**
     * The texture used for tracking brush strokes.
     */
    draw_texture: RenderTexture;
    /**
     * The texture used for tracking eraser strokes.
     */
    erase_texture: RenderTexture;
    /**
     * The sprite used for displaying the composite of the draw and erase textures.
     */
    composite: Sprite;
    /**
     * The filter used for combining the draw and erase textures into a composite texture.
     */
    filter?: Filter;
}
/**
 * Interface for managing layers.
 */
interface LayerManager {
    /**
     * Adds a layer to the container.
     * @param layer The container to add the layer to.
     * @param renderer The renderer to use for the layer.
     * @param width the width of the layer
     * @param height the height of the layer
     */
    add_layer(container: Container, renderer: IRenderer, width: number, height: number, sprite?: Sprite): Command;
    /**
     * Swaps the layer with the layer above or below it.
     * @param layer The index layer to swap.
     * @param direction The direction to swap the layer.
     */
    swap_layers(layer: number, direction: "up" | "down"): LayerScene;
    /**
     * Changes the active layer.
     * @param layer The index of the layer to make active.
     */
    change_active_layer(layer: number): LayerScene;
    /**
     * Resizes the layers.
     * @param width The new width of the layers.
     * @param height The new height of the layers.
     */
    reset(): void;
    /**
     * Gets the layers.
     * @returns The layers.
     */
    get_layers(): LayerScene[];
    layers: Writable<LayerScene[]>;
    active_layer: Writable<LayerScene | null>;
    add_layer_from_blob(container: Container, renderer: IRenderer, blob: Blob, view: HTMLCanvasElement): Promise<Command>;
}
/**
 * Creates a layer manager.
 * @param canvas_resize a function to resize the canvas
 * @returns a layer manager
 */
export declare function layer_manager(): LayerManager;
export {};
