import { Application, Container, Graphics, Sprite, Rectangle, RenderTexture } from "pixi.js";
import {} from "../layers/utils";
/**
 * Creates a PIXI app and attaches it to a DOM element
 * @param target DOM element to attach PIXI app to
 * @param width Width of the PIXI app
 * @param height Height of the PIXI app
 * @param antialias Whether to use antialiasing
 * @returns object with pixi container and renderer
 */
export function create_pixi_app({ target, dimensions: [width, height], antialias }) {
    const ratio = window.devicePixelRatio || 1;
    const app = new Application({
        width,
        height,
        antialias: antialias,
        backgroundAlpha: 0,
        eventMode: "static"
    });
    const view = app.view;
    // ensure that we can sort the background and layer containers
    app.stage.sortableChildren = true;
    view.style.maxWidth = `${width / ratio}px`;
    view.style.maxHeight = `${height / ratio}px`;
    view.style.width = "100%";
    view.style.height = "100%";
    target.appendChild(app.view);
    // we need a separate container for the background so that we can
    // clear its content without knowing too much about its children
    const background_container = new Container();
    background_container.zIndex = 0;
    const layer_container = new Container();
    layer_container.zIndex = 1;
    // ensure we can reorder  layers via zIndex
    layer_container.sortableChildren = true;
    const mask_container = new Container();
    mask_container.zIndex = 1;
    const composite_container = new Container();
    composite_container.zIndex = 0;
    mask_container.addChild(background_container);
    mask_container.addChild(layer_container);
    app.stage.addChild(mask_container);
    app.stage.addChild(composite_container);
    const mask = new Graphics();
    let text = RenderTexture.create({
        width,
        height
    });
    const sprite = new Sprite(text);
    mask_container.mask = sprite;
    app.render();
    function reset_mask(width, height) {
        background_container.removeChildren();
        mask.beginFill(0xffffff, 1);
        mask.drawRect(0, 0, width, height);
        mask.endFill();
        text = RenderTexture.create({
            width,
            height
        });
        app.renderer.render(mask, {
            renderTexture: text
        });
        const sprite = new Sprite(text);
        mask_container.mask = sprite;
    }
    function resize(width, height) {
        app.renderer.resize(width, height);
        view.style.maxWidth = `${width / ratio}px`;
        view.style.maxHeight = `${height / ratio}px`;
        reset_mask(width, height);
    }
    async function get_blobs(_layers, bounds, [w, h]) {
        const background = await get_canvas_blob(app.renderer, background_container, bounds, w, h);
        const layers = await Promise.all(_layers.map((layer) => get_canvas_blob(app.renderer, layer.composite, bounds, w, h)));
        const composite = await get_canvas_blob(app.renderer, mask_container, bounds, w, h);
        return {
            background,
            layers,
            composite
        };
    }
    return {
        layer_container,
        renderer: app.renderer,
        destroy: () => app.destroy(true),
        view: app.view,
        background_container,
        mask_container,
        resize,
        get_blobs
    };
}
/**
 * Creates a pixi graphics object.
 * @param z_index the z index of the graphics object
 * @returns a graphics object
 */
export function make_graphics(z_index) {
    const graphics = new Graphics();
    graphics.eventMode = "none";
    graphics.zIndex = z_index;
    return graphics;
}
/**
 * Clamps a number between a min and max value.
 * @param n The number to clamp.
 * @param min The minimum value.
 * @param max The maximum value.
 * @returns The clamped number.
 */
export function clamp(n, min, max) {
    return n < min ? min : n > max ? max : n;
}
/**
 * Generates a blob from a pixi object.
 * @param renderer The pixi renderer.
 * @param obj The pixi object to generate a blob from.
 * @param bounds The bounds of the canvas that we wish to extract
 * @param width The full width of the canvas
 * @param height The full height of the canvas
 * @returns A promise with the blob.
 */
function get_canvas_blob(renderer, obj, bounds, width, height) {
    return new Promise((resolve) => {
        // for some reason pixi won't extract a cropped canvas without distorting it
        // so we have to extract the whole canvas and crop it manually
        const src_canvas = renderer.extract.canvas(obj, new Rectangle(0, 0, width, height));
        // Create a new canvas for the cropped area with the appropriate size
        let dest_canvas = document.createElement("canvas");
        dest_canvas.width = bounds.width;
        dest_canvas.height = bounds.height;
        let dest_ctx = dest_canvas.getContext("2d");
        if (!dest_ctx) {
            resolve(null);
            throw new Error("Could not create canvas context");
        }
        // Draw the cropped area onto the destination canvas
        dest_ctx.drawImage(src_canvas, 
        // this is the area of the source that we want to copy (the crop box)
        bounds.x, bounds.y, bounds.width, bounds.height, 
        // this is where we want to draw the crop box on the destination canvas
        0, 0, bounds.width, bounds.height);
        // we grab a blob here so we can upload it
        dest_canvas.toBlob?.((blob) => {
            if (!blob) {
                resolve(null);
            }
            resolve(blob);
        });
    });
}
