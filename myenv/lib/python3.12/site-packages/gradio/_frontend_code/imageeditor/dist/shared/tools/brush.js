import { Container, Graphics, Color, RenderTexture, Sprite } from "pixi.js";
import {} from "../utils/commands";
import { make_graphics } from "../utils/pixi";
/**
 * Draws a circle on the given graphics object.
 * @param graphics the graphics object to draw on
 * @param x the x coordinate of the circle
 * @param y the y coordinate of the circle
 * @param brush_color the color of the circle
 * @param brush_size the radius of the circle
 */
function drawCircle(graphics, x, y, brush_color = new Color("black"), brush_size) {
    const color = new Color(brush_color);
    graphics.beginFill(color);
    graphics.drawCircle(x, y, brush_size);
    graphics.endFill();
}
/**
 * Interpolates between two points.
 * @param point1 the first point
 * @param point2 the second point
 * @returns an array of points between the two points
 */
function interpolate(point1, point2) {
    let points = [];
    const dx = point2.x - point1.x;
    const dy = point2.y - point1.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.ceil(distance / 2);
    const stepX = dx / steps;
    const stepY = dy / steps;
    for (let j = 0; j < steps; j++) {
        const x = point1.x + j * stepX;
        const y = point1.y + j * stepY;
        points.push({ x, y });
    }
    return points;
}
export function draw_path(renderer, stage, layer, mode) {
    const paths = [];
    let initial_path;
    let graphics;
    let InitialTexture;
    let has_drawn = false;
    let id = 0;
    return {
        drawing: false,
        start: function ({ x, y, size, color = new Color("black"), opacity, set_initial_texture = true }) {
            if (set_initial_texture) {
                InitialTexture = RenderTexture.create({
                    width: layer.draw_texture.width,
                    height: layer.draw_texture.height
                });
                renderer.render(layer.composite, {
                    renderTexture: InitialTexture
                });
            }
            initial_path = { x, y, size, color, opacity };
            paths.push({ x, y });
            graphics = make_graphics(id++);
            drawCircle(graphics, x, y, color, size);
            renderer.render(graphics, {
                renderTexture: mode === "draw" ? layer.draw_texture : layer.erase_texture,
                clear: false
            });
            this.drawing = true;
        },
        continue: function ({ x, y }) {
            const last_point = paths[paths.length - 1];
            const new_points = interpolate(last_point, { x, y });
            for (let i = 0; i < new_points.length; i++) {
                const { x, y } = new_points[i];
                drawCircle(graphics, x, y, initial_path.color, initial_path.size);
                paths.push({ x, y });
            }
            renderer.render(graphics, {
                renderTexture: mode === "draw" ? layer.draw_texture : layer.erase_texture,
                clear: false
            });
            graphics.clear();
        },
        stop: function () {
            const current_sketch = RenderTexture.create({
                width: layer.draw_texture.width,
                height: layer.draw_texture.height
            });
            renderer.render(layer.composite, {
                renderTexture: current_sketch
            });
            renderer.render(new Sprite(current_sketch), {
                renderTexture: layer.draw_texture
            });
            const clear_graphics = new Graphics()
                .beginFill(0x000000, 0) // Use a fill color with 0 alpha for transparency
                .drawRect(0, 0, layer.erase_texture.width, layer.erase_texture.height)
                .endFill();
            renderer.render(clear_graphics, {
                renderTexture: layer.erase_texture,
                clear: true
            });
            has_drawn = true;
            this.drawing = false;
        },
        execute: function () {
            if (!has_drawn) {
                for (let i = 1; i < paths.length; i++) {
                    const { x, y } = paths[i];
                    drawCircle(graphics, x, y, initial_path.color, initial_path.size);
                }
                renderer.render(graphics, {
                    renderTexture: mode === "draw" ? layer.draw_texture : layer.erase_texture,
                    clear: false
                });
                this.stop();
            }
        },
        undo: function () {
            const clear_graphics = new Graphics()
                .beginFill(0x000000, 0)
                .drawRect(0, 0, layer.erase_texture.width, layer.erase_texture.height)
                .endFill();
            renderer.render(new Sprite(InitialTexture), {
                renderTexture: layer.draw_texture
            });
            renderer.render(clear_graphics, {
                renderTexture: layer.erase_texture,
                clear: true
            });
            this.stop();
            has_drawn = false;
        }
    };
}
