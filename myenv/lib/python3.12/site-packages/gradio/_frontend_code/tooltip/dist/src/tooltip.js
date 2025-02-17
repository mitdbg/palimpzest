import Tooltip from "./Tooltip.svelte";
export function tooltip(element, { color, text }) {
    let tooltipComponent;
    function mouse_over(event) {
        tooltipComponent = new Tooltip({
            props: {
                text,
                x: event.pageX,
                y: event.pageY,
                color
            },
            target: document.body
        });
        return event;
    }
    function mouseMove(event) {
        tooltipComponent.$set({
            x: event.pageX,
            y: event.pageY
        });
    }
    function mouseLeave() {
        tooltipComponent.$destroy();
    }
    const el = element;
    el.addEventListener("mouseover", mouse_over);
    el.addEventListener("mouseleave", mouseLeave);
    el.addEventListener("mousemove", mouseMove);
    return {
        destroy() {
            el.removeEventListener("mouseover", mouse_over);
            el.removeEventListener("mouseleave", mouseLeave);
            el.removeEventListener("mousemove", mouseMove);
        }
    };
}
