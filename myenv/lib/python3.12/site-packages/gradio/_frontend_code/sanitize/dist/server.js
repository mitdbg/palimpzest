import { default as sanitize_html_ } from "sanitize-html";
export function sanitize(source, root) {
    return sanitize_html_(source);
}
