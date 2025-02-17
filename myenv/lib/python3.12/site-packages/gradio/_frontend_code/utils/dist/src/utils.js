export class ShareError extends Error {
    constructor(message) {
        super(message);
        this.name = "ShareError";
    }
}
export async function uploadToHuggingFace(data, type) {
    if (window.__gradio_space__ == null) {
        throw new ShareError("Must be on Spaces to share.");
    }
    let blob;
    let contentType;
    let filename;
    if (type === "url") {
        let url;
        if (typeof data === "object" && data.url) {
            url = data.url;
        }
        else if (typeof data === "string") {
            url = data;
        }
        else {
            throw new Error("Invalid data format for URL type");
        }
        const response = await fetch(url);
        blob = await response.blob();
        contentType = response.headers.get("content-type") || "";
        filename = response.headers.get("content-disposition") || "";
    }
    else {
        let dataurl;
        if (typeof data === "object" && data.path) {
            dataurl = data.path;
        }
        else if (typeof data === "string") {
            dataurl = data;
        }
        else {
            throw new Error("Invalid data format for base64 type");
        }
        blob = dataURLtoBlob(dataurl);
        contentType = dataurl.split(";")[0].split(":")[1];
        filename = "file." + contentType.split("/")[1];
    }
    const file = new File([blob], filename, { type: contentType });
    // Send file to endpoint
    const uploadResponse = await fetch("https://huggingface.co/uploads", {
        method: "POST",
        body: file,
        headers: {
            "Content-Type": file.type,
            "X-Requested-With": "XMLHttpRequest"
        }
    });
    // Check status of response
    if (!uploadResponse.ok) {
        if (uploadResponse.headers.get("content-type")?.includes("application/json")) {
            const error = await uploadResponse.json();
            throw new ShareError(`Upload failed: ${error.error}`);
        }
        throw new ShareError(`Upload failed.`);
    }
    // Return response if needed
    const result = await uploadResponse.text();
    return result;
}
function dataURLtoBlob(dataurl) {
    var arr = dataurl.split(","), mime = arr[0].match(/:(.*?);/)[1], bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
}
export function copy(node) {
    node.addEventListener("click", handle_copy);
    async function handle_copy(event) {
        const path = event.composedPath();
        const [copy_button] = path.filter((e) => e?.tagName === "BUTTON" && e.classList.contains("copy_code_button"));
        if (copy_button) {
            event.stopImmediatePropagation();
            const copy_text = copy_button.parentElement.innerText.trim();
            const copy_sucess_button = Array.from(copy_button.children)[1];
            const copied = await copy_to_clipboard(copy_text);
            if (copied)
                copy_feedback(copy_sucess_button);
            function copy_feedback(_copy_sucess_button) {
                _copy_sucess_button.style.opacity = "1";
                setTimeout(() => {
                    _copy_sucess_button.style.opacity = "0";
                }, 2000);
            }
        }
    }
    return {
        destroy() {
            node.removeEventListener("click", handle_copy);
        }
    };
}
async function copy_to_clipboard(value) {
    let copied = false;
    if ("clipboard" in navigator) {
        await navigator.clipboard.writeText(value);
        copied = true;
    }
    else {
        const textArea = document.createElement("textarea");
        textArea.value = value;
        textArea.style.position = "absolute";
        textArea.style.left = "-999999px";
        document.body.prepend(textArea);
        textArea.select();
        try {
            document.execCommand("copy");
            copied = true;
        }
        catch (error) {
            console.error(error);
            copied = false;
        }
        finally {
            textArea.remove();
        }
    }
    return copied;
}
export const format_time = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const seconds_remainder = Math.round(seconds) % 60;
    const padded_minutes = `${minutes < 10 ? "0" : ""}${minutes}`;
    const padded_seconds = `${seconds_remainder < 10 ? "0" : ""}${seconds_remainder}`;
    if (hours > 0) {
        return `${hours}:${padded_minutes}:${padded_seconds}`;
    }
    return `${minutes}:${padded_seconds}`;
};
const is_browser = typeof window !== "undefined";
export class Gradio {
    #id;
    theme;
    version;
    i18n;
    #el;
    root;
    autoscroll;
    max_file_size;
    client;
    _load_component;
    load_component = _load_component.bind(this);
    constructor(id, el, theme, version, root, autoscroll, max_file_size, i18n = (x) => x, client, virtual_component_loader) {
        this.#id = id;
        this.theme = theme;
        this.version = version;
        this.#el = el;
        this.max_file_size = max_file_size;
        this.i18n = i18n;
        this.root = root;
        this.autoscroll = autoscroll;
        this.client = client;
        this._load_component = virtual_component_loader;
    }
    dispatch(event_name, data) {
        if (!is_browser || !this.#el)
            return;
        const e = new CustomEvent("gradio", {
            bubbles: true,
            detail: { data, id: this.#id, event: event_name }
        });
        this.#el.dispatchEvent(e);
    }
}
function _load_component(name, variant = "component") {
    return this._load_component({
        name,
        api_url: this.client.config?.root,
        variant
    });
}
export const css_units = (dimension_value) => {
    return typeof dimension_value === "number"
        ? dimension_value + "px"
        : dimension_value;
};
