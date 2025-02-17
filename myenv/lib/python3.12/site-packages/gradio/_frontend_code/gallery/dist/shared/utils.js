import { uploadToHuggingFace } from "@gradio/utils";
export async function format_gallery_for_sharing(value) {
    if (!value)
        return "";
    let urls = await Promise.all(value.map(async ([image, _]) => {
        if (image === null || !image.url)
            return "";
        return await uploadToHuggingFace(image.url, "url");
    }));
    return `<div style="display: flex; flex-wrap: wrap; gap: 16px">${urls
        .map((url) => `<img src="${url}" style="height: 400px" />`)
        .join("")}</div>`;
}
